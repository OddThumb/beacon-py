# ================================================================
# Built-in and Standard Library
import warnings  # For suppressing warnings during KPSS
from typing import (
    Union,
    Optional,
    Dict,
    Any,
    Sequence,
    List,
    Literal,
)  # For type annotations

# Numpy and Pandas
import numpy as np  # For numerical arrays and operations
import pandas as pd  # For DataFrame handling (KPSS p-values, MA DataFrame)

# Statsmodels
from statsmodels.tsa.stattools import kpss  # For KPSS stationarity test
from statsmodels.tools.sm_exceptions import (
    InterpolationWarning,
)  # For warning suppression

# Sklearn
from sklearn.decomposition import PCA  # For PCA in EoA smoother

# Scipy
from scipy.signal import butter, firwin, filtfilt, freqz, hilbert
from scipy.optimize import brentq

# Custom modules
from . import _burg as burg  # For AR model estimation via Burg's method
from .TS import *  # For 'ts' class (time series object)
from .plot import message_verb  # For printing message if verbose=True
from .etc import Rist  # For R-style list container
from .Calc import welch_window  # For Bandpass filter consistent with R version


# ================================================================

# ________________________________________________________________
# Correct shifted phase by filter
def zero_phasing(data: np.ndarray, H_func, fs: float) -> np.ndarray:
    """
    Apply zero-phase correction to filtered data using zero-padded FFT.

    Uses 2x zero-padding to perform linear (non-circular) convolution,
    preventing wrap-around edge artifacts that occur with standard
    circular FFT.

    Corrects phase distortion from causal filters (diff, AR) by:
        Y'(z) = Y(z) * e^{-iφ}  where φ = arg(H(z))

    Args:
        data (np.ndarray): Filtered data (e.g., AR residuals).
        H_func (callable): Function f_array -> H_array that computes the
            filter transfer function at arbitrary frequency grids.
        fs (float): Sampling frequency.

    Returns:
        np.ndarray: Phase-corrected data with same length as input.
    """
    from scipy.fft import fft, ifft

    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    n_fft = 2 * n  # zero-pad to avoid circular convolution

    # Transfer function on zero-padded frequency grid
    f = np.fft.fftfreq(n=n_fft, d=1.0 / fs)
    H = H_func(f)

    # Phase response of H(z): φ = arg(H(z))
    phase = np.angle(H)

    # Apply phase correction: Y'(z) = Y(z) * e^{-iφ}
    X_f = fft(data, n=n_fft)
    X_corrected = X_f * np.exp(-1j * phase)

    return np.real(ifft(X_corrected))[:n]


# ________________________________________________________________
# Differencing (Integrated process)

# Simple calculation of difference filter coefficients; Binomial
def diff_coef(d: int) -> np.ndarray:
    """Compute (1 - B)^d coefficients."""
    coef = np.array([1.0])
    for _ in range(d):
        coef = np.convolve(coef, [1, -1])
    return coef


# ===========================================================
# Fractional Differencing: (1-B)^d
#
# Weight recursion: w_0=1, w_k = w_{k-1} * (-(d-k+1)/k)
# For d>1: decompose as (1-B)^floor(d) * (1-B)^frac(d)
#   → integer part is exact (np.diff)
#   → fractional part (0 < d_frac < 1) has decaying weights
#
# Ref: Hosking (1981) Biometrika 68(1), 165-176
#      Lopez de Prado (2018) AFML, Ch.5
# ===========================================================
def frac_diff_coefs(d, window, thresh=1e-8):
    """Fractional differencing weights via recursion.
    For integer d, reproduces exact binomial coefficients."""
    w = [1.0]
    for k in range(1, window):
        w_k = w[-1] * (-(d - k + 1) / k)
        if abs(w_k) < thresh and k > max(20, int(d) + 5):
            break
        w.append(w_k)
    return np.array(w)


def frac_diff(x, d, window=1024):
    """Apply (1-B)^d to series x.
    For d > 1: decomposes into integer + fractional parts."""
    d_int = int(np.floor(d))
    d_frac = d - d_int

    # Integer part: exact via np.diff
    y = x.copy()
    if d_int > 0:
        y = np.diff(y, n=d_int)

    # Fractional part: FIR filter
    if abs(d_frac) > 1e-10:
        w = frac_diff_coefs(d_frac, window)
        W = len(w)
        y_filt = np.convolve(y, w, mode="full")[: len(y)]
        return y_filt[W - 1 :]  # trim initial transient
    return y

# Split time series into segments and run KPSS tests for stationarity
def check_stationary(ts_obj: ts, t_seg: float = 0.5) -> pd.DataFrame:
    """
    Split the time series into segments and perform KPSS tests for stationarity.

    Args:
        ts_obj (ts): Time series object.
        t_seg (float, optional): Duration of each segment in seconds. Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame with p-values of KPSS tests (Level and Trend) per segment.
    """
    data = ts_obj.data
    freq = ts_obj.sampling_freq

    n = len(data)
    chunk_length = int(t_seg * freq)
    if chunk_length < 2:
        chunk_length = 2

    p_values_level = []
    p_values_trend = []
    indices = []

    for i in range(0, n, chunk_length):
        segment = data[i : i + chunk_length]
        if len(segment) < 2:
            continue

        # KPSS test for level
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InterpolationWarning)
            try:
                stat_level, pval_level, _, _ = kpss(
                    segment, regression="c", nlags="legacy"
                )
            except Exception:
                pval_level = 1.0

        # KPSS test for trend
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InterpolationWarning)
            try:
                stat_trend, pval_trend, _, _ = kpss(
                    segment, regression="ct", nlags="legacy"
                )
            except Exception:
                pval_trend = 1.0

        p_values_level.append(pval_level)
        p_values_trend.append(pval_trend)
        indices.append(f"{i}-{i + len(segment)}")

    df = pd.DataFrame({"Level": p_values_level, "Trend": p_values_trend}, index=indices)
    return df


# Automatically determine differencing order to achieve stationarity
def auto_diff(
    ts_obj: ts,
    t_seg: float = 0.5,
    d_max: float = 2.0,
    frac: bool = False,
    verbose: bool = True,
) -> Rist:
    """
    Automatically determine the differencing order required for stationarity by KPSS test.

    Args:
        ts_obj (ts): Time series object.
        t_seg (float, optional): Segment duration in seconds for stationarity testing. Defaults to 0.5.
        d_max (float, optional): Maximum differencing order. Defaults to 2.
        frac (bool): Allow fractional differencing?
        verbose (bool, optional): If True, print progress messages. Defaults to True.

    Returns:
        Rist: R style list containing:
            - 'd': Selected differencing order.
            - 'out': Differenced time series object.
            - 'p_values': History of p-values per differencing iteration.
    """
    d_int = 0
    pval_history = Rist()

    message_verb(f"|> Phase 1: Finding integer differencing order", verb=verbose)

    # --- Phase 1: Find the first integer d that achieves stationarity ---
    current_data = ts_obj.data.copy()
    while d_int <= d_max:
        message_verb(f"||> Testing integer d={d_int}", verb=verbose)

        ts_temp = ts(
            current_data, start=ts_obj.start, sampling_freq=ts_obj.sampling_freq
        )
        pvals_df = check_stationary(ts_temp, t_seg)
        pval_history[f"d{d_int}.0"] = pvals_df

        if (pvals_df >= 0.1).all().all():
            message_verb(f"|||> Stationary at integer d={d_int}", verb=verbose)
            break

        if d_int >= d_max:
            message_verb(f"||! Reached d_max={d_max}", verb=verbose)
            break

        d_int += 1
        current_data = np.diff(ts_obj.data, n=d_int)

    final_d = float(d_int)
    final_data = current_data

    # --- Phase 2: Refine with fractional differencing if requested ---
    # Only refine if d_int > 0 and the series became stationary at d_int
    if frac and d_int > 0 and (pvals_df >= 0.1).all().all():
        message_verb(
            f"|> Phase 2: Refining with fractional orders in (d={d_int - 1}, d={d_int})",
            verb=verbose,
        )

        # Search from d_int - 0.9 to d_int - 0.1
        for f_step in range(1, 10):
            d_frac = round((d_int - 1) + (f_step * 0.1), 1)
            message_verb(f"||> Testing fractional d={d_frac}", verb=verbose)

            frac_data = frac_diff(ts_obj.data, d_frac)
            ts_temp = ts(
                frac_data, start=ts_obj.start, sampling_freq=ts_obj.sampling_freq
            )
            pvals_df = check_stationary(ts_temp, t_seg)
            pval_history[f"d{d_frac}"] = pvals_df

            if (pvals_df >= 0.1).all().all():
                message_verb(
                    f"|||> Optimal fractional d found at {d_frac}", verb=verbose
                )
                final_d = d_frac
                final_data = frac_data
                break

    # Adjust start time based on the number of samples lost
    shift_n = len(ts_obj.data) - len(final_data)
    out_ts = ts(
        final_data,
        start=ts_obj.start + shift_n * (1 / ts_obj.sampling_freq),
        sampling_freq=ts_obj.sampling_freq,
    )

    return Rist({"d": final_d, "out": out_ts, "p_values": pval_history})

# High-level wrapper to apply differencing manually or via KPSS-based auto differencing
def Differencing(
    ts_obj: ts,
    d: Union[int, float, str],
    t_seg: float = 0.5,
    d_max: float = 2.0, # Changed to float to support fractional d_max
    frac: bool = False,
    return_pvals: bool = False,
    verbose: bool = True,
) -> ts:
    """
    Apply differencing to a time series to induce stationarity.

    Args:
        ts_obj (ts): Input time series object.
        d (int or float or 'auto'): Differencing strategy.
            - 'auto': perform KPSS-based auto-differencing
            - int > 0: apply fixed-order differencing
            - 0: no differencing
        t_seg (float): Segment length (in seconds) for KPSS test (used only if d='auto')
        d_max (float): Maximum d to search in 'auto' mode.
        frac (bool): Allow fractional differencing?
        return_pvals (bool): Include KPSS p-values in output metadata (only if d='auto')
        verbose (bool): Print progress messages

    Returns:
        ts: Differenced time series object with `.diff_meta` attribute (Rist)
    """
    # 1. Determine the Strategy
    if d == "auto":
        # Automated search (Integer or Fractional based on 'frac' flag)
        diff_res = auto_diff(ts_obj, t_seg=t_seg, d_max=d_max, frac=frac, verbose=verbose)
        d_order, out_data = diff_res['d'], diff_res['out'].data
        method = "auto_frac" if frac else "auto_int"
        p_values = diff_res.p_values if return_pvals else None

    elif isinstance(d, (int, float)):
        # Fixed order differencing
        d_order = d
        method = "fixed_frac" if isinstance(d, float) or frac else "fixed_int"
        p_values = None
        
        if d == 0:
            out_data = ts_obj.data.copy()
        elif isinstance(d, int) and not frac:
            out_data = np.diff(ts_obj.data, n=d)
        else:
            # Covers float d or int d with frac=True
            out_data = frac_diff(ts_obj.data, d)
    else:
        raise ValueError("d must be 'auto' or a numeric value (int/float).")

    # 2. Construct the Output Object
    # Calculate samples lost to adjust start time
    shift_n = len(ts_obj.data) - len(out_data)
    out_ts = ts(
        out_data,
        start=ts_obj.start + shift_n * (1 / ts_obj.sampling_freq),
        sampling_freq=ts_obj.sampling_freq,
    )

    # 3. Metadata Attachment
    meta = Rist(
        method=method, 
        d_order=d_order, 
        is_frac=(d_order % 1 != 0), # True if there is a fractional part
        unbounded=(d == "auto")
    )
    if p_values is not None:
        meta["p_values"] = p_values

    inherit_ts_attrs(ts_obj, out_ts)
    setattr(out_ts, "diff_meta", meta)

    if verbose:
        message_verb(f"|> Differencing applied: method={method}, d={d_order}", verb=verbose)

    return out_ts



# ________________________________________________________________
# Autoregressive

# Estimate AR model coefficients using Burg's method
def burgar(
    x: Union[np.ndarray, Sequence[float]],
    ic: str = "AIC",
    order_max: Optional[int] = None,
    demean: bool = True,
    var_method: int = 2,
) -> Rist:
    """
    Python version of R's ar.burg function (but modified as burgar).
    Estimate AR model coefficients using Burg's method.

    Args:
        x (array-like): Input time series data.
        ic (str, optional): Information criterion to select model order ('AIC', 'BIC', 'FPE', 'AICc', 'KIC', 'AKICc', 'HQIC'). Defaults to 'AIC'. 'HQIC' = Hannan-Quinn (penalty 2*log(log(n)), between AIC and BIC).
        order_max (int, optional): Maximum AR order. If None, computed automatically.
        demean (bool, optional): Whether to remove the mean before fitting. Defaults to True.
        var_method (int, optional): Innovation variance method (1 or 2). Defaults to 2.

    Returns:
        Rist: R style list containing:
            - 'order': Selected AR order.
            - 'ar': Estimated AR coefficients.
            - 'var_pred': Prediction variance at selected order.
            - 'vars_pred': All prediction variances.
            - 'x_mean': Mean of input series (if demean=True).
            - 'ic': Normalized information criteria.
            - 'n_used': Number of samples.
            - 'order_max': Maximum AR order considered.
            - 'partialacf': Partial autocorrelations.
            - 'method': Method description.
            - 'series': Input label.
            - 'asy_var_coef': Asymptotic variance of coefficients (None).
    """

    # Information criterion sub-functions
    def AIC(
        order_max: int, vars_pred: np.ndarray, n_used: int, demean: bool = False
    ) -> np.ndarray:
        orders = np.arange(order_max + 1)
        return 2 * orders + n_used * np.log(vars_pred) + 2 * int(demean)

    def BIC(
        order_max: int, vars_pred: np.ndarray, n_used: int, demean: bool = False
    ) -> np.ndarray:
        orders = np.arange(order_max + 1)
        return (
            orders * np.log(n_used)
            + n_used * np.log(vars_pred)
            + int(demean) * np.log(n_used)
        )

    def FPE(
        order_max: int, vars_pred: np.ndarray, n_used: int, demean: bool = False
    ) -> np.ndarray:
        orders = np.arange(order_max + 1)
        k = orders + int(demean)
        return (n_used + k + 1) / (n_used - k - 1) * vars_pred

    def AICc(
        order_max: int, vars_pred: np.ndarray, n_used: int, demean: bool = False
    ) -> np.ndarray:
        orders = np.arange(order_max + 1)
        k = orders + int(demean)
        return n_used * np.log(vars_pred) + 2 * k + (2 * k * (k + 1)) / (n_used - k - 1)

    def KIC(
        order_max: int, vars_pred: np.ndarray, n_used: int, demean: bool = False
    ) -> np.ndarray:
        orders = np.arange(order_max + 1)
        return n_used * np.log(vars_pred) + 3 * orders + 3 * int(demean)

    def AKICc(
        order_max: int, vars_pred: np.ndarray, n_used: int, demean: bool = False
    ) -> np.ndarray:
        orders = np.arange(order_max + 1)
        k = orders + int(demean)
        return n_used * np.log(vars_pred) + 3 * k + (3 * k * (k + 1)) / (n_used - k - 1)

    def HQIC(
        order_max: int, vars_pred: np.ndarray, n_used: int, demean: bool = False
    ) -> np.ndarray:
        # Hannan-Quinn (1979): penalty per parameter = 2*log(log(n)),
        # which lies strictly between AIC (2) and BIC (log(n)).
        orders = np.arange(order_max + 1)
        c = 2.0 * np.log(np.log(n_used))
        return n_used * np.log(vars_pred) + c * orders + c * int(demean)

    x = np.asarray(x, dtype=np.float64).ravel()
    n_used = x.size

    if demean:
        x_mean = np.mean(x)
        x = x - x_mean
    else:
        x_mean = 0.0

    # Default order_max
    if order_max is None:
        order_max = min(n_used - 1, int(10 * np.log10(n_used)))
    else:
        order_max = int(order_max)

    if order_max < 1:
        raise ValueError("'order_max' must be >=1")
    if order_max >= n_used:
        raise ValueError("'order_max' must be < length of x")

    # Call burg.burg()
    coefs, var1, var2 = burg.burg(x, order_max)
    coefs = np.asarray(coefs, dtype=np.float64)
    var1 = np.asarray(var1, dtype=np.float64)
    var2 = np.asarray(var2, dtype=np.float64)

    # Partial ACF
    partialacf = np.diag(coefs)

    # Innovation variance
    vars_pred = var1 if var_method == 1 else var2
    if np.any(np.isnan(vars_pred)):
        raise ValueError("zero-variance series")

    if ic is None:
        # No model selection, use order_max directly
        selected_order = order_max
        xic = None
        xic_norm = None
    else:
        ic_fun = {
            "AIC": AIC,
            "BIC": BIC,
            "FPE": FPE,
            "AICc": AICc,
            "KIC": KIC,
            "AKICc": AKICc,
            "HQIC": HQIC,
        }.get(ic)

        if ic_fun is None:
            raise ValueError(
                f"Unknown ic: {ic}. Must be one of 'AIC', 'BIC', 'FPE', 'AICc', 'KIC', 'AKICc', 'HQIC', or None"
            )
    
        xic = ic_fun(order_max, vars_pred, n_used, demean)
        mic = np.nanmin(xic)
        xic_norm = np.where(np.isfinite(mic), xic - mic, np.where(xic == mic, 0, np.inf))
        selected_order = np.flatnonzero(xic_norm == 0)[0]

    # AR coefficients
    if selected_order > 0:
        ar = coefs[:selected_order, selected_order - 1]
    else:
        ar = np.array([])
    var_pred = vars_pred[selected_order]

    # Residuals (convolution-based)
    if selected_order > 0:
        a = np.r_[1.0, -ar]
        resid = np.convolve(x, a, mode="valid")  # mode="valid" returns only the region where full overlap exists (length: n - p)
    else:
        resid = x.copy()

    # Comments in burgar() of R source file
    # WE DON'T NEED THIS WHICH TAKES TIME A LOT!
    # if (order) {
    #    xacf <- acf(x, type = "covariance", lag.max = order,
    #                plot = FALSE)$acf
    #    res$asy.var.coef <- solve(toeplitz(drop(xacf)[seq_len(order)])) *
    #        var.pred/n.used
    # }

    # Return
    return Rist(
        {
            "order": int(selected_order),
            "ar": ar,
            "resid": resid,
            "var_pred": var_pred,
            "vars_pred": vars_pred,
            "x_mean": x_mean,
            "ic": xic_norm,  # xic_dict,
            "n_used": n_used,
            "order_max": order_max,
            "partialacf": partialacf,
            "method": f"Burg{var_method}",
            "series": "x",
            "asy_var_coef": None,
        }
    )

def pred_resid(ts_obj, arcoef, zero_phase=False):
    """
    Predict AR residuals using fitted AR coefficients from other dataset.
    Internally, it also performs `zero_phasing()` as `sar()` does.

    Args:
        x (ts): Input time series object.
        arcoef: AR coefficients with convention of [a_1, a_2, a_3, ..., a_p] (NOT the [1, -a_1, -a_2, -a_3, ..., -a_p])

    Returns:
        ts: AR residual time series.
    """
    data = ts_obj.data
    p_order = len(arcoef)
    a = np.r_[1.0, -arcoef]

    # Use mode="valid" to avoid boundary effects
    resid = np.convolve(data, a, mode="valid")

    # Zero-phase correction
    if zero_phase:
        _fs = ts_obj.sampling_freq
        resid = zero_phasing(resid, lambda f: H_ar(f, _fs, arcoef), _fs)

    new_start = ts_obj.start + p_order / ts_obj.sampling_freq

    return ts(resid, start=new_start, sampling_freq=ts_obj.sampling_freq)

def sar(
    ts_obj: ts,
    ic: str = "AIC",
    order_max: Optional[int] = None,
    zero_phase: bool = False,
    **kwargs: Any,
) -> Rist:
    """
    Fit a single autoregressive (AR) model using Burg's method and return zero-phase residuals and features.

    Args:
        ts_obj (ts): Input time series object.
        ic (str, optional): Information criterion to select model order.
            One of 'AIC', 'BIC', 'FPE', 'AICc', 'KIC', 'AKICc'. Defaults to 'AIC'.
        order_max (int, optional): Maximum AR order to consider. If None, determined automatically.
        **kwargs: Additional arguments passed to `burgar()` (e.g., demean, var_method).

    Returns:
        Rist: Container with:
            - resid: Zero-phase residual time series as `ts` object.
            - ar_coef: AR coefficients (np.ndarray).
            - var_pred: Prediction variance (float).
            - p_order: Selected AR order.
            - ar_collector: Always "single" for SAR.
            - AR_obj: Full AR model result from `burgar()`.
    """
    # Run Burg method AR
    ar_result = burgar(ts_obj.data, ic=ic, order_max=order_max, **kwargs)
    p = ar_result.order
    resid = ar_result.resid

    # Zero-phase correction
    if zero_phase:
        _fs = ts_obj.sampling_freq
        resid = zero_phasing(resid, lambda f: H_ar(f, _fs, ar_result.ar), _fs)

    # Re-arrange residual w.r.t. the lack of initial data
    new_start = ts_obj.start + p / ts_obj.sampling_freq
    resid_ts = ts(resid, start=new_start, sampling_freq=ts_obj.sampling_freq)

    return Rist(
        resid=resid_ts,
        ar_coef=ar_result.ar,
        var_pred=ar_result.var_pred,
        parcor=ar_result.partialacf,
        p_order=p,
        ar_collector="single",
        AR_obj=ar_result,
    )


# Fit ensemble of AR models and return aggregated residuals
def ear(
    ts_obj: ts,
    ps: Sequence[int] = (100, 500, 1000),
    ic: Union[str, bool] = True,
    ar_collector: str = "median",
    zero_phase: bool = False,
) -> Rist:
    """
    Fit multiple AR models (ensemble) and aggregate residuals.

    Args:
        ts_obj (ts): Input time series object.
        ps (Sequence[int], optional): List of AR orders to fit. Defaults to (100, 500, 1000).
        ic (str or bool, optional): Information criterion for order selection. Defaults to True.
        ar_collector (str, optional): Method to aggregate residuals ('median', 'mean', 'pca'). Defaults to 'median'.

    Returns:
        Rist: Container with:
            - resid: Aggregated residuals as `ts` object.
            - ar_coef: Rist of AR coefficients keyed by order (e.g., {p100: [...], p500: [...]}).
            - var_pred: Rist of prediction variances keyed by order.
            - p_order: Selected AR orders.
            - ar_collector: Aggregation method used.
    """
    ar_fits = [
        burgar(ts_obj.data, ic=ic, order_max=p) for p in ps
    ]  # Collect fitted ar obj
    orders = np.array([fit.order for fit in ar_fits])  # Collect fitted orders

    # Filter out only unique fittings
    # e.g. order_max=10, 15: both may be fitted with order=9; They are duplicated
    unique_indices = np.unique(orders, return_index=True)[1]
    ar_fits = [ar_fits[i] for i in unique_indices]
    psel = orders[unique_indices]

    # Collect residuals
    resids_list = [fit.resid for fit in ar_fits]
    arcoef_list = [fit.ar for fit in ar_fits]
    # Zero-phase correction
    if zero_phase:
        _fs = ts_obj.sampling_freq
        zp_list = []
        for i in range(len(resids_list)):
            resid = resids_list[i]
            arcoef = arcoef_list[i]
            zp_list.append(zero_phasing(resid, lambda f, _a=arcoef: H_ar(f, _fs, _a), _fs))
        resids_list = zp_list  # overwrite

    min_len = min(map(len, resids_list))
    resid_mat = np.stack([r[-min_len:] for r in resids_list], axis=1)

    if len(psel) == 1:
        resid_ens_core = resid_mat[:, 0]
        ar_collector_name = "Not aggregated"
    else:
        if ar_collector == "median":
            resid_ens_core = np.median(resid_mat, axis=1)
        elif ar_collector == "mean":
            resid_ens_core = np.mean(resid_mat, axis=1)
        elif ar_collector == "pca":
            resid_ens_core = extract_pc(resid_mat, pc="PC1")
        else:
            raise ValueError(f"Unsupported collector: {ar_collector}")
        ar_collector_name = ar_collector

    new_start = ts_obj.start + (len(ts_obj.data) - min_len) / ts_obj.sampling_freq
    resids_ts = ts(resid_ens_core, start=new_start, sampling_freq=ts_obj.sampling_freq)

    # Build Rist for ar_coef and var_pred keyed by p order
    ar_coef_rist = Rist({f"ar{fit.order}": fit.ar for fit in ar_fits})
    var_pred_rist = Rist({f"ar{fit.order}": fit.var_pred for fit in ar_fits})
    parcor_rist = Rist({f"ar{fit.order}": fit.partialacf for fit in ar_fits})
    return Rist(
        resid=resids_ts,
        ar_coef=ar_coef_rist,
        var_pred=var_pred_rist,
        parcor=parcor_rist,
        p_order=psel,
        ar_collector=ar_collector_name,
    )


# High-level wrapper to fit AR model(s) and prepare residuals/features
def Autoregressive(
    ts_obj: ts,
    p: Union[int, Sequence[int]],
    ic: str = "AIC",
    verbose: bool = True,
    ar_collector: str = "median",
    zero_phase: bool = False,
    **kwargs: Any,
) -> ts:
    """
    Fit autoregressive (AR) model(s) and return residuals with associated features.

    - For a single order (int), uses `sar()` (single AR).
    - For multiple candidate orders, uses `ear()` (ensemble AR with aggregation).

    Args:
        ts_obj (ts): Input time series object.
        p (int or Sequence[int]): AR order(s) to consider.
            - If single int: fit AR(p).
            - If sequence: ensemble modeling across multiple p.
        ic (str or bool, optional): Information criterion for order selection (e.g., 'aic', 'bic', True for default). Default is True.
        verbose (bool, optional): If True, print progress messages. Default is True.
        ar_collector (str, optional): Aggregation strategy for ensemble ('median', 'mean', 'pca'). Default is 'median'.
        **kwargs: Additional keyword arguments passed to `sar()` or `ear()`.

    Returns:
        ts: Residuals as a time series (`ts`) object, with `ar_meta` attribute (Rist) containing:
            - 'ar_coef': AR coefficients (ndarray for SAR, Rist for EAR).
            - 'var_pred': Prediction variance (float for SAR, Rist for EAR).
            - 'p_order': Selected AR order(s).
            - 'ar_collector': Aggregation strategy used.
    """
    if isinstance(p, (list, tuple)) and len(p) > 1:
        result = ear(ts_obj, ps=p, ic=ic, ar_collector=ar_collector, zero_phase=zero_phase, **kwargs)
        message_verb(
            f"|> p={result.p_order} selected and aggregated by: {result.ar_collector}",
            verb=verbose,
        )
    else:
        p_single = p[0] if isinstance(p, (list, tuple)) else p
        result = sar(ts_obj, ic=ic, order_max=p_single, zero_phase=zero_phase, **kwargs)
        message_verb(f"|> p={result.p_order} selected!", verb=verbose)

    resid = result.resid

    # Inherit attributes
    inherit_ts_attrs(ts_obj, resid)

    # Attach metadata to ts object
    meta = Rist(
        ar_coef=result.ar_coef,
        var_pred=result.var_pred,
        parcor=result.parcor,
        p_order=result.p_order,
        ar_collector=result.ar_collector,
    )
    setattr(resid, "ar_meta", meta)

    return resid


# ________________________________________________________________
# MovingAverage


# Compute moving average smoother replicating R forecast::ma
def sma(ts_obj: ts, order: int, centre: bool = True, na_rm: bool = True) -> ts:
    """
    Moving Average smoother replicating R's forecast::ma behavior.

    Args:
        ts_obj (ts): Time series object.
        order (int): Moving average order.
        centre (bool): Centered moving average if True, causal if False.
        na_rm (bool): Remove NaN values if True.

    Returns:
        ts: Smoothed time series object with:
            - q_order: MA window size (same as order)
            - ma_collector: Always 'single'
    """
    if abs(order - round(order)) > 1e-8:
        raise ValueError("order must be an integer")
    order = int(order)

    # Define weights
    if order % 2 == 0 and centre:
        w = np.concatenate(([0.5], np.ones(order - 1), [0.5])) / order
    else:
        w = np.ones(order) / order

    if centre:
        y_raw = np.convolve(ts_obj.data, w, mode="valid")
        pad = len(w) // 2
        y = np.concatenate([np.full(pad, np.nan), y_raw, np.full(pad, np.nan)])
        new_start = ts_obj.start
    else:
        y = np.convolve(ts_obj.data, w, mode="valid")
        new_start = ts_obj.start + (order - 1) / ts_obj.sampling_freq

    # Handle na_rm
    if na_rm:
        valid = ~np.isnan(y)
        y = y[valid]
        if len(y) == 0:
            raise ValueError("All data were removed due to NA.")
        if centre:
            new_start = ts_obj.start + np.flatnonzero(valid)[0] / ts_obj.sampling_freq

    smoothed_ts = ts(y, start=new_start, sampling_freq=ts_obj.sampling_freq)

    # Attach attributes
    setattr(smoothed_ts, "q_order", order)
    setattr(smoothed_ts, "ma_collector", "single")

    return smoothed_ts


# Apply PCA to matrix and fix sign of loadings
def apply_pca(
    x: Union[np.ndarray, pd.DataFrame],
    retx: bool = True,
    center: bool = False,
    scale: bool = False,
    tol: Optional[float] = None,
    rank: Optional[int] = None,
    **kwargs,
) -> Rist:
    """
    Apply PCA and fix loadings' sign.

    Args:
        x: Data matrix.
        retx: If True, return transformed data.
        center: Center the data.
        scale: Scale the data to unit variance.
        tol: Ignored (placeholder).
        rank: Number of components to keep.
        **kwargs: Additional args for PCA.

    Returns:
        Rist: Container with 'rotation' and 'x' (if retx).
    """
    x_arr = np.asarray(x, dtype=np.float64)

    # Centering
    if center:
        x_arr = x_arr - np.nanmean(x_arr, axis=0)

    # Scaling
    if scale:
        x_arr = x_arr / np.nanstd(x_arr, axis=0)

    # Rank = n_components
    n_components = rank if rank is not None else min(x_arr.shape)

    pca = PCA(n_components=n_components, **kwargs)
    pca.fit(x_arr)

    rotation = pca.components_.T

    # Fix sign
    neg_cols = np.where(rotation[0, :] < 0)[0]
    rotation[:, neg_cols] *= -1

    # Result container
    result = Rist(rotation=rotation)

    if retx:
        transformed = np.dot(x_arr, rotation)
        result["x"] = transformed

    return result


# Extract specified principal components from PCA results
def extract_pc(
    x: Union[np.ndarray, pd.DataFrame], pc: Union[str, Sequence[str]] = "PC1"
) -> np.ndarray:
    """
    Extract specified principal components.

    Args:
        x: Data matrix.
        pc: e.g., "PC1" or ["PC1","PC2"].

    Returns:
        np.ndarray: Extracted PC(s).
    """
    pca_res = apply_pca(x, retx=True, center=False, scale=False)
    pc_matrix = pca_res["x"]
    pc_names = [f"PC{i+1}" for i in range(pc_matrix.shape[1])]

    if isinstance(pc, str):
        pc = [pc]

    indices = [pc_names.index(p) for p in pc]
    out = pc_matrix[:, indices]
    if out.shape[1] == 1:
        out = out.ravel()
    return out


# Ensemble smoother combining moving averages via mean, median, or PCA
def eoa(
    ts_obj: ts,
    qs: Union[np.ndarray, List[int]],
    collector: Literal["mean", "median", "pca"] = "median",
    return_mas: bool = False,
) -> ts:
    """
    Ensemble of Averages (EoA) smoother using pandas DataFrame.

    Args:
        ts_obj: Time series object.
        qs: List of orders for moving averages.
        collector: Aggregation method ('mean', 'median', or 'pca').
        return_mas: If True, attach pandas DataFrame of all MAs as 'mas' attribute.

    Returns:
        ts: Smoothed time series object with:
            - q_order: List of MA orders used.
            - ma_collector: Aggregation method name.
            - mas: DataFrame of all MAs (optional).
    """
    # Compute moving averages
    ma_series = [sma(ts_obj, order=int(q), centre=True, na_rm=False).data for q in qs]

    # Create pandas DataFrame
    df = pd.DataFrame(
        {f"q{q}": col for q, col in zip(qs, ma_series)}, index=ts_obj.times
    )

    # Drop Missing values
    df = df.dropna()
    new_start = df.index[0]

    # Collector aggregation
    if collector == "mean":
        agg = df.mean(axis=1).values
    elif collector == "median":
        agg = df.median(axis=1).values
    elif collector == "pca":
        agg = extract_pc(df.values, pc="PC1")
    else:
        raise ValueError("Invalid collector.")

    result_ts = ts(agg, start=new_start, sampling_freq=ts_obj.sampling_freq)

    # Attach attributes
    setattr(result_ts, "q_order", np.array(list(qs)))
    setattr(result_ts, "ma_collector", collector)
    if return_mas:
        setattr(result_ts, "mas", df)

    return result_ts


# High-level wrapper for Moving-Average or Ensemble of Averages.
def MovingAverage(
    ts_obj: ts, q: Union[int, Sequence[int], np.ndarray], verbose: bool = True, **kwargs
) -> ts:
    """
    Apply Moving-Average (single) or Ensemble of Averages (EoA) smoothing.

    Args:
        ts_obj (ts): Input time series object.
        q (int or Sequence[int]): Single MA order or multiple orders for ensemble.
        verbose (bool): If True, print progress messages.
        **kwargs: Extra arguments passed to `sma()` or `eoa()`.

    Returns:
        ts: Smoothed time series object with attached `ma_meta` (Rist) containing:
            - 'q_order': MA order(s) used.
            - 'ma_collector': Aggregation method ('single', 'mean', 'median', 'pca').
            - 'mas': Raw MAs (only available for EoA if `return_mas=True`).
    """
    # Convert to array
    q_arr = np.array([q]) if np.isscalar(q) else np.asarray(q, dtype=int)

    if len(q_arr) > 1:
        res_ts = eoa(ts_obj, qs=q_arr, **kwargs)
        message_verb(
            f"|> q={{ {', '.join(map(str, q_arr))} }} (collector: {res_ts.ma_collector})",
            verbose,
        )
    else:
        q_single = int(q_arr[0])
        res_ts = sma(ts_obj, order=q_single, **kwargs)
        message_verb(f"|> q={q_single}", verbose)

    # Wrap all related metadata into one attribute
    meta = Rist(
        q_order=getattr(res_ts, "q_order", q_arr.tolist()),
        ma_collector=getattr(res_ts, "ma_collector", "single"),
        mas=getattr(res_ts, "mas", None),
    )

    # Clean up: optionally remove original attributes if needed
    #   by assigning ts again.
    res_ts = ts(res_ts.data, start=res_ts.start, sampling_freq=res_ts.sampling_freq)
    setattr(res_ts, "ma_meta", meta)

    # Inherit other attributes from input
    inherit_ts_attrs(ts_obj, res_ts)

    return res_ts


def calculate_ma_cutoff_seqarima(q):
    """
    Calculate the -3dB cutoff frequency for seqARIMA's sma() filter.

    The transfer function is:
        H(ω) = Σ_{k=0}^{L-1} w[k] · e^{-jωk}
    where L is the filter length. The -3dB cutoff frequency f_c satisfies:
        |H(2π f_c)|² = 0.5

    Args:
        q (int): Moving average order. Must be >= 2.

    Returns:
        float: Normalized cutoff frequency (0 to 0.5, where 0.5 is Nyquist).
               Multiply by sampling frequency to get cutoff in Hz.

    Raises:
        ValueError: If q < 2, since a 1-point MA performs no filtering
                    (|H(ω)| = 1 for all ω) and has no cutoff frequency.
    """
    if q < 2:
        raise ValueError(
            f"q must be >= 2. For q=1, the filter weight is [1], "
            f"giving |H(ω)|=1 for all frequencies (no filtering). "
            f"The -3dB cutoff is undefined."
        )

    if q % 2 == 0:
        w = np.concatenate(([0.5], np.ones(q - 1), [0.5])) / q
    else:
        w = np.ones(q) / q

    def mag_sq_minus_half(omega_c):
        _, H = freqz(w, 1, worN=[omega_c])
        return np.abs(H[0]) ** 2 - 0.5

    omega_c = brentq(mag_sq_minus_half, 1e-6, np.pi)
    return omega_c / (2 * np.pi)

# ________________________________________________________________
# BandPass
def BandPass(
    ts_obj: ts,
    fl: Optional[float] = None,
    fu: Optional[float] = None,
    resp: str = "FIR",
    filt_order: Optional[int] = None,
    verbose: bool = True,
) -> ts:
    """
    Apply a band-pass, high-pass, or low-pass filter to a time series.

    Args:
        ts_obj (ts): Input time series object.
        fl (float or None): Lower cutoff frequency (Hz). If None, acts as low-pass.
        fu (float or None): Upper cutoff frequency (Hz). If None, acts as high-pass.
        resp (str): Filter type, "FIR" (default) or "IIR".
        filt_order (int or None): Filter order. Defaults to 512 for FIR and 8 for IIR.
        verbose (bool): Whether to print progress messages.

    Returns:
        ts: Filtered time series object with `.bp_meta` attribute containing:
            - 'resp': Filter type ("FIR" or "IIR")
            - 'order': Filter order
            - 'type': One of {"pass", "high", "low"}
            - 'cutoff': Tuple of normalized cutoff frequencies
    """
    sampling_freq = ts_obj.sampling_freq
    nyq = sampling_freq / 2

    # Choose filter type
    if resp.upper() == "FIR":
        n = filt_order or 512
        fir = True
    elif resp.upper() == "IIR":
        n = filt_order or 8
        fir = False
    else:
        raise ValueError("resp must be either 'FIR' or 'IIR'")

    # Determine filter mode
    if fl is not None and fu is not None:
        ftype = "bandpass"
        cutoff = [fl / nyq, fu / nyq]
    elif fl is not None:
        ftype = "highpass"
        cutoff = fl / nyq
    elif fu is not None:
        ftype = "lowpass"
        cutoff = fu / nyq
    else:
        raise ValueError("At least one of fl or fu must be specified.")

    # Design filter
    if fir:
        # filt = firwin(numtaps=n + 1, cutoff=cutoff, pass_zero=ftype, window="hann")
        filt = firwin(numtaps=n + 1, cutoff=cutoff, pass_zero=ftype, window="boxcar")
        filt = filt * welch_window(n + 1)
        filt_out = filtfilt(filt, [1.0], ts_obj.data)
    else:
        b, a = butter(N=n, Wn=cutoff, btype=ftype)
        filt_out = filtfilt(b, a, ts_obj.data)

    # Create output ts
    out_ts = ts(filt_out, start=ts_obj.start, sampling_freq=sampling_freq)
    inherit_ts_attrs(ts_obj, out_ts)

    # Attach metadata
    meta = Rist(resp=resp.upper(), order=n, type=ftype, cutoff=[fl, fu])
    setattr(out_ts, "bp_meta", meta)

    message_verb(
        f"|> Band-pass ({resp}) filter applied: type={ftype}, order={n}, cutoff={fl, fu} Hz",
        verb=verbose,
    )

    return out_ts


# ________________________________________________________________
# seqARIMA


# Final wrapper function for seqARIMA denoising
def seqarima(
    ts_obj: ts,
    p: Union[int, Sequence[int]],
    d: Union[int, float, str, None] = None,
    q: Union[int, Sequence[int], None] = None,
    fl: Optional[float] = None,
    fu: Optional[float] = None,
    diff_max: int = 2,
    diff_tseg: float = 0.5,
    diff_frac: bool = False,
    ar_collector: str = "mean",
    ma_collector: str = "mean",
    ar_ic: str = "AIC",
    zero_phase: bool = True,
    verbose: bool = True,
) -> ts:
    """
    Sequential ARIMA Denoising:
        Differencing -> AR -> MA -> Bandpass

    Args:
        ts_obj (ts): Input time series.
        p (int or list of int): AR order(s). Required.
        d (int or None): Differencing order (None to skip).
        q (int or list of int or None): MA order(s).
        fl (float or None): Bandpass lower frequency bound.
        fu (float or None): Bandpass upper frequency bound.
        ar_collector (str): Aggregation method for AR stage.
        ma_collector (str): Aggregation method for MA stage.
        ar_aic (str): Information criterion for AR order selection.
        verbose (bool): Print progress messages.

    Returns:
        ts: Final output with stage metadata as attributes.
    """
    message_verb("> Running seqarima...", verb=verbose)

    out = ts_obj

    # Step 1: Differencing (optional)
    if d is not None:
        message_verb("> (1) Difference stage", verb=verbose)
        out = Differencing(
            out, d=d, t_seg=diff_tseg, frac=diff_frac, d_max=diff_max, verbose=verbose
        )

    # Step 2: Autoregressive (required)
    message_verb("> (2) Autoregressive stage", verb=verbose)
    out = Autoregressive(
        out, p=p, ic=ar_ic, zero_phase=False, verbose=verbose, ar_collector=ar_collector
    )

    # Zero phase correction for Diff + AR together
    if zero_phase:
        is_ear = out.ar_meta.ar_collector != "single"
        if is_ear:
            warnings.warn("Zero-phase correction is not supported for ensemble AR. Skipping.")
        else:
            _fs = ts_obj.sampling_freq
            _d_order = out.diff_meta.d_order if (hasattr(out, "diff_meta") and out.diff_meta is not None) else None
            _ar_coef = out.ar_meta.ar_coef
            if _d_order is not None:
                H_func = lambda f: H_diff(f, _fs, _d_order) * H_ar(f, _fs, _ar_coef)
            else:
                H_func = lambda f: H_ar(f, _fs, _ar_coef)
            out_zp = tsref(zero_phasing(out.data, H_func, _fs), out)
            inherit_ts_attrs(out, out_zp)
            out = out_zp

    # Step 3: Moving Average (optional)
    if q: #q is not None:
        message_verb("> (3) Moving-average stage", verbose)
        out = MovingAverage(out, q=q, verbose=verbose, collector=ma_collector)

    # Step 4: Bandpass Filter (optional)
    if (fl or fu) and (fl != 0 or fu != 0):
        message_verb("> (4) Pass filter stage", verb=verbose)
        out = BandPass(out, fl=fl, fu=fu, verbose=verbose)

    return out

# ________________________________________________________________
# Parameter extraction
def extract_seqarima_params(seqarima_obj) -> Rist:
    """
    Extract transfer function parameters from seqARIMA result object.

    AR stage is required - raises error if ar_meta is missing.

    Returns Rist with parameters:
        - fs: sampling frequency
        - d: differencing order (if diff applied)
        - ar_coef: AR coefficients
        - var_pred: AR prediction variance
        - q_list: EoA window sizes as list (if MA/EoA applied)
        - fl, fu: bandpass cutoffs (if BP applied)
        - bp_order: bandpass filter order (if BP applied)
    """
    # AR is required
    if not (hasattr(seqarima_obj, "ar_meta") and seqarima_obj.ar_meta is not None):
        raise ValueError("AR stage is required. ar_meta not found.")

    params = Rist(
        fs=seqarima_obj.sampling_freq,
        ar_coef=seqarima_obj.ar_meta.ar_coef,
        var_pred=seqarima_obj.ar_meta.var_pred,
        parcor=seqarima_obj.ar_meta.parcor,
        ar_collector=seqarima_obj.ar_meta.ar_collector,
    )

    # Differencing (optional)
    if hasattr(seqarima_obj, "diff_meta") and seqarima_obj.diff_meta is not None:
        params["d"] = seqarima_obj.diff_meta.d_order
        params["diff_frac"] = seqarima_obj.diff_meta.is_frac

    # EoA (optional)
    if hasattr(seqarima_obj, "ma_meta") and seqarima_obj.ma_meta is not None:
        q = seqarima_obj.ma_meta.q_order
        params["q_list"] = list(np.atleast_1d(q))
        params["ma_collector"] = seqarima_obj.ma_meta.ma_collector

    # Bandpass (optional)
    if hasattr(seqarima_obj, "bp_meta") and seqarima_obj.bp_meta is not None:
        fl, fu = seqarima_obj.bp_meta.cutoff
        params["fl"] = fl
        params["fu"] = fu
        params["bp_order"] = getattr(seqarima_obj.bp_meta, "order", 512)

    return params

def has_param(params: Rist, key: str) -> bool:
    """Check if parameter exists in extracted params."""
    return key in params._name_to_index

# ________________________________________________________________
# seqARIMA transfer functions
# Pipeline: Differencing -> AR filtering -> EoA -> BP filter (filtfilt) -> x_out


def H_diff(f: np.ndarray, fs: float, d: float = 1) -> np.ndarray:
    """
    Transfer function for d-th order differencing filter.

    H_diff(f) = (1 - e^{-j2πf/fs})^d

    Args:
        f: Frequency array (Hz)
        fs: Sampling frequency (Hz)
        d: Differencing order (default: 1)

    Returns:
        Complex transfer function H(f)
    """
    f = np.asarray(f)
    return (1 - np.exp(-1j * 2 * np.pi * f / fs)) ** d # same expression with using freqz()


def H_ar(f: np.ndarray, fs: float, ar_coef: np.ndarray) -> np.ndarray:
    """
    Transfer function for AR filter.

    H_AR(f) = 1 / A(e^{j2πf/fs}) where A(z) = 1 - Σ a_k z^{-k}

    Args:
        f: Frequency array (Hz)
        fs: Sampling frequency (Hz)
        ar_coef: AR coefficients [a_1, a_2, ..., a_p]

    Returns:
        Complex transfer function H(f)
    """
    if len(ar_coef) == 0:
        return np.ones_like(f, dtype=complex)

    a_poly = np.r_[1.0, -ar_coef]
    w = 2 * np.pi * f / fs
    _, H = freqz(a_poly, 1, worN=w)

    return H


def H_ma(f: np.ndarray, fs: float, q: int) -> np.ndarray:
    """
    Transfer function for single centered MA filter.

    Matches sma() weights exactly:
    - Even q: [0.5, 1, 1, ..., 1, 0.5] / q (length q+1)
    - Odd q:  [1, 1, ..., 1] / q (length q)

    Args:
        f: Frequency array (Hz)
        fs: Sampling frequency (Hz)
        q: MA window size

    Returns:
        Complex transfer function H(f)
    """
    if q % 2 == 0:
        w = np.concatenate(([0.5], np.ones(q - 1), [0.5])) / q
    else:
        w = np.ones(q) / q

    omega = 2 * np.pi * f / fs
    _, H = freqz(w, 1, worN=omega)

    return H


def H_eoa(f: np.ndarray, fs: float, q_list: list[int]) -> np.ndarray:
    """
    Transfer function for Ensemble of Averages.

    H_EoA(f) = (1/K) Σ H_MA,q_k(f)

    Args:
        f: Frequency array (Hz)
        fs: Sampling frequency (Hz)
        q_list: List of MA window sizes [q_1, q_2, ..., q_K]

    Returns:
        Complex transfer function H(f)
    """
    if len(q_list) == 0:
        return np.ones_like(f, dtype=complex)

    H_sum = np.zeros_like(f, dtype=complex)
    for q in q_list:
        H_sum += H_ma(f, fs, int(q))

    return H_sum / len(q_list)


def H_bp(
    f: np.ndarray, fs: float, fl: float, fu: float, order: int = 512
) -> np.ndarray:
    """
    Transfer function for bandpass filter (FIR with Welch window).

    Args:
        f: Frequency array (Hz)
        fs: Sampling frequency (Hz)
        fl: Lower cutoff frequency (Hz)
        fu: Upper cutoff frequency (Hz)
        order: Filter order (default: 512)

    Returns:
        Complex transfer function H(f)
    """
    nyq = fs / 2
    numtaps = order + 1

    # Determine filter type and its cutoff
    if fl is not None and fu is not None:
        ftype = "bandpass"
        cutoff = [fl / nyq, fu / nyq]
    elif fl is not None:
        ftype = "highpass"
        cutoff = fl / nyq
    elif fu is not None:
        ftype = "lowpass"
        cutoff = fu / nyq
    else:
        raise ValueError("At least one of fl or fu must be specified.")

    filt = firwin(
        numtaps=numtaps,
        cutoff=cutoff,
        pass_zero=ftype,
        window="boxcar", # for welch tapering
    )
    filt = filt * welch_window(numtaps)

    w = 2 * np.pi * np.asarray(f) / fs
    _, H = freqz(filt, worN=w)

    return H


# ________________________________________________________________
# Combined transfer function and PSD estimation

def H_seqarima(f: np.ndarray, params: Rist) -> np.ndarray:
    """
    Combined transfer function H(f) for seqARIMA pipeline.

    Args:
        f: Frequency array (Hz)
        params: Rist containing seqARIMA parameters.

    Required params:
        - fs: Sampling frequency (Hz)

    Optional params:
        - d: Differencing order
        - ar_coef: AR coefficients
        - q_list: EoA window sizes
        - fl, fu: Bandpass cutoffs (Hz)
        - bp_order: Bandpass filter order (default: 512)

    Returns:
        Complex transfer function H(f)
    """
    fs = params.fs
    H_out = np.ones_like(f, dtype=complex)

    if has_param(params, "d"):
        H_out *= H_diff(f, fs, params.d)
    if has_param(params, "ar_coef"):
        H_out *= H_ar(f, fs, params.ar_coef)
    if has_param(params, "q_list"):
        H_out *= H_eoa(f, fs, params.q_list)
    if has_param(params, "fl") and has_param(params, "fu"):
        bp_order = params.bp_order if has_param(params, "bp_order") else 512
        H_out *= H_bp(f, fs, params.fl, params.fu, bp_order) ** 2  # filtfilt

    return H_out


def var_seqarima(
    f: np.ndarray, params: Rist, domain: Literal["freq", "time"] = "freq"
) -> float:
    """
    Noise variance after seqARIMA filtering.

    Args:
        f: Frequency array (Hz)
        params: Rist containing seqARIMA parameters.
        domain:
            - "freq": Frequency-domain variance (normalized by bandwidth)
            - "time": Time-domain variance (Parseval's theorem)

    Returns:
        Filtered noise variance
    """
    fs = params.fs
    var_pred = params.var_pred
    df = f[1] - f[0]

    H_total_sq = np.ones_like(f)

    if has_param(params, "q_list"):
        H_total_sq *= np.abs(H_eoa(f, fs, params.q_list)) ** 2

    if has_param(params, "fl") and has_param(params, "fu"):
        bp_order = params.bp_order if has_param(params, "bp_order") else 512
        H_total_sq *= np.abs(H_bp(f, fs, params.fl, params.fu, bp_order)) ** 4

    integral = np.sum(H_total_sq) * df

    if domain == "time":
        # Time-domain: Parseval's theorem σ² = (2/fs) ∫|H|² df
        return var_pred * (2 / fs) * integral
    else:  # "freq"
        # Frequency-domain: normalized by bandwidth
        fl, fu = params.fl, params.fu
        if fu is None:
            fu=fs/2
        if fl is None:
            fl=0
        bw = fu - fl
        return var_pred * (2 / fs) * integral / bw


def psd_seqarima(f: np.ndarray, params: Rist) -> np.ndarray:
    """
    Noise PSD estimated by seqARIMA model.

    S_n(f) = σ² / |H(f)|²

    Args:
        f: Frequency array (Hz)
        params: Rist containing seqARIMA parameters.

    Required params:
        - fs: Sampling frequency (Hz)
        - var_pred: AR prediction variance
        - ar_coef: AR coefficients

    Optional params:
        - d: Differencing order
        - q_list: EoA window sizes
        - fl, fu: Bandpass cutoffs (Hz)
        - bp_order: Bandpass filter order (default: 512)

    Returns:
        PSD array S(f)
    """
    Hf = H_seqarima(f, params)
    var = var_seqarima(f, params)

    return var / np.abs(Hf) ** 2


# Signal-to-noise ratio
def envelope_snr(seqarima_obj) -> np.ndarray:
    """
    Compute envelope SNR time series from seqarima result.

    SNR(t) = A(t) / sqrt(2 * var_filtered)

    where A(t) is the envelope (magnitude of analytic signal).
    Normalized so E[SNR] = 1 for noise only (chi-squared with 2 DOF).

    Args:
        seqarima_obj: seqarima result object with ar_meta attribute.

    Returns:
        ts: SNR time series with variance_result attribute.
    """
    params = extract_seqarima_params(seqarima_obj)
    freqs = np.fft.rfftfreq(seqarima_obj.length, 1 / seqarima_obj.sampling_freq)
    sigma2 = var_seqarima(f=freqs, params=params, domain="time")

    analytic_signal = hilbert(seqarima_obj.data)
    envelope = np.abs(analytic_signal)

    snr_ts = np.sqrt(envelope**2 / (2 * sigma2))
    snr_ts = tsref(snr_ts, seqarima_obj)

    return snr_ts


# ________________________________________________________________
# Prediction using trained seqARIMA model

def pred_seqarima(
    ts_obj: ts,
    params: Rist,
    zero_phase: bool = True,
    verbose: bool = True,
) -> ts:
    """
    Apply seqARIMA filtering using extracted parameters.

    This function applies the same filtering pipeline (Diff -> AR -> MA -> BP)
    using parameters extracted from a trained model via extract_seqarima_params().

    Args:
        ts_obj (ts): Input time series to filter.
        params (Rist): Parameters from extract_seqarima_params() containing:
            - fs: sampling frequency
            - ar_coef (required): AR coefficients
            - var_pred: AR prediction variance
            - d (optional): differencing order
            - q_list (optional): MA window sizes
            - ma_collector (optional): MA aggregation method
            - fl, fu (optional): bandpass cutoffs
            - bp_order (optional): bandpass filter order

    Returns:
        ts: Filtered time series.

    Example:
        >>> # Train on noise data
        >>> arm_noise = bc.seqARIMA.seqarima(noise_data, p=4096, d=2, q=range(1,21), fl=32, fu=512)
        >>>
        >>> # Extract parameters (can be saved/loaded separately)
        >>> params = bc.seqARIMA.extract_seqarima_params(arm_noise)
        >>> params.save("seqarima_params.pkl")
        >>>
        >>> # Apply to target signal
        >>> filtered_signal = bc.seqARIMA.pred_seqarima(signal_data, params=params)
    """
    message_verb("> Applying seqARIMA filtering...", verb=verbose)

    out = ts_obj

    # Step 1: Differencing (optional)
    if has_param(params, "d") and params.d > 0:
        message_verb(f"> (1) Differencing: d={params.d}", verb=verbose)
        out = Differencing(out, d=params.d, frac=params.diff_frac, verbose=False)

    # Step 2: AR filtering (required)
    if not has_param(params, "ar_coef"):
        raise ValueError("params must contain ar_coef (AR stage is required)")

    ar_coef = params.ar_coef
    message_verb(f"> (2) AR filtering: p={len(ar_coef)}", verb=verbose)
    out = pred_resid(out, arcoef=ar_coef)

    # Insert ar_meta
    setattr(out, "ar_meta", Rist(
        ar_coef=params.ar_coef,
        var_pred=params.var_pred,
        parcor=params.parcor,
        p_order=len(ar_coef) if isinstance(ar_coef, np.ndarray) else [len(v) for v in ar_coef.values()],
        ar_collector=params.ar_collector,
    ))
    
    # Zero-phase correction for Diff + AR together
    if zero_phase:
        is_ear = out.ar_meta.ar_collector != "single"
        if is_ear:
            warnings.warn("Zero-phase correction is not supported for ensemble AR. Skipping.")
        else:
            _fs = ts_obj.sampling_freq
            _d_order = out.diff_meta.d_order if (hasattr(out, "diff_meta") and out.diff_meta is not None) else None
            _ar_coef = out.ar_meta.ar_coef
            if _d_order is not None:
                H_func = lambda f: H_diff(f, _fs, _d_order) * H_ar(f, _fs, _ar_coef)
            else:
                H_func = lambda f: H_ar(f, _fs, _ar_coef)
            out_zp = tsref(zero_phasing(out.data, H_func, _fs), out)
            inherit_ts_attrs(out, out_zp)
            out = out_zp

    # Step 3: Moving Average (optional)
    if has_param(params, "q_list"):
        q_list = params.q_list
        ma_collector = params.ma_collector if has_param(params, "ma_collector") else "mean"
        message_verb(f"> (3) Moving Average: q={q_list}, collector={ma_collector}", verb=verbose)
        out = MovingAverage(out, q=q_list, verbose=False, collector=ma_collector)

    # Step 4: Bandpass (optional)
    if has_param(params, "fl") and has_param(params, "fu"):
        fl, fu = params.fl, params.fu
        message_verb(f"> (4) Bandpass: {fl}-{fu} Hz", verb=verbose)
        out = BandPass(out, fl=fl, fu=fu, verbose=False)

    return out


# ________________________________________________________________
# Convenience wrapper: fit + extract parameters

def fit_seqarima(ts_obj, verbose=True, **kwargs):
    """Fit seqARIMA and extract model parameters in one step.

    Args:
        ts_obj: beacon ts time series.
        verbose: print progress.
        **kwargs: forwarded to seqarima() (d, p, q, fl, fu, etc.).

    Returns:
        (fit, params): seqarima result and extracted Rist of parameters.
    """
    fit = seqarima(ts_obj, **kwargs, verbose=verbose)
    params = extract_seqarima_params(fit)
    return fit, params