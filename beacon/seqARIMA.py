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
from scipy.signal import butter, firwin, filtfilt  # For band-pass filter

# Custom modules
from . import _burg as burg  # For AR model estimation via Burg's method
from .TS import *  # For 'ts' class (time series object)
from .plot import message_verb  # For printing message if verbose=True
from .etc import Rist  # For R-style list container

# ================================================================


# ________________________________________________________________
# Differencing (Integrated process)


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
    ts_obj: ts, t_seg: float = 0.5, d_max: int = 2, verbose: bool = True
) -> Rist:
    """
    Automatically determine the differencing order required for stationarity by KPSS test.

    Args:
        ts_obj (ts): Time series object.
        t_seg (float, optional): Segment duration in seconds for stationarity testing. Defaults to 0.5.
        d_max (int, optional): Maximum differencing order. Defaults to 2.
        verbose (bool, optional): If True, print progress messages. Defaults to True.

    Returns:
        Rist: R style list containing:
            - 'd': Selected differencing order.
            - 'out': Differenced time series object.
            - 'p_values': History of p-values per differencing iteration.
    """
    d = 0
    out_data = ts_obj.data.copy()
    pval_history = Rist()

    message_verb(f"|> KPSS test on segments (each {t_seg} second)", verb=verbose)
    message_verb(f"||> d={d}", verb=verbose)

    pvals_df = check_stationary(ts_obj, t_seg)
    pval_history[f"d{d}"] = pvals_df

    while not ((pvals_df >= 0.1).all().all()):
        message_verb(
            "|||> Non-stationary segment detected (p-value < 0.1)", verb=verbose
        )
        d += 1
        out_data = np.diff(out_data, n=1)
        message_verb(f"||> d={d}", verb=verbose)

        ts_diff = ts(out_data, start=ts_obj.start, sampling_freq=ts_obj.sampling_freq)
        pvals_df = check_stationary(ts_diff, t_seg)
        pval_history[f"d{d}"] = pvals_df

        if d_max is not None and d >= d_max:
            break

    out_ts = ts(
        out_data,
        start=ts_obj.start + d * (1 / ts_obj.sampling_freq),
        sampling_freq=ts_obj.sampling_freq,
    )
    return Rist({"d": d, "out": out_ts, "p_values": pval_history})


from typing import Union
import numpy as np


# High-level wrapper to apply differencing manually or via KPSS-based auto differencing
def Differencing(
    ts_obj: ts,
    d: Union[int, str],
    t_seg: float = 0.5,
    return_pvals: bool = False,
    verbose: bool = True,
) -> ts:
    """
    Apply differencing to a time series to induce stationarity.

    Args:
        ts_obj (ts): Input time series object.
        d (int or 'auto'): Differencing strategy.
            - 'auto': perform KPSS-based auto-differencing
            - int > 0: apply fixed-order differencing
            - 0: no differencing
        t_seg (float): Segment length (in seconds) for KPSS test (used only if d='auto')
        return_pvals (bool): Include KPSS p-values in output metadata (only if d='auto')
        verbose (bool): Print progress messages

    Returns:
        ts: Differenced time series object with `.diff_meta` attribute (Rist)
    """

    if d == "auto":
        # KPSS-based auto differencing
        diff_res = auto_diff(ts_obj, t_seg=t_seg, verbose=verbose)
        message_verb(f"|> d={diff_res['d']} selected!", verb=verbose)
        out_ts = diff_res.out
        d_order = diff_res.d
        meta = Rist(method="auto", d_order=d_order, unbounded=True)
        if return_pvals:
            meta["p_values"] = diff_res.p_values

    elif isinstance(d, int) and d > 0:
        # Fixed differencing of order d
        out_data = np.diff(ts_obj.data, n=d)
        out_ts = ts(
            out_data,
            start=ts_obj.start + d * (1 / ts_obj.sampling_freq),
            sampling_freq=ts_obj.sampling_freq,
        )
        d_order = d
        meta = Rist(method="fixed", d_order=d_order)

    elif d == 0:
        # Explicit no differencing
        out_ts = ts_obj
        d_order = 0
        meta = Rist(method=None)

    else:
        raise ValueError("d must be 'auto', 0, or a positive integer.")

    # Inherit attributes and attach metadata
    inherit_ts_attrs(ts_obj, out_ts)
    setattr(out_ts, "diff_meta", meta)

    return out_ts


# ________________________________________________________________
# Autoregressive


# Estimate AR model coefficients using Burg's method
def burgar(
    x: Union[np.ndarray, Sequence[float]],
    ic: Union[str, bool] = True,
    order_max: Optional[int] = None,
    demean: bool = True,
    var_method: int = 2,
) -> Rist:
    """
    Python version of R's ar.burg function (but modified as burgar).
    Estimate AR model coefficients using Burg's method.

    Args:
        x (array-like): Input time series data.
        ic (str or bool, optional): Information criterion to select model order ('AIC', 'BIC', 'FPE', 'AICc', 'KIC', 'AKICc', or bool). Defaults to True.
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
    def AIC(order_max: int, vars_pred: np.ndarray, n_used: int) -> np.ndarray:
        return 2 * np.arange(order_max + 1) + n_used * np.log(vars_pred)

    def BIC(order_max: int, vars_pred: np.ndarray, n_used: int) -> np.ndarray:
        return np.arange(order_max + 1) * np.log(n_used) + n_used * np.log(vars_pred)

    def FPE(order_max: int, vars_pred: np.ndarray, n_used: int) -> np.ndarray:
        orders = np.arange(order_max + 1)
        return (n_used + (orders + 1)) / (n_used - (orders + 1)) * vars_pred

    def AICc(order_max: int, vars_pred: np.ndarray, n_used: int) -> np.ndarray:
        orders = np.arange(order_max + 1)
        return n_used * np.log(vars_pred) + 2 * orders + (2 * orders * (orders + 1)) / (n_used - orders - 1)

    def KIC(order_max: int, vars_pred: np.ndarray, n_used: int) -> np.ndarray:
        orders = np.arange(order_max + 1)
        return n_used * np.log(vars_pred) + 3 * orders

    def AKICc(order_max: int, vars_pred: np.ndarray, n_used: int) -> np.ndarray:
        orders = np.arange(order_max + 1)
        return n_used * np.log(vars_pred) + 3 * orders + (3 * orders * (orders + 1)) / (n_used - orders - 1)

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

    # Information criteria
    if isinstance(ic, bool):
        xic = (
            n_used * np.log(vars_pred)
            + 2 * (np.arange(order_max + 1))
            + 2 * int(demean)
        )
        ic_label = "default"
    else:
        ic_fun = {
            "AIC": AIC,
            "BIC": BIC,
            "FPE": FPE,
            "AICc": AICc,
            "KIC": KIC,
            "AKICc": AKICc,
        }.get(ic)

        if ic_fun is None:
            raise ValueError(f"Unknown ic: {ic}")

        xic = ic_fun(order_max, vars_pred, n_used)
        ic_label = ic

    # Normalize IC
    mic = np.nanmin(xic)
    xic_norm = np.where(np.isfinite(mic), xic - mic, np.where(xic == mic, 0, np.inf))
    # xic_dict = dict(zip(range(order_max + 1), xic_norm))
    attr_ic = ic_label

    # Select order
    if isinstance(ic, bool) and ic is True:
        selected_order = np.flatnonzero(xic_norm == 0)[0]
    elif isinstance(ic, bool) and ic is False:
        selected_order = order_max
    else:
        selected_order = np.flatnonzero(xic_norm == 0)[0]

    # AR coefficients
    if selected_order > 0:
        ar = coefs[:selected_order, selected_order - 1]
    else:
        ar = np.array([])
    var_pred = vars_pred[selected_order]

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


def residual(x: Union[np.ndarray, Sequence[float]], ar: np.ndarray) -> np.ndarray:
    """
    Compute zero-phase AR residuals.

    Calculates AR model residuals with zero-phase correction using FFT-based
    phase adjustment. This function applies an AR filter causally, then corrects
    for phase distortion in the frequency domain to produce zero-phase residuals.

    Args:
        x (array-like): Input time series data.
        ar (np.ndarray): AR coefficients from an AR model fit.

    Returns:
        np.ndarray: Zero-phase corrected residuals with the same length as `x`.
            The first `order` and last `order` values are set to NaN where
            `order = len(ar)`.

    Details:
        The function performs the following steps:
        1. Applies causal AR filter: constructs polynomial (1, -ar) and convolves
        2. Fills NaN values with 0 for FFT computation
        3. Computes phase response of AR coefficients in frequency domain
        4. Corrects phase distortion by multiplying FFT of residuals with exp(-i * phase)
        5. Transforms back to time domain and restores edge NaNs

    See Also:
        burgar, sar
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    ar = np.asarray(ar, dtype=np.float64).ravel()

    # Construct AR coef vector: 1 - a_1 - a_2 - ...
    a = np.insert(-ar, 0, 1.0)
    order = len(ar)

    # Faster than embed-based calculation of residual
    # Causal convolution
    resid = np.convolve(x, a, mode="same")
    # Set first `order` values to NaN (edge effect)
    resid[:order] = np.nan

    # Copy of resid only for FFT input
    resid_filled = resid.copy()
    resid_filled[np.isnan(resid_filled)] = 0.0

    # Phase response of AR coefs
    n = len(x)
    A_f = np.fft.fft(np.r_[a, np.zeros(n - len(a))])
    phase_A = np.angle(A_f)

    # Phase correction
    Y_f = np.fft.fft(resid_filled)
    Y_fp = Y_f * np.exp(-1j * phase_A)
    y_p = np.real(np.fft.ifft(Y_fp))

    # Refill NaN values at edges
    y_p[:order] = np.nan
    y_p[-order:] = np.nan

    return y_p


# Fit a single AR model and return residuals/features
def sar(
    ts_obj: ts,
    ic: Union[str, bool] = True,
    order_max: Optional[int] = None,
    **kwargs: Any,
) -> Rist:
    """
    Fit a single autoregressive (AR) model using Burg's method and return zero-phase residuals and features.

    Args:
        ts_obj (ts): Input time series object.
        ic (str or bool, optional): Information criterion to select model order.
            One of 'AIC', 'BIC', 'FPE', or True/False for default behavior. Defaults to True.
        order_max (int, optional): Maximum AR order to consider. If None, determined automatically.
        **kwargs: Additional arguments passed to `burgar()` (e.g., demean, var_method).

    Returns:
        Rist: Container with:
            - resid: Zero-phase residual time series as `ts` object (NA padding removed).
            - feature: pd.Series of AR coefficients, innovation variance, and input mean.
            - p_order: Selected AR order.
            - ar_collector: Always "single" for SAR.
            - AR_obj: Full AR model result from `burgar()`.
    """
    ar_result = burgar(ts_obj.data, ic=ic, order_max=order_max, **kwargs)
    p = ar_result.order

    # Compute zero-phase residuals using the residual() function
    resids_zp = residual(ts_obj.data, ar_result.ar)
    resids_no_na = resids_zp[~np.isnan(resids_zp)]
    new_start = ts_obj.start + p / ts_obj.sampling_freq
    resids_ts = ts(resids_no_na, start=new_start, sampling_freq=ts_obj.sampling_freq)

    coeff_labels = [f"ar{p}_{i+1}" for i in range(len(ar_result.ar))]
    coeff_series = pd.Series(ar_result.ar, index=coeff_labels)
    extra_series = pd.Series(
        {f"ar{p}_var": ar_result.var_pred, f"ar{p}_mean": ar_result.x_mean}
    )
    feature = pd.concat([coeff_series, extra_series])

    return Rist(
        resid=resids_ts,
        feature=feature,
        p_order=p,
        ar_collector="single",
        AR_obj=ar_result,
    )


# Fit ensemble of AR models and return aggregated residuals/features
def ear(
    ts_obj: ts,
    ps: Sequence[int] = (100, 500, 1000),
    ic: Union[str, bool] = True,
    ar_collector: str = "median",
    return_vec: bool = True,
    return_var: bool = True,
    return_mean: bool = True,
) -> Rist:
    """
    Fit multiple AR models (ensemble) and aggregate residuals and features.

    Args:
        ts_obj (ts): Input time series object.
        ps (Sequence[int], optional): List of AR orders to fit. Defaults to (100, 500, 1000).
        ic (str or bool, optional): Information criterion for order selection. Defaults to True.
        ar_collector (str, optional): Method to aggregate residuals ('median', 'mean', 'pca'). Defaults to 'median'.
        return_vec (bool, optional): If True, return single pd.Series; else list of feature dicts.
        return_var (bool, optional): Include innovation variance in features. Defaults to True.
        return_mean (bool, optional): Include input mean in features. Defaults to True.

    Returns:
        Rist: Container with:
            - resid: Aggregated residuals as `ts` object (aligned and NA-trimmed).
            - feature: Aggregated feature vector (pd.Series or list of dicts).
            - p_order: Selected AR orders.
            - ar_collector: Aggregation method used.
    """
    ar_fits = [burgar(ts_obj.data, ic=ic, order_max=p) for p in ps]
    orders = np.array([fit.order for fit in ar_fits])

    unique_indices = np.unique(orders, return_index=True)[1]
    ar_fits = [ar_fits[i] for i in unique_indices]
    psel = orders[unique_indices]

    resids_list = [fit.resid[fit.order :] for fit in ar_fits]
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

    feat_list = []
    for fit in ar_fits:
        p = fit.order
        feat = {f"ar{p}_{i+1}": coef for i, coef in enumerate(fit.ar)}
        if return_var:
            feat[f"ar{p}_var"] = fit.var_pred
        if return_mean:
            feat[f"ar{p}_mean"] = fit.x_mean
        feat_list.append(feat)

    if return_vec:
        flat = {}
        for d in feat_list:
            flat.update(d)
        feature_out = pd.Series(flat)
    else:
        feature_out = feat_list

    return Rist(
        resid=resids_ts,
        feature=feature_out,
        p_order=psel,
        ar_collector=ar_collector_name,
    )


# High-level wrapper to fit AR model(s) and prepare residuals/features
def Autoregressive(
    ts_obj: ts,
    p: Union[int, Sequence[int]],
    ic: Union[str, bool] = True,
    verbose: bool = True,
    ar_collector: str = "median",
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
            - 'feature': Extracted AR coefficients and statistics.
            - 'p_order': Selected AR order(s).
            - 'ar_collector': Aggregation strategy used.
    """
    if isinstance(p, (list, tuple)) and len(p) > 1:
        result = ear(ts_obj, ps=p, ic=ic, ar_collector=ar_collector, **kwargs)
        message_verb(
            f"|> p={result.p_order} selected and aggregated by: {result.ar_collector}",
            verb=verbose,
        )
    else:
        p_single = p[0] if isinstance(p, (list, tuple)) else p
        result = sar(ts_obj, ic=ic, order_max=p_single, **kwargs)
        message_verb(f"|> p={result.p_order} selected!", verb=verbose)

    resid = result.resid

    # Inherit attributes
    inherit_ts_attrs(ts_obj, resid)

    # Attach features to ts object
    meta = Rist(
        feature=result.feature, p_order=result.p_order, ar_collector=result.ar_collector
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
        filt = firwin(numtaps=n + 1, cutoff=cutoff, pass_zero=ftype, window="hann")
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
    d: Union[int, str, None] = None,
    p: Union[int, Sequence[int], None] = None,
    q: Union[int, Sequence[int], None] = None,
    fl: Optional[float] = None,
    fu: Optional[float] = None,
    ar_collector: str = "median",
    ma_collector: str = "median",
    verbose: bool = True,
) -> ts:
    """
    Sequential ARIMA Denoising:
        Differencing -> AR -> MA -> Bandpass

    Args:
        ts_obj (ts): Input time series.
        d (int or None): Differencing order (None for auto).
        p (int or list of int or None): AR order(s).
        q (int or list of int or None): MA order(s).
        fl (float or None): Bandpass lower frequency bound.
        fu (float or None): Bandpass upper frequency bound.
        ar_collector (str): Aggregation method for AR stage ('median', 'mean', 'pca').
        ma_collector (str): Aggregation method for MA stage ('median', 'mean', 'pca').
        verbose (bool): If True, print progress messages.

    Returns:
        ts: Final output time series with stage meta info as attributes:
            - .diff_meta (from Differencing)
            - .ar_meta   (from Autoregressive)
            - .ma_meta   (from MovingAverage)
            - .bp_meta   (from BandPass)
    """
    message_verb("> Running seqarima...", verb=verbose)

    out = ts_obj

    # Step 1: Differencing
    if d is not None:
        message_verb("> (1) Difference stage", verb=verbose)
        out = Differencing(out, d=d, verbose=verbose)

    # Step 2: Autoregressive
    if p is not None:
        message_verb(f"> (2) Autoregressive stage", verb=verbose)
        out = Autoregressive(
            out, p=p, ic=True, verbose=verbose, ar_collector=ar_collector
        )

    # Step 3: Moving Average
    if q is not None:
        message_verb("> (3) Moving-average stage", verbose)
        out = MovingAverage(out, q=q, verbose=verbose, collector=ma_collector)

    # Step 4: Bandpass Filter
    if (fl or fu) and (fl != 0 or fu != 0):
        message_verb("> (4) Pass filter stage", verb=verbose)
        out = BandPass(out, fl=fl, fu=fu, verbose=verbose)

    return out
