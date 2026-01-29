# ===========================================================
# Scrap from other source codes
from .TS import *
from .DQ import *
from .seqARIMA import seqarima
from .Calc import *
from .etc import Rist

# For type annotations
from typing import Sequence, Optional, Union, List, Callable, Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
import polars as pl
import pandas as pd

from sklearn.cluster import DBSCAN
from scipy.stats import poisson

# For pipe_net in running parallel
from concurrent.futures import ThreadPoolExecutor

# For measuring pipe_net() per each batch inside stream()
import time

# Turn off KPSS interpolation warning
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.filterwarnings("ignore", category=InterpolationWarning)
# ===========================================================


# Misc. ----
def nan_to_null(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    """
    Replace NaN values with nulls in specified float64 columns.

    This function targets only the user-specified columns and replaces
    any `np.nan` values with Polars `null`. It is useful for unifying
    missing-value representations in downstream computations such as
    joins, filtering, or log transformations.

    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        cols (Sequence[str]): List of column names to process. Only columns of float64 type are affected.

    Returns:
        pl.DataFrame: A new DataFrame where the specified float64 columns
                      have NaN values replaced with null.
    """
    target_cols = [c for c in cols if c in df.columns and df.schema[c] == pl.Float64]

    return df.with_columns(
        [
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c in target_cols
        ]
    )


def as_pl(ts_obj) -> pl.DataFrame:
    """
    Convert ts object to polars DataFrame with time and value columns.

    Args:
        ts_obj (ts): Custom time series object.

    Returns:
        pl.DataFrame: DataFrame with 'time' and 'x' columns.
    """
    return pl.DataFrame({"time": ts_obj.times, "x": ts_obj.data})


def transpose_Rist(rist: Rist) -> Rist:
    """
    Reshape a Rist of Rist objects into a Rist where each element is a Rist
    of corresponding named elements (with names preserved).

    Args:
        rist (Rist): Rist of Rist objects with named elements.

    Returns:
        Rist: Transposed Rist where each name maps to a Rist of values across input Rists.
    """
    if not rist:
        return Rist()

    names = rist[0].names  # Assume all Rist have same keys
    transposed = {}

    for name in names:
        values = [r[name] for r in rist]
        transposed[name] = Rist(
            dict(zip(rist.names, values))
        )  # preserve detector names (e.g., H1, L1)

    return Rist(transposed)


# Preparing batches ----
# batching function for single detector
def batching(ts_obj: ts, t_bch: float = 1.0, has_DQ: bool = True) -> Rist:
    """
    Split a ts object into batches. Distribute DQ info per batch,
    and retain general meta info in the returned Rist container.

    Args:
        ts_obj (ts): Input full time series object.
        t_bch (float): Desired batch duration in seconds.
        has_DQ (bool): Whether to split and assign dqmask per batch.

    Returns:
        Rist: Named Rist of batch ts objects, with shared .meta on container.
    """
    sampling_freq = ts_obj.sampling_freq
    total_len = len(ts_obj.data)
    n_bch = int(np.round(ts_obj.duration / t_bch))
    batch_len = int(t_bch * sampling_freq)

    batches = []
    # time_index = ts_obj.start + np.arange(total_len) / sampling_freq

    dq_df = None
    dq_level = None
    if has_DQ and hasattr(ts_obj, "meta"):
        dq_meta = ts_obj.meta["DQ"]
        if dq_meta is not None:
            dq_df = dq_meta["dqmask"]
            dq_level = dq_meta["level"]

    for i in range(n_bch):
        start_idx = i * batch_len
        end_idx = min((i + 1) * batch_len, total_len)
        if start_idx >= end_idx:
            continue

        data_slice = ts_obj.data[start_idx:end_idx]
        start_time = ts_obj.start + start_idx / sampling_freq
        batch_ts = ts(data_slice, start=start_time, sampling_freq=sampling_freq)

        # Attach only DQ-related meta
        if has_DQ and dq_df is not None:
            t0 = int(np.floor(start_time))
            t1 = int(np.floor(start_time + (end_idx - start_idx) / sampling_freq))
            dq_batched = dq_df.filter(
                (pl.col("t_floor") >= t0) & (pl.col("t_floor") < t1)
            )

            batch_ts.meta = Rist(DQ=Rist(level=dq_level, dqmask=dq_batched))

        batches.append(batch_ts)

    names = [f"batch{str(i + 1).zfill(4)}" for i in range(len(batches))]
    out = Rist(dict(zip(names, batches)))

    # Attach general metadata to the Rist container
    if hasattr(ts_obj, "meta"):
        shared_meta = ts_obj.meta.copy()
        if "DQ" in shared_meta.names:
            del shared_meta["DQ"]  # exclude DQ from top-level
        out.meta = shared_meta

    return out


# batching function for detector network
def batching_network(det_ts: Rist, t_bch: float = 1.0, has_DQ: bool = True) -> Rist:
    """
    Batch multiple detector time series and return batch-major Rist using transpose.

    Args:
        det_ts (Rist): Rist of ts objects per detector (e.g., H1, L1, ...).
        t_bch (float): Batch duration in seconds.
        has_DQ (bool): Whether to handle dqmask splitting.

    Returns:
        Rist: Rist of batches. Each batch contains a Rist of detector ts objects.
              Structure: Rist[batch][detector] = ts
    """
    # Step 1: Apply batching to each detector separately → Rist of Rists
    batched_detectors = Rist(
        {det: batching(det_ts[det], t_bch=t_bch, has_DQ=has_DQ) for det in det_ts.names}
    )

    # Step 2: Transpose to batch-major format
    result = transpose_Rist(batched_detectors)

    # Step 3: Copy meta of each detector
    # * This step is valid only for data loaded by `beacon.IO.read_H5()`
    if hasattr(det_ts[0], "meta"):
        result.meta = Rist({})
        for det in det_ts.names:
            meta = det_ts[det].meta.copy()
            if "DQ" in meta.names:
                del meta["DQ"]
            result.meta[det] = meta

    return result


# Anomaly detection ----
def iqr(
    x: np.ndarray | pl.Series, alpha: float = 0.1, max_anoms: int = 100
) -> pl.DataFrame:
    """
    Detect anomalies using the IQR (Interquartile Range) method.

    Args:
        x (np.ndarray or pl.Series): Input 1D signal.
        alpha (float): Significance level for thresholding. Threshold = (0.15 / alpha) * IQR.
        max_anoms (int): Maximum number of anomalies to report (based on largest deviation).

    Returns:
        pl.DataFrame: Table with index, value, lower/upper bounds, anomaly flags.
    """
    # Convert to NumPy array
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    x = np.asarray(x)
    n = len(x)

    # Compute IQR bounds
    q25, q75 = np.nanpercentile(x, [25, 75])
    iqr_val = q75 - q25
    factor = 0.15 / alpha
    lower, upper = q25 - factor * iqr_val, q75 + factor * iqr_val
    center = (lower + upper) / 2
    dist = np.abs(x - center)

    # Determine outliers and direction
    is_outlier = (x < lower) | (x > upper)
    direction = np.where(x > upper, "Up", np.where(x < lower, "Down", None))

    # Construct full DataFrame
    df = pl.DataFrame(
        {
            "index": np.arange(n, dtype=np.uint32),
            "value": x,
            "limit_lower": np.full(n, lower),
            "limit_upper": np.full(n, upper),
            "outlier": is_outlier.astype(int),
            "direction": direction,
            "sorting": dist,
        }
    )

    # Filter outliers only
    df_out = df.filter(pl.col("outlier") == 1)

    # Rank outliers by deviation from center
    df_out = df_out.sort("sorting", descending=True).with_columns(
        [
            pl.Series("rank", np.arange(1, len(df_out) + 1, dtype=np.uint32)),
            (pl.arange(1, len(df_out) + 1) <= max_anoms)
            .cast(pl.Int8)
            .alias("reported"),
        ]
    )

    # Assign reported = 0 for non-outliers
    df_other = df.filter(pl.col("outlier") == 0).with_columns(
        [
            pl.lit(None, dtype=pl.UInt32).alias("rank"),
            pl.lit(0).cast(pl.Int8).alias("reported"),
        ]
    )

    # Merge and restore original order
    df_final = pl.concat([df_out, df_other]).sort("index")

    # Return full result without renaming
    return df_final.select(
        [
            "index",
            "value",
            "limit_lower",
            "limit_upper",
            "reported",
            "outlier",
            "direction",
        ]
    )


def anomalize(
    data: pl.DataFrame,
    target: str,
    method: str = "iqr",
    alpha: float = 0.1,
    max_anoms: int = 100,
) -> pl.DataFrame:
    """
    Apply anomaly detection on a specific column using IQR or GESD method.

    Args:
        data (pl.DataFrame): Input time series data.
        target (str): Column name to apply anomaly detection on.
        method (str): Anomaly detection method, only 'iqr' is available for the moment.
        alpha (float): Significance level for thresholding.
        max_anoms (int): Maximum number of anomalies to flag.

    Returns:
        pl.DataFrame: DataFrame with anomaly column and threshold bounds.
    """
    if target not in data.columns:
        raise ValueError(f"Column '{target}' not found in input DataFrame.")
    x = data[target]

    if method == "iqr":
        outlier_table = iqr(x, alpha=alpha, max_anoms=max_anoms)
    else:
        raise NotImplementedError("Only 'iqr' method is currently supported.")

    # Rename limit columns directly to match target
    lwr_col = f"{target}_l1"
    upr_col = f"{target}_l2"

    outlier_table = outlier_table.rename(
        {"limit_lower": lwr_col, "limit_upper": upr_col}
    )

    result = (
        data.with_row_index(name="index", offset=0)
        .join(
            outlier_table.select(["index", lwr_col, upr_col, "outlier"]),
            on="index",
            how="left",
        )
        .drop("index")
    )

    result = result.with_columns(
        [pl.col("outlier").fill_null(0).cast(pl.Int8).alias("anomaly")]
    ).drop(["outlier"])

    return result


def anomaly(
    ts_obj, max_anom: int = 100, scale: float = 1.5, method: str = "iqr"
) -> pl.DataFrame:
    """
    High-level wrapper to perform anomaly detection on a ts object.

    Args:
        ts_obj (ts): Time series object.
        max_anom (int): Maximum number of anomalies to detect.
        scale (float): Multiplier for IQR threshold; alpha = 0.15 / scale.
        method (str): Anomaly detection method ('iqr' only).
        tzero (float): Time alignment (unused, reserved).

    Returns:
        pl.DataFrame: Anomaly detection result with bounds and flags.
    """
    # Compute alpha from scale
    alpha = 0.15 / scale

    # Convert ts object to polars DataFrame
    tpl = as_pl(ts_obj).rename({"x": "observed"})

    # Apply anomaly detection
    out = anomalize(
        data=tpl, target="observed", method=method, alpha=alpha, max_anoms=max_anom
    )

    return out


def run_dbscan(
    anom_df: pl.DataFrame,
    eps: float,
    min_samples: int = 1,
    time_col: str = "time",
    anomaly_col: str = "anomaly",
) -> pl.DataFrame:
    """
    Run DBSCAN clustering on time values where anomaly == 1.

    Args:
        anom_df (pl.DataFrame): Anomaly detection result.
        eps (float): DBSCAN epsilon (distance threshold, e.g., 1/fs).
        min_samples (int): Minimum samples per cluster.
        time_col (str): Column containing time values.
        anomaly_col (str): Column indicating anomaly (1 = true anomaly).

    Returns:
        pl.DataFrame: Input DataFrame with added 'cluster' column (null if not clustered).
    """
    df = anom_df
    mask = df[anomaly_col] == 1
    times = df.filter(mask)[time_col].to_numpy().reshape(-1, 1)

    # Run DBSCAN on time values
    if len(times) == 0:
        labels = np.array([], dtype=int)
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = db.fit_predict(times)
        labels = np.where(labels >= 0, labels + 1, -1)  # Shift to start from 1

    # Initialize cluster column with None
    cluster_full = np.empty(len(df), dtype=object)
    cluster_full[:] = None
    anomaly_indices = np.where(mask)[0]

    if labels.size:
        pos = labels > 0
        cluster_full[anomaly_indices[pos]] = labels[pos].astype(int)

    # Add 'cluster' column
    return df.with_columns(
        pl.Series("cluster", cluster_full.tolist()).cast(pl.Int64, strict=False)
    )


def arch(ts_obj: ts, params: Rist) -> pl.DataFrame:
    """
    Full pipeline: Denoising → Anomaly detection → DBSCAN → Merge raw.

    Args:
        ts_obj (ts): Input time series.
        params (Rist): Rist of parameters:
            - nmax (int): Max number of anomalies.
            - scale (float): IQR scale factor.
            - d_max, p_max, q_max: ARIMA orders.
            - fl, fu: Bandpass settings.
            - method, decomp: anomaly() options.
            - eps (float): DBSCAN epsilon.

    Returns:
        pl.DataFrame: Final processed DataFrame.
    """
    # Step 1: Denoising
    deno = seqarima(
        ts_obj,
        d=params.d,
        p=params.p,
        q=params.q,
        fl=params.fl,
        fu=params.fu,
        verbose=False,
    )

    # Step 2: Anomaly detection
    anom = anomaly(deno, max_anom=params.nmax, scale=params.scale, method=params.method)

    # Step 3: DBSCAN
    clustered = run_dbscan(anom, eps=params.eps)

    # Step 4: Merge raw signal
    raw_df = as_pl(ts_obj).rename({"x": "raw"})
    merged = clustered.join(raw_df, on="time", how="left")

    # Step 5: Column arrangement
    base_cols = [
        "time",
        "anomaly",
        "cluster",
        "raw",
        "observed",
        "observed_l1",
        "observed_l2",
    ]
    extra_cols = [col for col in merged.columns if col not in base_cols]
    col_order = base_cols + extra_cols
    merged = merged.select(col_order)

    return merged


# Probabilistic Models ----
def ppois_anom(q: np.ndarray, lam: float) -> np.ndarray:
    """
    Compute normalized Poisson CCDF: P(n >= q) / P(n >= 1)

    Args:
        q (np.ndarray): Array of anomaly counts (n_i).
        lam (float): Estimated lambda for anomaly count (lambda_a).

    Returns:
        np.ndarray: Normalized complementary CDF values.
    """
    q = np.asarray(q)
    ccdf = poisson.sf(q - 1, mu=lam)  # P(n >= q)
    norm = poisson.sf(0, mu=lam)  # P(n >= 1)
    return ccdf / norm


def pexp_cdf(t: np.ndarray, lam: float) -> np.ndarray:
    """
    Exponential CDF: P(t <= T) = 1 - exp(-lambda_c * t)

    Args:
        t (np.ndarray): Waiting times between clusters.
        lam (float): Estimated rate for cluster occurrence (lambda_c).

    Returns:
        np.ndarray: CDF values for waiting times.
    """
    return 1 - np.exp(-lam * t)


def add_Ppois(df: pl.DataFrame, lambda_a: float) -> pl.DataFrame:
    """
    Add Poisson CCDF column (Ppois) to cluster summary.

    Args:
        df (pl.DataFrame): Must include 'N_anom' column.
        lambda_a (float): Estimated lambda for Poisson.

    Returns:
        pl.DataFrame: Updated DataFrame with 'Ppois'.
    """
    ccdf = ppois_anom(df["N_anom"].to_numpy(), lambda_a)
    return df.with_columns(pl.Series("Ppois", ccdf))


def add_Pexp(df: pd.DataFrame, lambda_c: float) -> pl.DataFrame:
    """
    Add Exponential CDF column (Pexp) to cluster summary.

    Args:
        df (pl.DataFrame): Must include 't_lag' column.
        lambda_c (float): Estimated lambda for Exponential.

    Returns:
        pl.DataFrame: Updated DataFrame with 'Pexp'.
    """
    cdf = pexp_cdf(df["t_lag"].to_numpy(), lambda_c)
    return df.with_columns(pl.Series("Pexp", cdf))


def add_P0(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add final combined probability column (P0 = P_NT = Ppois * Pexp) per anomaly.

    Args:
        df (pl.DataFrame): Must include 'anomaly', 'Ppois', 'Pexp'.

    Returns:
        pl.DataFrame: Updated DataFrame with 'P0'.
    """
    return df.with_columns(
        [
            (
                pl.when(pl.col("anomaly") == 1)
                .then(pl.col("Ppois") * pl.col("Pexp"))
                .otherwise(np.nan)
                .alias("P0")
            )
        ]
    )


# Pipeline ----
def config_pipe(replace: Optional[Rist] = None, show_config: bool = True) -> Rist:
    """
    Generate default parameter set for pipe(), with optional replacement via Rist.

    Args:
        replace (Rist, optional): Rist of keys to override default options.
        show_config (bool, optional): If True, print configuration summary. Defaults to True.

    Returns:
        Rist: Default configuration as a Rist object, including 'n_missed'.
    """

    # Default options
    t_batch = 1
    f_sampl = 4096.0
    conf = Rist(
        tbch=t_batch,
        sampling_freq=f_sampl,
        arch=arch,
        DQ="BURST_CAT2",
        n_workers=None, # Defulat is None, this will be handled inside pipe_net()
        # seqARIMA
        d=2,
        p=1024,
        q=range(1, 21),
        fl=32,
        fu=512,
        # Anomaly Detection
        nmax=int(f_sampl * t_batch),
        scale=1.5,
        method="iqr",
        decomp=None,
        # Clustering
        eps=1 / f_sampl,
        # Coincidence
        window_size=128,
        overlap=0.0,
        mean_func=har_mean,
        # lambda update cutoff
        P_update=0.05,
    )

    # Replace values with user-provided overrides
    if replace is not None:
        if not isinstance(replace, Rist):
            raise TypeError("replace must be a Rist.")

        if "tbch" in replace.names:
            t_batch = replace["tbch"]
        if "sampling_freq" in replace.names:
            f_sampl = replace["sampling_freq"]
        conf["nmax"] = int(f_sampl * t_batch)

        for name in replace.names:
            conf[name] = replace[name]

    # Compute n_missed = (Mh, Mt) based on ARIMA loss size
    conf["n_missed"] = tr_overlap(conf["d"], conf["p"], conf["q"], split=True)

    # Print configuration if requested
    if show_config:
        print_config(conf)

    return conf


def _format_config(config: Rist) -> str:
    """
    Build a formatted string summary of pipeline configuration parameters.

    Args:
        config (Rist): Configuration Rist returned by config_pipe().

    Returns:
        str: Multi-line formatted configuration summary.
    """
    if config['n_worker'] is None:
        n_worker = "number of detector"
    else:
        n_worker = {config["n_worker"]}
        
    lines = []
    lines.append("=" * 60)
    lines.append("BEACON  CONFIGURATION SUMMARY")
    lines.append("=" * 60)

    # Input Data
    lines.append("\n[Input Data]")
    lines.append(f"  Batch Duration    : {config['tbch']} s")
    lines.append(f"  Sampling Frequency: {config['sampling_freq']} Hz")

    # Pipeline Architecture & Data Quality
    lines.append("\n[Pipeline Architecture]")
    arch_func = config["arch"]
    arch_name = arch_func.__name__ if callable(arch_func) else str(arch_func)
    lines.append(f"  Processing routine: {arch_name}()")
    lines.append(f"  Data Quality (DQ) : {config['DQ']}")
    lines.append(f"  Number of Workers : {n_worker}")

    # seqARIMA Parameters
    lines.append("\n[Sequential ARIMA]")
    lines.append(f"  Differencing (d)  : {config['d']}")
    lines.append(f"  AR order (p)      : {config['p']}")
    q_range = config["q"]
    if hasattr(q_range, "__iter__") and not isinstance(q_range, str):
        try:
            q_str = f"[{min(q_range)}, {max(q_range)}]"
        except (TypeError, ValueError):
            q_str = str(list(q_range))
    else:
        q_str = str(q_range)
    lines.append(f"  MA orders (q)     : {q_str}")
    lines.append(f"  Low freq (fl)     : {config['fl']} Hz")
    lines.append(f"  High freq (fu)    : {config['fu']} Hz")

    # Computed overlap from ARIMA
    n_missed = config["n_missed"]
    lines.append(f"  Head loss (Mh)    : {n_missed[0]} samples")
    lines.append(f"  Tail loss (Mt)    : {n_missed[1]} samples")

    # Anomaly Detection
    lines.append("\n[Anomaly Detection]")
    lines.append(f"  Max anomalies     : {config['nmax']}")
    lines.append(f"  IQR scale         : {config['scale']}")
    lines.append(f"  Method            : {config['method']}")
    lines.append(f"  Decomposition     : {config['decomp']}")

    # Clustering
    lines.append("\n[Clustering (DBSCAN)]")
    eps_sec = config["eps"]
    sampling_freq = config["sampling_freq"]
    lines.append(
        f"  Epsilon (eps)     : {eps_sec:.6f} sec ({eps_sec * sampling_freq:.2f} samples @ {sampling_freq} Hz)"
    )

    # Coincidence Analysis
    lines.append("\n[Coincidence Analysis]")
    lines.append(
        f"  Window size       : {config['window_size']} samples (± {config['window_size'] / 2 / config['sampling_freq'] * 1000} ms)"
    )
    lines.append(f"  Overlap           : {config['overlap'] * 100:.1f}%")
    mean_func = config["mean_func"]
    mean_func_name = mean_func.__name__ if callable(mean_func) else str(mean_func)
    lines.append(f"  Mean function     : {mean_func_name}")

    # Statistical Thresholds
    lines.append("\n[Statistical Parameters]")
    lines.append(f"  Lambda update P   : {config['P_update']}")

    lines.append("=" * 60)
    return "\n".join(lines)


def print_config(config: Rist, save_path: Optional[str] = None) -> None:
    """
    Print a formatted summary of pipeline configuration parameters,
    and optionally save the same content to a text file.

    Args:
        config (Rist): Configuration Rist returned by config_pipe().
        save_path (str, optional): File path to save the summary. If None, only prints.
    """
    text = _format_config(config)
    print(text)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(text + "\n")


def init_pipe(dets: List[str] = ["H1", "L1"]):
    """
    Initialize pipeline components for multiple detectors.

    Args:
        dets (List[str]): List of detector names.

    Returns:
        Tuple[dict, dict, list]:
            - prev_batch: Rist with empty lists per detector.
            - res_net: Rist with initialized Rist per detector.
            - coinc_lis: Empty Rist for coincidences.
    """
    # Empty previous batch per detector
    prev_batch = Rist(**{det: None for det in dets})

    # Result template per detector
    res_det = Rist(
        proc=Rist(),
        stat=Rist(),
        lamb=Rist(),
        ustat=Rist(
            Rist(
                last_tcen=np.nan,
                stats=Rist(
                    t_batch=0, N_cl=0, N_anom=0, lambda_a=np.nan, lambda_c=np.nan
                ),
            )
        ),
    )

    # Copy template per detector
    res_net = Rist(**{det: res_det.copy() for det in dets})

    # Coincidence list
    coinc_lis = Rist()

    return prev_batch, res_net, coinc_lis


def tr_overlap(d: int, p: int, q: int, split: bool = False) -> Union[int, Rist]:
    """
    Calculate the number of overlapping points needed at head and tail
    when applying seqarima, given maximum ARIMA orders.

    Args:
        max_d (int): Maximum differencing order.
        max_p (int): Maximum AR order.
        max_q (int): Maximum MA order.
        split (bool): If True, return separate Mh (head) and Mt (tail) in a Rist.
                      If False, return total overlap as a single integer.

    Returns:
        int or Rist: Total number of overlapping points, or Rist with Mh and Mt.
    """
    max_d = np.max(d)
    max_p = np.max(p)
    max_q = np.max(q)
    if max_q % 2 == 0:
        Mh = max_d + max_p + max_q // 2
        Mt = max_q // 2
    else:
        Mh = max_d + max_p + (max_q - 1) // 2
        Mt = (max_q - 1) // 2

    if split:
        return Rist(Mh=Mh, Mt=Mt)
    else:
        return Rist(overlap=Mh + Mt)


def is_anomdet(proc: pl.DataFrame) -> bool:
    """
    Check whether any anomaly has been detected in the given DataFrame.

    Args:
        proc (pl.DataFrame): DataFrame containing an 'anomaly' column.

    Returns:
        bool: True if there is at least one anomaly (anomaly == 1), else False.
    """
    if "anomaly" not in proc.columns:
        return False
    return proc.filter(pl.col("anomaly") == 1).height > 0


def is_all_nan(obj) -> bool:
    """
    Return True if all values in the object are NaN.

    Supports: pl.DataFrame, Rist, list, np.ndarray, float, int
    """
    if isinstance(obj, pl.DataFrame):
        return obj.select(pl.all().is_nan()).to_numpy().all()

    elif isinstance(obj, Rist):
        return all(is_all_nan(v) for v in obj.values)

    elif isinstance(obj, (list, np.ndarray)):
        return np.isnan(obj).all()

    elif isinstance(obj, float):
        return np.isnan(obj)

    elif isinstance(obj, int):
        return False  # int는 NaN이 될 수 없음 → 무조건 False

    else:
        raise TypeError(f"is_all_nan() does not support type {type(obj)}")


def rist_append(
    r: Rist, where: Union[int, str], value: Any, name: Optional[str] = None
) -> Rist:
    if name is None:
        r[where].append(value)
    else:
        r[where][name] = value
    return r


def append_result_NaN(res_rist: Rist) -> Rist:
    rist_append(res_rist, "stat", np.nan)
    rist_append(res_rist, "lamb", Rist(a=np.nan, c=np.nan))
    rist_append(res_rist, "prob", np.nan)
    rist_append(res_rist, "proc", np.nan)
    rist_append(res_rist, "updated_stat", np.nan)
    rist_append(res_rist, "current_stat", np.nan)
    return res_rist


def adjust_proc(proc: pl.DataFrame, curr_batch: ts, n_missed: Rist) -> pl.DataFrame:
    """
    Adjust post-detection DataFrame by cropping to current batch time range
    and shifting cluster labels to start from 1.

    This is necessary because some cluster detections may occur
    before the actual batch start due to extended pre-padding for denoising.

    Args:
        proc (pl.DataFrame): Detection result table, includes 'GPS' and 'cluster' columns.
        curr_batch (ts): Time series object for current batch.
        n_missed (Rist): Rist including  "Mt" specifying number of pre-padding points.

    Returns:
        pl.DataFrame: Adjusted DataFrame filtered to current batch range and cluster-shifted.
    """
    # Time window bounds
    t_start = curr_batch.start - (n_missed.Mt / curr_batch.sampling_freq)
    t_end = curr_batch.end

    # Step 1: Crop to target time window
    proc = proc.filter((pl.col("time") >= t_start) & (pl.col("time") <= t_end))

    # Step 2: Cluster shift if anomaly detection was performed
    if is_anomdet(proc):
        first_cluster = proc["cluster"].drop_nulls().min()
        if first_cluster is not None:
            proc = proc.with_columns(
                [(pl.col("cluster") - first_cluster + 1).alias("cluster")]
            )

    return proc


def concat_ts(prev: ts, curr: ts, n_former: int) -> ts:
    """
    Concatenate `n_former` points from `prev` with `curr`.

    Args:
        prev (ts): Previous batch.
        curr (ts): Current batch.
        n_former (int): Number of samples to take from end of prev.

    Returns:
        ts: Concatenated time series.
    """
    if prev is None:
        return curr
    elif prev.length < n_former:
        return curr

    prev_part = prev.data[-n_former:]
    new_data = np.concatenate([prev_part, curr.data])
    new_start = prev.times[-n_former]
    return ts(new_data, start=new_start, sampling_freq=curr.sampling_freq)


def stat_anom(
    proc: pl.DataFrame, last_tcen: Optional[float] = None, sampling_freq: float = 4096.0
) -> Rist:
    """
    Compute statistics of anomaly clusters and estimate global event and cluster rates.

    Args:
        proc (pl.DataFrame): DataFrame with columns ['anomaly', 'cluster', 'time'].
        last_tcen (float, optional): Last center time from previous batch (for t_lag).
        sampling_freq (float): Sampling frequency used to convert time steps into duration.

    Returns:
        Rist:
            - table (pl.DataFrame): Cluster summary with ['cluster', 'N_anom', 't_cen', 't_lag']
            - stats (Rist): Global statistics including:
                - t_batch: Duration of batch
                - N_cl: Number of clusters
                - N_anom: Total number of anomalies
                - lambda_a: Mean anomalies per cluster
                - lambda_c: Mean clusters per unit time
            - last_tcen (float): Final cluster center time for chaining
    """
    # Step 1: filter anomalies with valid clusters
    df = proc.filter((pl.col("anomaly") == 1) & (pl.col("cluster").is_not_null()))

    if df.is_empty():
        # Return empty result consistent with R's init_pipe
        return Rist(
            table=pl.DataFrame(
                {
                    "cluster": [np.nan],
                    "t_cen": [np.nan],
                    "N_anom": [np.nan],
                    "t_lag": [np.nan],
                }
            ),
            stats=Rist(t_batch=0, N_cl=0, N_anom=0, lambda_a=np.nan, lambda_c=np.nan),
            last_tcen=np.nan,
        )

    # Step 2: compute median time (t_cen) per cluster
    table = (
        df.group_by("cluster")
        .agg([pl.median("time").alias("t_cen"), pl.len().alias("N_anom")])
        .sort("t_cen")
    )

    # Step 3: insert last_tcen for t_lag calculation
    t_cens = table["t_cen"].to_numpy()
    if last_tcen is not None and not np.isnan(last_tcen):
        t_lags = np.diff(np.insert(t_cens, 0, last_tcen))
    else:
        t_lags = np.insert(np.diff(t_cens), 0, np.nan)

    # Step 4: attach t_lag column
    table = table.with_columns(pl.Series("t_lag", t_lags))

    # Step 5: compute global statistics
    t_batch = len(proc) / sampling_freq
    N_cl = table.height
    N_anom = int(table["N_anom"].sum())

    lambda_c = N_cl / t_batch if t_batch > 0 else np.nan
    lambda_a = N_anom / N_cl if N_cl > 0 else np.nan

    return Rist(
        table=table,
        stats=Rist(
            t_batch=t_batch,
            N_cl=N_cl,
            N_anom=N_anom,
            lambda_c=lambda_c,
            lambda_a=lambda_a,
        ),
        last_tcen=float(table["t_cen"][-1]),
    )


def add_Pstats(proc: pl.DataFrame, stat: Rist) -> pl.DataFrame:
    """
    Evaluate statistical significance (Poisson × Exponential) per anomaly.

    Args:
        proc (pl.DataFrame): Original sample-level table (includes anomaly, cluster).
        stat (Rist): Output of stat_anom().

    Returns:
        pl.DataFrame: Joined table with Ppois, Pexp, and P0 columns.
    """
    table = stat["table"]
    lambda_a = stat["stats"]["lambda_a"]
    lambda_c = stat["stats"]["lambda_c"]

    # Step 1: Add probability columns
    table = add_Ppois(table, lambda_a)
    table = add_Pexp(table, lambda_c)
    table = nan_to_null(table, cols=["Ppois", "Pexp"])

    # Step 2: Join 'anomaly' info from proc
    table = table.join(proc.select(["cluster", "anomaly"]), on="cluster", how="left")

    # Step 3: Compute P0 = Ppois * Pexp only for anomaly == 1
    table = add_P0(table)

    # Step 4: Merge back to full sample-level data
    final = proc.join(
        table.select(["cluster", "Ppois", "Pexp", "P0"]), on="cluster", how="left"
    )

    return final


def add_DQ(proc: pl.DataFrame, curr_batch: object) -> pl.DataFrame:
    """
    Add data quality (DQ) mask columns into the processed DataFrame.

    Args:
        proc (pl.DataFrame): Processed sample-level data.
        curr_batch (ts): Original ts object with 'meta.DQ' attribute.

    Returns:
        pl.DataFrame: DataFrame with additional DQ columns.
    """
    # Extract mask and level from ts.meta
    dqmask = curr_batch.meta["DQ"]["dqmask"]

    # Floor time to match DQ resolution (1s)
    t_floor = proc["time"].floor().cast(pl.Int64)
    proc = proc.with_columns(t_floor.alias("t_floor"))

    # Join by floored time
    joined = proc.join(dqmask, on="t_floor", how="left")

    # Drop helper column
    return joined.drop("t_floor")


def add_P0_DQ(
    proc: pl.DataFrame, DQ: Union[str, list[str]] = "BURST_CAT2"
) -> pl.DataFrame:
    """
    Add P0_DQ column(s) which mask P0 values by DQ=1.

    Args:
        proc (pl.DataFrame): DataFrame with 'P0' and DQ columns.
        DQ (str or list of str): DQ flag(s) to apply masking. Defaults to 'BURST_CAT2'.

    Returns:
        pl.DataFrame: DataFrame with new column(s) 'P0_{DQ}' added.
    """
    if isinstance(DQ, str):
        DQ = [DQ]

    for dq in DQ:
        proc = proc.with_columns(
            [
                pl.when(pl.col(dq) == 1)
                .then(pl.col("P0"))
                .otherwise(np.nan)
                .alias(f"P0_{dq}")
            ]
        )

    return proc


def add_stat(proc: pl.DataFrame, stat_table: pl.DataFrame) -> pl.DataFrame:
    """
    Join cluster-level statistics (t_cen, N_anom, t_lag) to sample-level data.

    Args:
        proc (pl.DataFrame): Sample-level data with 'cluster' column.
        stat_table (pl.DataFrame): Cluster-level statistics including 'cluster', 't_cen', 'N_anom', 't_lag'.

    Returns:
        pl.DataFrame: Updated sample-level data with added stat columns, inserted after 'cluster'.
    """
    # unify join-key dtype before join (robust to NaN/None)
    proc = proc.with_columns(pl.col("cluster").cast(pl.Int64, strict=False))
    stat_table = stat_table.with_columns(pl.col("cluster").cast(pl.Int64, strict=False))

    # Step 1: Join on 'cluster'
    joined = proc.join(stat_table, on="cluster", how="left")

    # Step 2: Reorder columns to relocate stat columns after 'cluster'
    cluster_idx = joined.columns.index("cluster")
    pre = joined.columns[: cluster_idx + 1]
    stat_cols = ["t_cen", "N_anom", "t_lag"]
    post = [col for col in joined.columns if col not in pre + stat_cols]
    new_order = pre + stat_cols + post

    return joined.select(new_order)


def get_last_tcen(proc: pl.DataFrame, prev_tcen: Optional[float] = None) -> float:
    """
    Extract the latest cluster center time (t_cen) from the result.

    Args:
        proc (pl.DataFrame): DataFrame with 'cluster' and 't_cen' columns.
        prev_tcen (float, optional): Fallback value if no valid cluster is found.

    Returns:
        float: Latest t_cen for the last cluster, or fallback.
    """
    if "cluster" not in proc.columns or "t_cen" not in proc.columns:
        raise ValueError("Required columns 'cluster' and 't_cen' not found in proc.")

    # Drop null clusters and get distinct cluster entries (keep first)
    nonnull = proc.filter(pl.col("cluster").is_not_null())
    if nonnull.is_empty():
        return prev_tcen

    distinct = nonnull.unique(subset=["cluster"], keep="first")
    max_cluster = nonnull["cluster"].max()

    # Get t_cen for the max cluster
    tcen = distinct.filter(pl.col("cluster") == max_cluster)["t_cen"]

    return tcen[0] if len(tcen) > 0 else prev_tcen


def update_stat(upd: Rist, cur: Rist) -> Rist:
    """
    Update cumulative statistics using updated and current statistics.

    Args:
        upd (Rist): Previously updated statistics.
        cur (Rist): Current batch statistics.

    Returns:
        Rist: New updated statistics Rist with 'stats' name.
    """
    # Compute the MOST updated statistics with updated statistics (upd) and current statistics (cur)

    # total batch time
    t_batch_upd = upd.stats["t_batch"] + cur.stats["t_batch"]

    # total cluster number
    N_cl_upd = upd.stats["N_cl"] + cur.stats["N_cl"]

    # total anomaly number
    N_anom_upd = upd.stats["N_anom"] + cur.stats["N_anom"]

    # Update lambda_c
    lambda_c_upd = N_cl_upd / t_batch_upd

    # Update lambda_a
    lambda_a_upd = N_anom_upd / N_cl_upd

    # Return (`last_tcen` will be added in pipe(), outside)
    return Rist(
        stats=Rist(
            t_batch=t_batch_upd,
            N_cl=N_cl_upd,
            N_anom=N_anom_upd,
            lambda_c=lambda_c_upd,
            lambda_a=lambda_a_upd,
        )
    )


def update_logic(
    updated: Optional[Rist],
    current: Optional[Rist],
    P_update: Optional[float] = None,
    proc: Optional[pl.DataFrame] = None,
    prev_tcen: Optional[float] = None,
) -> Rist:
    """
    Logic to update statistics given current and previous stats.

    Args:
        updated (Rist or None): Previous updated statistics.
        current (Rist or None): Current batch statistics.
        P_update (float or None): Threshold to apply FAP filtering.
        proc (pl.DataFrame): Full detection result for current batch.
        prev_tcen (float): Previous central time (for stat_anom).

    Returns:
        Rist: Updated statistics.
    """
    # updated_stat cannot be NA except for the first batch
    if updated is None or is_all_nan(updated.stats):
        # use current_stat
        return current

    elif current is None or is_all_nan(current.stats):
        # use only updated_stat so far
        return updated
    else:
        # Perform updating procedure
        if P_update is not None:
            # Filter out anomalies with FAP < P_update
            proc_filtered = proc.with_columns(
                [
                    pl.when(pl.col("P0") < P_update)
                    .then(0)
                    .otherwise(pl.col("anomaly"))
                    .alias("anomaly")
                ]
            )
            # If computed FAP < P_update, it's less-likely from noise fluctuations.
            # Thus, exclude those FAP < P_update by changing anomaly==1 into anomaly==0
            # Then following function `stat_anom` will filter out anomaly==0 in calculating statistics.

            # Recompute current statistics
            current_filtered = stat_anom(proc_filtered, last_tcen=prev_tcen)

            # Update statistics with filtered statistics
            updated_new = update_stat(upd=updated, cur=current_filtered)
        else:
            # Ordinary updating procedure w/o any filtering
            updated_new = update_stat(upd=updated, cur=current)

        return updated_new


# Main pipeline (for single detector)
def pipe(
    curr_batch: ts, prev_batch: ts, res_list: Rist, arch_params: Rist, verb: bool = True
) -> Rist:
    if np.all(np.isnan(curr_batch.data)):
        # If given current batch is NaN (by such as duty cycle issue)
        append_result_NaN(res_list)
        message_verb(
            "WARNING: The current batch is NaN, might be due to the duty cycle", v=verb
        )
    else:
        arch = arch_params["arch"]
        n_missed = arch_params["n_missed"]
        DQ = arch_params["DQ"]
        P_update = arch_params["P_update"]

        # Processing
        proc = arch(
            concat_ts(prev=prev_batch, curr=curr_batch, n_former=n_missed["Mh"]),
            arch_params,
        )
        proc = adjust_proc(proc, curr_batch=curr_batch, n_missed=n_missed)
        if DQ is not None:
            proc = add_DQ(proc, curr_batch)

        # Compute statistics on current batch
        prev_updated_stat = res_list["ustat"][-1]
        prev_tcen = prev_updated_stat["last_tcen"]
        current_stat = stat_anom(
            proc, last_tcen=prev_tcen, sampling_freq=curr_batch.sampling_freq
        )
        proc = add_stat(proc, stat_table=current_stat["table"])

        # set_trace()

        # Compute probabilities based on prev_updated_stat
        proc = add_Ppois(proc, prev_updated_stat.stats.lambda_a)
        proc = add_Pexp(proc, prev_updated_stat.stats.lambda_c)
        proc = add_P0(proc)
        if DQ is not None:
            proc = add_P0_DQ(proc, DQ)

        # Update statistics with previous updated & current one
        # if P_update != None, `update_logic()` will use prev_tcen and proc inside
        updated_stat = update_logic(
            updated=prev_updated_stat,
            current=current_stat,
            P_update=P_update,
            proc=proc,
            prev_tcen=prev_tcen,
        )

        # Extract the last cluster's t_cen for the next batch
        updated_stat["last_tcen"] = get_last_tcen(proc, prev_tcen)

        # Store results
        rist_append(res_list, "stat", current_stat)
        rist_append(
            res_list,
            "lamb",
            Rist(
                a=updated_stat["stats"]["lambda_a"], c=updated_stat["stats"]["lambda_c"]
            ),
        )
        rist_append(res_list, "ustat", updated_stat)
        res_list["proc"] = proc

    return res_list


# Pipeline on network ----
def coincide_P0(
    shift_proc: pl.DataFrame,
    ref_proc: pl.DataFrame,
    n_shift: Optional[int] = None,
    window_size: int = 100,
    overlap: float = 0.5,
    step_size: Optional[int] = None,
    mean_func: Callable[[pl.Series], float] = har_mean,
    p_col: str = "P0",
    return_mode: int = 1,
) -> Union[pl.DataFrame, Rist]:
    """
    Compute coincident probability (P0) over time-binned windows between two detectors.

    Args:
        shift_proc (pl.DataFrame): Shifted detector data with 'time' and P0 column.
        ref_proc (pl.DataFrame): Reference detector data with 'time' and P0 column.
        n_shift (int, optional): Number of circular shifts to apply to `shift_proc`.
        window_size (int): Size of time window (in rows) for aggregation.
        overlap (float): Fractional overlap between windows.
        step_size (int, optional): Step size between windows. Defaults to (1-overlap) * window_size.
        mean_func (Callable): Aggregation function for each window (default: harmonic mean).
        p_col (str): Column name of per-detector probability.
        return_mode (int): 1 = result only, 2 = joined + result.

    Returns:
        pl.DataFrame or Rist: Time-binned coincident probability result (and optionally, joined raw data).
    """
    if return_mode not in (1, 2):
        raise ValueError("return_mode must be 1 or 2")

    if step_size is None:
        step_size = int((1 - overlap) * window_size)

    # Circular shift (in-place not allowed → do with numpy)
    if n_shift is not None:
        time_col = shift_proc["time"].to_numpy()
        time_col = np.roll(time_col, -n_shift)
        shift_proc = shift_proc.with_columns(pl.Series("time", time_col))

    # Select and rename columns
    shift_proc = shift_proc.select(["time", p_col]).rename({p_col: "P0_H1"})
    ref_proc = ref_proc.select(["time", p_col]).rename({p_col: "P0_L1"})

    # Join by time
    if shift_proc.height > ref_proc.height:
        joined = ref_proc.join(shift_proc, on="time", how="left")
    else:
        joined = shift_proc.join(ref_proc, on="time", how="left")

    # Make time bins
    total_rows = joined.height
    if total_rows < window_size:
        raise ValueError("Not enough rows to perform windowed coincidence analysis.")

    start_indices = np.arange(0, total_rows - window_size + 1, step_size)

    # Bin and tag with bin_id
    bins = [
        joined.slice(start, window_size).with_columns(pl.lit(i + 1).alias("bin_id"))
        for i, start in enumerate(start_indices)
    ]
    joined_overlap = pl.concat(bins, how="vertical")

    # Aggregate over bins
    # Use group-level UDF via map_groups so each group returns scalars (no list dtype)
    def _agg_group(gf: pl.DataFrame) -> pl.DataFrame:
        # Compute per-group scalars with user-supplied mean_func (e.g., har_mean)
        time_bin = float(gf["time"].median())  # time has no NaNs in normal cases
        p0_h1 = gf["P0_H1"].to_numpy()
        p0_l1 = gf["P0_L1"].to_numpy()
        return pl.DataFrame(
            {
                "bin_id": gf["bin_id"][0],
                "time_bin": time_bin,
                "P0_H1_bin": float(mean_func(p0_h1)),
                "P0_L1_bin": float(mean_func(p0_l1)),
            }
        )

    grouped = (
        joined_overlap.group_by("bin_id")
        .map_groups(_agg_group)
        .with_columns(
            (pl.col("P0_H1_bin") * pl.col("P0_L1_bin")).alias(
                "P0_net"
            )  # scalar product
        )
    )

    if return_mode == 1:
        return grouped

    elif return_mode == 2:
        joined_trimmed = joined_overlap.join(
            grouped.select(["bin_id", "P0_net"]), on="bin_id", how="left"
        ).select(["time", "bin_id", "P0_H1", "P0_L1", "P0_net"])

        return Rist(joined=joined_trimmed, result=grouped)


def pipe_net(
    batch_net: Rist,
    prev_batch: Rist,
    res_net: Rist,
    coinc_lis: Rist,
    arch_params: Rist,
    use_thread: bool = True,
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> tuple[Rist, Rist, Rist]:
    """
    Execute the anomaly detection pipeline in parallel across detectors.

    Args:
        batch_net (Rist): Current batch of time series per detector.
        prev_batch (Rist): Previous batch per detector (used to detect edges).
        res_net (Rist): Rist of lists accumulating per-detector pipeline results.
        coinc_lis (Rist): Rist accumulating coincidence analysis results over batches.
        arch_params (Rist): Rist containing pipeline parameters (window_size, overlap, DQ, etc.).
        use_thread (bool): If True, run each detector in parallel using ThreadPoolExecutor.
        verbose (bool): If True, print λ_c and λ_a after each detector update.

    Returns:
        Tuple[Rist, Rist, Rist]: Updated (res_net, prev_batch, coinc_lis).
    """
    dets = batch_net.names

    res_list_map = {det: res_net[det].copy() for det in dets}

    def run_pipe(det):
        return pipe(
            curr_batch=batch_net[det],
            prev_batch=prev_batch[det],
            res_list=res_list_map[det],
            arch_params=arch_params,
            verb=verbose,
        )

    if use_thread:
        max_workers = arch_params.n_workers
        if max_workers is None:
            # If `n_workers` (`max_workers`` here) is not given (given as None (default)), number of detector will be used.
            max_workers = len(dets) 

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            res_list = list(executor.map(run_pipe, dets))
    else:
        res_list = [run_pipe(det) for det in dets]

    res_net_updated = Rist(**dict(zip(dets, res_list)))

    for det in dets:
        prev_batch[det] = batch_net[det]

    if verbose:
        for det in dets:
            try:
                last_lambda = res_net_updated[det]["lamb"][-1]
                lambda_c = last_lambda["c"]
                lambda_a = last_lambda["a"]
                print(f"  {det}: λ_c={lambda_c:.3f}, λ_a={lambda_a:.3f}")
            except Exception:
                print(f"  {det}: λ not available")

    if any(is_all_nan(res_net_updated[det]["proc"]) for det in dets):
        coinc_res = None
    else:
        try:
            coinc_res = coincide_P0(
                shift_proc=res_net_updated["H1"]["proc"],
                ref_proc=res_net_updated["L1"]["proc"],
                window_size=arch_params["window_size"],
                overlap=arch_params["overlap"] if "overlap" in arch_params else 0.0,
                mean_func=(
                    arch_params["mean_func"] if "mean_func" in arch_params else har_mean
                ),
                p_col=(
                    f"P0_{arch_params['DQ']}" if arch_params["DQ"] is not None else "P0"
                ),
                return_mode=2,
            )
        except Exception as e:
            print(f"  [coincide_P0 error] {e}")
            coinc_res = None

    coinc_lis.append(coinc_res)

    return res_net_updated, prev_batch, coinc_lis


# Streaming batch data into pipe_net
def stream(batch_set: Rist, arch_params: Rist, use_model: Rist = None) -> Rist:
    """
    Run full anomaly detection stream over multiple batches.

    Args:
        batch_set (Rist): Sequence of batches, each a Rist of detectors.
        arch_params (Rist): Pipeline configuration parameters.
        use_model (Rist, optional): Pretrained ustat per detector.

    Returns:
        Rist: {
            'res_net': updated results per detector,
            'coinc_lis': coincidence results over batches,
            'model': final ustat per detector,
            'arch_params': unchanged input config,
            'summary': polars.DataFrame summarizing final ustat,
            'eta': list of elapsed times per batch
        }
    """
    dets = batch_set[0].names

    prev_batch, res_net, coinc_lis = init_pipe(dets)

    # If pretrained model is provided
    if use_model is not None:
        for det in dets:
            res_net[det]["ustat"] = Rist(use_model[det])

    eta_lis = []
    for i in range(len(batch_set)):
        print(f"{i + 1}-th batch:")
        start = time.time()

        res_net, prev_batch, coinc_lis = pipe_net(
            batch_net=batch_set[i],
            prev_batch=prev_batch,
            res_net=res_net,
            coinc_lis=coinc_lis,
            arch_params=arch_params
        )

        eta_lis.append(time.time() - start)

    # Summary (last ustat per detector)
    summary_rows = []
    for det in dets:
        last_ustat = res_net[det]["ustat"][-1]
        df_row = pl.DataFrame(last_ustat.to_dict(flat=True)).with_columns(
            pl.lit(det).alias("detector"),
            pl.lit(batch_set[0][det].start).alias("start_time"),
        )
        summary_rows.append(df_row)
    summary_df = pl.concat(summary_rows, how="vertical")

    # Final model: last ustat per detector
    model = Rist(**{det: res_net[det]["ustat"][-1] for det in dets})

    return Rist(
        res_net=res_net,
        coinc_lis=coinc_lis,
        model=model,
        arch_params=arch_params,
        summary=summary_df,
        eta=eta_lis,
    )


def reproduce(
    batch_set: List["Rist"],
    result: "Rist",
    batch_at: Optional[float] = None,
    batch_num: Optional[int] = None,
    window_size: Optional[int] = None,
    overlap: Optional[float] = None,
) -> "Rist":
    """
    Recompute the pipeline and coincidence result for a specific batch.

    This function reconstructs the prior state up to the selected batch and re-executes
    the per-detector pipeline and H1–L1 coincidence analysis for that batch.

    Args:
        batch_set (List[Rist]): Batches produced by `batching_network()`. Each element is a Rist
            with detector keys (e.g., "H1", "L1") mapped to `ts` objects. Each `ts` exposes
            `.trange -> (start, end)` in GPS seconds.
        result (Rist): Output of `stream()`, containing at least:
            - result["res_net"][det] with keys "proc", "stat", "lamb", "ustat"
            - result["arch_params"] with keys "window_size", "overlap", "mean_func"
        batch_at (float, optional): GPS time to locate the batch (inclusive in [start, end]).
        batch_num (int, optional): 1-based index of the batch to reproduce.
        window_size (int, optional): Coincidence window size in samples. If None, taken from
            `result["arch_params"]["window_size"]`.
        overlap (float, optional): Fractional overlap in [0, 1). If None, taken from
            `result["arch_params"]["overlap"]`.

    Returns:
        Rist: Rist(res_net=<Rist by detector>, coinc_res=<pl.DataFrame>, batch_num=<int>)
            - res_net: updated per-detector pipeline result for the chosen batch
            - coinc_res: coincidence result DataFrame (columns include 'P0_net')
            - batch_num: 1-based index of the reproduced batch

    Notes:
        - Detectors are fixed to H1–L1 as per current specification.
        - Column names are not altered inside this function.
        - The function does not mutate the input `result` object.
    """
    # --- 0) Validate exclusive selector ---
    if (batch_at is None and batch_num is None) or (
        batch_at is not None and batch_num is not None
    ):
        raise ValueError("Provide exactly one of 'batch_at' or 'batch_num'.")

    # --- 1) Resolve target batch index (1-based) ---
    if batch_at is not None:
        # Use H1 (preferred) if present, otherwise fall back to the first detector.
        def _pick_detector_name(bnet: "Rist") -> str:
            return "H1" if "H1" in getattr(bnet, "names", []) else bnet.names[0]

        idx_found = None
        for i, bnet in enumerate(batch_set, start=1):  # 1-based loop
            det_name = _pick_detector_name(bnet)
            ts_obj: "ts" = bnet[det_name]
            t0, t1 = ts_obj.trange  # inclusive selection
            if (batch_at >= t0) and (batch_at <= t1):
                idx_found = i
                break
        if idx_found is None:
            raise ValueError("No batch contains the provided 'batch_at' time.")
        i_bch = idx_found
    else:
        # batch_num is 1-based; validate range
        if batch_num < 1 or batch_num > len(batch_set):
            raise IndexError(f"'batch_num' out of range: 1..{len(batch_set)}")
        i_bch = batch_num

    # Convert to Python 0-based for internal indexing
    i_py = i_bch - 1

    # --- 2) Prepare prev/current batches ---
    curr_batch: "Rist" = batch_set[i_py]
    det_names = getattr(curr_batch, "names", [])
    # Ensure H1–L1 fixed pair
    if not (("H1" in det_names) and ("L1" in det_names)):
        raise KeyError(
            "Current version expects both 'H1' and 'L1' detectors in each batch."
        )

    if i_bch == 1:
        # Build a named Rist of None for prev_batch (per detector)
        prev_batch = Rist(**{name: None for name in det_names})
    else:
        prev_batch = batch_set[i_py - 1]

    # --- 3) Trim result.res_net up to the previous batch (no in-place mutation) ---
    arch_params: "Rist" = result["arch_params"]
    base_res_net: "Rist" = result["res_net"].copy()  # deep copy as per your Rist.copy()

    def _slice_list_rist(lst_rist: "Rist", k: int) -> "Rist":
        """Return a new unnamed Rist with the first k elements (k may exceed length safely)."""
        n = len(lst_rist)
        k = max(0, min(k, n))
        items = [lst_rist[j] for j in range(k)]
        return Rist(*items)

    # i_prev replicates R's `i_bch.prev <- ifelse(i_bch == 1, 1, i_bch - 1)`
    i_prev = 1 if i_bch == 1 else (i_bch - 1)

    trimmed_res_net = Rist()
    for det in det_names:
        det_res: "Rist" = base_res_net[det].copy()  # copy per-detector block

        # 'stat' and 'lamb' keep first i_prev elements; 'ustat' keeps first i_bch elements.
        if "stat" in det_res.names:
            det_res["stat"] = _slice_list_rist(det_res["stat"], i_prev)
        if "lamb" in det_res.names:
            det_res["lamb"] = _slice_list_rist(det_res["lamb"], i_prev)
        if "ustat" in det_res.names:
            det_res["ustat"] = _slice_list_rist(det_res["ustat"], i_bch)

        trimmed_res_net[det] = det_res

    # (Optional) Build model from last ustat per detector to mirror R flow (not strictly required by pipe())
    model = Rist(
        **{
            det: trimmed_res_net[det]["ustat"][len(trimmed_res_net[det]["ustat"]) - 1]
            for det in det_names
            if len(trimmed_res_net[det]["ustat"]) > 0
        }
    )
    # We don't assign back to `result`, avoiding side effects.

    # --- 4) Re-run pipe() per detector with trimmed state ---
    updated_res_net = Rist()
    for det in det_names:
        updated_res_net[det] = pipe(
            curr_batch=curr_batch[det],
            prev_batch=prev_batch[det],
            res_list=trimmed_res_net[det],
            arch_params=arch_params,
            verb=False,  # suppress prints/logs
        )

    # --- 5) Coincidence analysis (H1–L1) ---
    ws = arch_params["window_size"] if window_size is None else window_size
    ov = arch_params["overlap"] if overlap is None else overlap
    mean_func = arch_params["mean_func"]

    h1_proc: pl.DataFrame = updated_res_net["H1"]["proc"]
    l1_proc: pl.DataFrame = updated_res_net["L1"]["proc"]

    # Use per-detector P0 (do not rename here; coincide_P0 handles internal renaming)
    coinc_res: pl.DataFrame = cast(
        pl.DataFrame,
        coincide_P0(
            shift_proc=h1_proc,
            ref_proc=l1_proc,
            n_shift=None,
            window_size=ws,
            overlap=ov,
            step_size=None,
            mean_func=mean_func,
            p_col="P0",
            return_mode=1,
        ),
    )

    # --- 6) Return Rist result ---
    return Rist(
        res_net=updated_res_net,
        model_at=model,
        coinc_res=coinc_res,
        batch_num=i_bch,  # keep 1-based index for user-facing consistency
    )


def Significance(P: ArrayLike, a: float = 2.3) -> Union[float, NDArray[np.float64]]:
    """
    Detection Significance from Probability.

    Compute detection significance on a logarithmic scale:
    S = -a * log10(P)

    Parameters
    ----------
    P : ArrayLike
        Probability values (typically 0 <= P <= 1). NaN is preserved.
    a : float, default 2.3
        Positive scaling factor for the significance.

    Returns
    -------
    float or numpy.ndarray
        Detection statistic S. Returns a float if the input is a scalar,
        otherwise a NumPy array with the same shape as `P`.

    Notes
    -----
    - For P == 0, the result is +inf (by definition of log10(0) -> -inf).
    - For P < 0, the result is NaN (follows NumPy's log10 behavior).
    - For P > 1, the result is negative.
    """
    # Convert to ndarray (keeps NaN as NaN; log10 handles edge cases)
    p = np.asarray(P, dtype=np.float64)
    s = -a * np.log10(p)
    # Preserve scalar return for scalar input
    return s.item() if s.ndim == 0 else s
