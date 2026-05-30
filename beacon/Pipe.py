# ===========================================================
# Scrap from other source codes
from .TS import *
from .DQ import *
from .seqARIMA import seqarima
from .plot import message_verb
from .Calc import *
from .etc import Rist

# For type annotations
from typing import Sequence, Optional, Union, List, Callable, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import polars as pl
import pandas as pd

from sklearn.cluster import DBSCAN
from scipy.stats import poisson

# For pipe_net in running parallel
from joblib import Parallel, delayed, parallel_backend
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# For measuring pipe_net() per each batch inside stream()
import time
import os

# For checkpoint saving
import json
import pickle
import pyarrow as pa
import pyarrow.parquet as pq

# For AR veto classification
from scipy.stats import chi2, beta as beta_dist
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture

# For seqARIMA on|off denoising
from .seqARIMA import pred_seqarima, extract_seqarima_params, fit_seqarima

# For AR feature extraction
from .features import (
    extract_trigger_features, extract_raw_features,
    whiten_with_bkg, decompose_vector, get_summary_feature,
)

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
    mask=None,
) -> pl.DataFrame:
    """
    Run DBSCAN clustering on time values where anomaly conditions are met.

    Args:
        anom_df (pl.DataFrame): Input dataframe containing anomaly detection results.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        time_col (str): Column name for time values.
        anomaly_col (str): Column name indicating anomalies.
        mask (pl.Expr, optional): Custom boolean mask for filtering. Defaults to None.

    Returns:
        pl.DataFrame: Original dataframe with an added 'cluster' column.
    """
    # Create a unique ID for each row to safely join clustering results back later
    df = anom_df.with_row_index("__id__")

    # 1. Determine the logical mask for filtering anomalies
    if mask is None:
        if time_col == "time" and anomaly_col == "anomaly":
            mask = pl.col(anomaly_col) == 1
        elif time_col == "time_bin" and anomaly_col == "S":
            # P0_thresh and significance calculation logic for specific case
            p0_thresh = 0.05
            mask = pl.col(anomaly_col) > Significance(p0_thresh, a=1)
        else:
            # Raise error if no predefined mask matches the given column combination
            raise ValueError(
                f"No default mask for time_col='{time_col}' and anomaly_col='{anomaly_col}'. "
                "You must provide a 'mask' argument for this combination."
            )

    # 2. Extract only the rows that meet the anomaly criteria
    target_data = df.filter(mask).select(["__id__", time_col])

    # If no anomalies found, return the dataframe with an empty 'cluster' column
    if target_data.is_empty():
        return anom_df.with_columns(pl.lit(None).cast(pl.Int64).alias("cluster"))

    # 3. Run DBSCAN on the filtered time values
    times = target_data[time_col].to_numpy().reshape(-1, 1)
    t_ref = times.min()
    times_rel = times - t_ref

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(times_rel)

    # Convert noise label (-1) to None and adjust cluster IDs to start from 1
    processed_labels = [int(l) + 1 if l >= 0 else None for l in labels]

    # 4. Map the cluster labels back to the original dataframe using the temporary ID
    cluster_mapping = target_data.with_columns(
        pl.Series("cluster", processed_labels).cast(pl.Int64)
    ).select(["__id__", "cluster"])

    # Perform a left join to append the 'cluster' column and drop the temporary ID
    return df.join(cluster_mapping, on="__id__", how="left").drop("__id__")


def arch(ts_obj: ts, params: Rist, deno_params=None) -> pl.DataFrame:
    """Full pipeline: Denoising -> Anomaly detection -> DBSCAN -> Merge raw.

    Args:
        ts_obj: input time series.
        params: Rist of parameters (nmax, scale, d, p, q, fl, fu, method, eps).
        deno_params: if None, fit seqarima (on|on). If provided, use
            pred_seqarima with these parameters (on|off).

    Returns:
        pl.DataFrame: processed DataFrame.
    """
    if deno_params is None:
        deno = seqarima(
            ts_obj, d=params.d, p=params.p, q=params.q,
            fl=params.fl, fu=params.fu, verbose=False,
        )
    else:
        deno = pred_seqarima(ts_obj, deno_params, verbose=False)

    anom = anomaly(deno, max_anom=params.nmax, scale=params.scale, method=params.method)
    clustered = run_dbscan(anom, eps=params.eps)
    raw_df = as_pl(ts_obj).rename({"x": "raw"})
    merged = clustered.join(raw_df, on="time", how="left")

    base_cols = ["time", "anomaly", "cluster", "raw",
                 "observed", "observed_l1", "observed_l2"]
    extra_cols = [col for col in merged.columns if col not in base_cols]
    return merged.select(base_cols + extra_cols)


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
        DQ="BURST_CAT2",
        n_workers=None,

        # _____Denoising(seqARIMA)_____
        d="auto",
        d_max=2,
        p=1024,
        q=range(1, 21),
        fl=32,
        fu=512,
        seqarima_mode="on|off",  # "on|off": cache params (default).
                                  # "on|on": refit every batch (v7 behavior).

        # _____Anomaly Clustering_____
        nmax=int(f_sampl * t_batch),
        scale=1.5,
        method="iqr",
        decomp=None,
        eps=1 / f_sampl,

        # _____Significance Evaluation_____
        P_update=0.05,  # lambda update cutoff
        smooth="cumulative",  # "cumulative" | "ema" | "kalman"
        smooth_params=None,  # {"alpha": 0.1} or {"q": 1e-4, "r": 1e-2}

        # _____Coincidence Analysis_____
        window_size=128,
        overlap=0.0,

        # _____Autoregressive Veto_____
        feat_dim=32,
        bkg_ref=None,
        fap_c=0.053,
        alpha_d=0.05,
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
    conf["q_max"] = 0 if conf["q"] is None else np.max(conf["q"])
    conf["n_missed"] = tr_overlap(conf["d_max"], conf["p"], conf["q_max"], split=True)

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
    if config['n_workers'] is None:
        n_workers = "number of detector"
    else:
        n_workers = {config["n_workers"]}
        
    lines = []
    lines.append("=" * 60)
    lines.append("BEACON  CONFIGURATION SUMMARY")
    lines.append("=" * 60)

    # Input Data
    lines.append("\n[Input Data]")
    lines.append(f"  Batch Duration     : {config['tbch']} s")
    lines.append(f"  Sampling Frequency : {config['sampling_freq']} Hz")

    # Pipeline Architecture & Data Quality
    lines.append("\n[Pipeline Architecture]")
    lines.append(f"  Data Quality (DQ)  : {config['DQ']}")
    lines.append(f"  Number of Workers  : {n_workers}")

    # seqARIMA Parameters
    lines.append("\n[Sequential ARIMA]")
    lines.append(f"  Differencing (d)   : {config['d']}")
    lines.append(f"  AR order (p)       : {config['p']}")
    q_range = config["q"]
    if hasattr(q_range, "__iter__") and not isinstance(q_range, str):
        try:
            q_str = f"[{min(q_range)}, {max(q_range)}]"
        except (TypeError, ValueError):
            q_str = str(list(q_range))
    else:
        q_str = str(q_range)
    lines.append(f"  MA orders (q)      : {q_str}")
    lines.append(f"  Low freq (fl)      : {config['fl']} Hz")
    lines.append(f"  High freq (fu)     : {config['fu']} Hz")

    # Computed overlap from ARIMA
    n_missed = config["n_missed"]
    lines.append(f"  Head loss (Mh)     : {n_missed[0]} samples")
    lines.append(f"  Tail loss (Mt)     : {n_missed[1]} samples")

    # Anomaly Clustering
    lines.append("\n[Anomaly Clustering]")
    lines.append(f"  Max anomalies      : {config['nmax']}")
    lines.append(f"  IQR scale          : {config['scale']}")
    lines.append(f"  Method             : {config['method']}")
    lines.append(f"  Decomposition      : {config['decomp']}")
    eps_sec = config["eps"]
    sampling_freq = config["sampling_freq"]
    lines.append(
        f"  Epsilon (eps)      : {eps_sec:.6f} sec ({eps_sec * sampling_freq:.2f} samples @ {sampling_freq} Hz)"
    )

    # Significance Evaluation
    lines.append("\n[Significance Evaluation]")
    lines.append(f"  Lambda update P    : {config['P_update']}")
    lines.append(f"  Updating strategy  : {config['smooth']}")
    lines.append(f"  Updating parameter : {config['smooth_params']}")

    # Coincidence Analysis
    lines.append("\n[Coincidence Analysis]")
    lines.append(
        f"  Window size        : {config['window_size']} samples (± {config['window_size'] / 2 / config['sampling_freq'] * 1000} ms)"
    )
    lines.append(f"  Overlap            : {config['overlap'] * 100:.1f}%")

    # Autoregressive Veto
    lines.append("\n[Autoregressive Veto]")
    is_bkg_prepared = config["bkg_ref"] is not None
    lines.append(f"  BKG Pool Prepared? : {is_bkg_prepared}")
    lines.append(f"  Thresh (GW vs NOS) : {config['fap_c']}")
    lines.append(f"  Thresh (BKG vs GLC): {config['alpha_d']}")

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
    """Initialize pipeline components for multiple detectors.

    Returns:
        (prev_batch, res_net, coinc_lis, deno_parlist)
    """
    prev_batch = Rist(**{det: None for det in dets})

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
    res_net = Rist(**{det: res_det.copy() for det in dets})
    coinc_lis = Rist()
    deno_parlist = {det: None for det in dets}

    return prev_batch, res_net, coinc_lis, deno_parlist


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
        #Mt = max_d + max_q // 2
    else:
        Mh = max_d + max_p + (max_q - 1) // 2
        Mt = (max_q - 1) // 2
        #Mt = max_d + (max_q - 1) // 2

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
    rist_append(res_rist, "ustat", res_rist["ustat"][-1])
    res_rist["proc"] = np.nan
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
    elif np.all(np.isnan(prev.data)):
        return curr
    elif prev.length < n_former:
        return curr

    prev_part = prev.data[-n_former:]
    new_data = np.concatenate([prev_part, curr.data])
    new_start = prev.times[-n_former]
    return ts(new_data, start=new_start, sampling_freq=curr.sampling_freq)


def get_transient_ranges(classif_res, coinc_clust):
    """Per-detector transient time ranges from the previous batch.

    GW: shared by both detectors. GLC: per glc_detail tag ((H)/(L)/(HL)).
    Returns {"H1": [...], "L1": [...]} where each item is
    {"t_start": gps, "t_end": gps}.
    """
    out = {"H1": [], "L1": []}
    if classif_res is None or coinc_clust is None:
        return out
    non_bkg = classif_res.filter(pl.col("label") != "BKG")
    if len(non_bkg) == 0:
        return out
    for row in non_bkg.iter_rows(named=True):
        cid = row["coincl_id"]
        ct = coinc_clust.filter(pl.col("coincl_id") == cid)["time_bin"].to_numpy()
        t_range = {"t_start": float(ct.min()), "t_end": float(ct.max())}
        if row["label"] == "GW":
            out["H1"].append(t_range)
            out["L1"].append(t_range)
        else:  # GLC
            if "H" in row["glc_detail"]:
                out["H1"].append(t_range)
            if "L" in row["glc_detail"]:
                out["L1"].append(t_range)
    return out


def compute_n_former(n_default, transient_ranges, curr_start_gps, fs):
    """Choose n_former so concat[0] doesn't fall inside a previous-batch transient.

    Default = n_default (p+d in on|off, pmax+dmax in on|on). If the check_point
    (curr_start_gps - n_default/fs) lies inside any transient range, extend
    back to that transient's t_start so concat[0] sits at the BKG/transient
    boundary. DBSCAN clustering guarantees check_point can be inside at most
    one cluster, so a single pass suffices.
    """
    check_point = curr_start_gps - n_default / fs
    for tr in transient_ranges:
        if tr["t_start"] <= check_point < tr["t_end"]:
            return int((curr_start_gps - tr["t_start"]) * fs)
    return n_default


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
    lambda_c_upd = N_cl_upd / t_batch_upd if t_batch_upd != 0 else np.nan

    # Update lambda_a
    lambda_a_upd = N_anom_upd / N_cl_upd if N_cl_upd != 0 else np.nan

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

def update_stat_smooth(
    upd: Rist, cur: Rist,
    method: str = "ema",
    alpha: float = None,
    N_eff: int = None,
    q: float = 1e-4,
    r: float = 1e-2,
) -> Rist:
    """
    Smooth λ update via fixed-α EMA or adaptive Kalman filter.

    method="ema":
        If N_eff is given:
            α_t = 1 / min(n_batch, N_eff)
            Warm-up: behaves as cumulative average until n_batch reaches N_eff,
            then holds α = 1/N_eff. (Abbott et al. 2020)
        If alpha is given:
            α_t = alpha (fixed, no warm-up)

    method="kalman":
        Scalar Kalman filter on random-walk + noise model (Muth 1960).
        K_t = (P_{t-1} + q) / (P_{t-1} + q + r)    (adaptive gain)
        λ_new = λ_old + K_t * (λ_cur - λ_old)

    Args:
        upd (Rist): Previously updated statistics.
        cur (Rist): Current batch statistics.
        method (str): "ema" for fixed-α, "kalman" for adaptive gain.
        alpha (float or None): Fixed smoothing factor (method="ema", no warm-up).
        N_eff (int or None): Effective window size (method="ema", with warm-up).
            If both alpha and N_eff are None, defaults to alpha=0.1.
        q (float): Process noise variance (method="kalman").
        r (float): Observation noise variance (method="kalman").
    """
    lambda_c_cur = cur.stats["lambda_c"]
    lambda_a_cur = cur.stats["lambda_a"]
    lambda_c_old = upd.stats["lambda_c"]
    lambda_a_old = upd.stats["lambda_a"]

    # Track batch count
    n_batch_prev = upd.stats["n_batch"] if "n_batch" in upd.stats._name_to_index else 1
    n_batch = n_batch_prev + 1

    if method == "kalman":
        P_c = upd.stats["P_c"] if "P_c" in upd.stats._name_to_index else r
        P_a = upd.stats["P_a"] if "P_a" in upd.stats._name_to_index else r

        if np.isnan(lambda_c_old):
            lambda_c_upd, P_c_new = lambda_c_cur, r
        else:
            P_c_pred = P_c + q
            K_c = P_c_pred / (P_c_pred + r)
            lambda_c_upd = lambda_c_old + K_c * (lambda_c_cur - lambda_c_old)
            P_c_new = (1 - K_c) * P_c_pred

        if np.isnan(lambda_a_old):
            lambda_a_upd, P_a_new = lambda_a_cur, r
        else:
            P_a_pred = P_a + q
            K_a = P_a_pred / (P_a_pred + r)
            lambda_a_upd = lambda_a_old + K_a * (lambda_a_cur - lambda_a_old)
            P_a_new = (1 - K_a) * P_a_pred
    else:
        # Determine α_t
        if N_eff is not None:
            alpha_t = 1.0 / min(n_batch, N_eff)
        elif alpha is not None:
            alpha_t = alpha
        else:
            alpha_t = 0.1

        if np.isnan(lambda_c_old):
            lambda_c_upd = lambda_c_cur
        else:
            lambda_c_upd = alpha_t * lambda_c_cur + (1 - alpha_t) * lambda_c_old

        if np.isnan(lambda_a_old):
            lambda_a_upd = lambda_a_cur
        else:
            lambda_a_upd = alpha_t * lambda_a_cur + (1 - alpha_t) * lambda_a_old

        P_c_new, P_a_new = None, None

    t_batch_upd = upd.stats["t_batch"] + cur.stats["t_batch"]
    N_cl_upd = upd.stats["N_cl"] + cur.stats["N_cl"]
    N_anom_upd = upd.stats["N_anom"] + cur.stats["N_anom"]

    result = Rist(
        stats=Rist(
            t_batch=t_batch_upd,
            N_cl=N_cl_upd,
            N_anom=N_anom_upd,
            lambda_c=lambda_c_upd,
            lambda_a=lambda_a_upd,
            n_batch=n_batch,
        )
    )

    if P_c_new is not None:
        result.stats["P_c"] = P_c_new
        result.stats["P_a"] = P_a_new

    return result

def update_logic(
    updated: Optional[Rist],
    current: Optional[Rist],
    P_update: Optional[float] = None,
    proc: Optional[pl.DataFrame] = None,
    prev_tcen: Optional[float] = None,
    smooth: str = "cumulative",
    smooth_params: Optional[dict] = None,
) -> Rist:
    """
    Logic to update statistics given current and previous stats.

    Args:
        updated (Rist or None): Previous updated statistics.
        current (Rist or None): Current batch statistics.
        P_update (float or None): Threshold to apply FAP filtering.
        proc (pl.DataFrame): Full detection result for current batch.
        prev_tcen (float): Previous central time (for stat_anom).
        smooth (str): "cumulative" (default), "ema" (fixed-α), or "kalman" (adaptive).
        smooth_params (dict or None): Parameters for smooth method.
            - "ema": {"alpha": float} or {"N_eff": int}
            - "kalman": {"q": float, "r": float}

    Returns:
        Rist: Updated statistics.
    """
    if updated is None or is_all_nan(updated.stats):
        return current

    elif current is None or is_all_nan(current.stats):
        return updated
    else:
        if P_update is not None:
            proc_filtered = proc.with_columns(
                [
                    pl.when(pl.col("P0") < P_update)
                    .then(0)
                    .otherwise(pl.col("anomaly"))
                    .alias("anomaly")
                ]
            )
            current_filtered = stat_anom(proc_filtered, last_tcen=prev_tcen)
        else:
            current_filtered = current

        if smooth == "cumulative":
            updated_new = update_stat(upd=updated, cur=current_filtered)
        else:
            params = smooth_params or {}
            updated_new = update_stat_smooth(
                upd=updated, cur=current_filtered,
                method=smooth, **params,
            )

        return updated_new


# Main pipeline (for single detector)
def pipe(
    curr_batch: ts, prev_batch: ts, res_list: Rist,
    config: Rist, deno_params=None,
    transient_ranges: Optional[List[dict]] = None,
    verb: bool = True,
) -> Rist:
    """Single-detector pipeline: arch + significance evaluation.

    Args:
        curr_batch: current batch time series.
        prev_batch: previous batch time series.
        res_list: accumulated results for this detector.
        config: pipeline configuration.
        deno_params: seqARIMA parameters for on|off mode (None = on|on).
        transient_ranges: previous-batch transient ranges for THIS detector
            (from `get_transient_ranges`). Used to decide n_former so concat[0]
            doesn't fall inside a transient. None/empty → no extension.
        verb: verbose output.
    """
    if np.all(np.isnan(curr_batch.data)):
        append_result_NaN(res_list)
        message_verb("WARNING: The current batch is NaN", verb=verb)
    else:
        n_missed = config["n_missed"]
        DQ = config["DQ"]
        P_update = config["P_update"]

        # n_default: on|off → fitted p+d, on|on/첫배치 → tr_overlap(pmax,dmax)
        if deno_params is not None:
            n_default = len(deno_params["ar_coef"]) + int(deno_params["d"])
        else:
            n_default = n_missed["Mh"]
        n_former = compute_n_former(
            n_default=n_default,
            transient_ranges=transient_ranges or [],
            curr_start_gps=curr_batch.start,
            fs=curr_batch.sampling_freq,
        )

        proc = arch(
            concat_ts(prev=prev_batch, curr=curr_batch, n_former=n_former),
            config, deno_params=deno_params,
        )
        proc = adjust_proc(proc, curr_batch=curr_batch, n_missed=n_missed)
        if DQ is not None:
            proc = add_DQ(proc, curr_batch)

        prev_updated_stat = res_list["ustat"][-1]
        prev_tcen = prev_updated_stat["last_tcen"]
        current_stat = stat_anom(
            proc, last_tcen=prev_tcen, sampling_freq=curr_batch.sampling_freq
        )
        proc = add_stat(proc, stat_table=current_stat["table"])

        proc = add_Ppois(proc, prev_updated_stat.stats.lambda_a)
        proc = add_Pexp(proc, prev_updated_stat.stats.lambda_c)
        proc = add_P0(proc)
        if DQ is not None:
            proc = add_P0_DQ(proc, DQ)

        updated_stat = update_logic(
            updated=prev_updated_stat,
            current=current_stat,
            P_update=P_update,
            proc=proc,
            prev_tcen=prev_tcen,
            smooth=config["smooth"],
            smooth_params=config["smooth_params"],
        )
        updated_stat["last_tcen"] = get_last_tcen(proc, prev_tcen)

        rist_append(res_list, "stat", current_stat)
        rist_append(
            res_list, "lamb",
            Rist(a=updated_stat["stats"]["lambda_a"],
                 c=updated_stat["stats"]["lambda_c"]),
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
) -> pl.DataFrame:
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

    Returns:
        pl.DataFrame: Time-binned coincident probability result with columns:
            - bin_id: Bin identifier (1-indexed)
            - time_bin: Median time of the bin
            - P0_H1_bin: Aggregated P0 for H1 detector
            - P0_L1_bin: Aggregated P0 for L1 detector
            - P0_net: Coincident probability (P0_H1_bin * P0_L1_bin)
    """
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

    return grouped

def _run_pipe_worker(det, batch_net, prev_batch, res_list_map,
                     config, deno_parlist, transient_ranges_net, verbose):
    """Worker function for parallel pipe execution."""
    return pipe(
        curr_batch=batch_net[det],
        prev_batch=prev_batch[det],
        res_list=res_list_map[det],
        config=config,
        deno_params=deno_parlist[det],
        transient_ranges=transient_ranges_net[det],
        verb=verbose,
    )

def pipe_net(
    batch_net: Rist,
    prev_batch: Rist,
    res_net: Rist,
    coinc_list: Rist,
    config: Rist,
    deno_parlist: dict,
    prev_classif_res=None,
    prev_coinc_clust=None,
    use_thread: bool = True,
    verbose: bool = True,
) -> tuple:
    """BEACON pipeline + AR veto per batch.

    Returns:
        (res_net, prev_batch, coinc_list, classif_res,
         deno_parlist_upd, raw_feature, bkg_flag)
    """
    dets = batch_net.names
    res_list_map = {det: res_net[det].copy() for det in dets}
    max_workers = (config.n_workers or len(dets)) if use_thread else 1

    seqarima_mode = config["seqarima_mode"]
    if seqarima_mode == "on|on":
        deno_parlist = {det: None for det in dets}

    # 이전 배치 classif/coinc → detector별 transient 범위
    transient_ranges_net = get_transient_ranges(prev_classif_res, prev_coinc_clust)

    with parallel_backend("loky", inner_max_num_threads=1):
        res_list = Parallel(n_jobs=max_workers)(
            delayed(_run_pipe_worker)(
                det, batch_net, prev_batch, res_list_map,
                config, deno_parlist, transient_ranges_net, verbose,
            )
            for det in dets
        )
    res_net_updated = Rist(**dict(zip(dets, res_list)))
    for det in dets:
        prev_batch[det] = batch_net[det]

    if verbose:
        for det in dets:
            try:
                lam = res_net_updated[det]["lamb"][-1]
                print(f"  {det}: lambda_c={lam['c']:.3f}, lambda_a={lam['a']:.3f}")
            except Exception:
                print(f"  {det}: lambda not available")

    if any(is_all_nan(res_net_updated[det]["proc"]) for det in dets):
        coinc_res = None
    else:
        try:
            coinc_res = coincide_P0(
                shift_proc=res_net_updated["H1"]["proc"],
                ref_proc=res_net_updated["L1"]["proc"],
                window_size=config["window_size"],
                overlap=config["overlap"],
                mean_func=har_mean,
                p_col=f"P0_{config['DQ']}" if config["DQ"] is not None else "P0",
            )
        except Exception as e:
            print(f"  [coincide_P0 error] {e}")
            coinc_res = None

    if coinc_res is not None:
        coinc_clust, triggers = cluster_coinc_triggers(
            coinc_res, p0_thresh = config["P_update"]
        )
    else:
        coinc_clust = None
        triggers = pl.DataFrame()

    coinc_list.append(coinc_clust)

    bkg_ref = config["bkg_ref"]
    fap_c = config["fap_c"]
    alpha_d = config["alpha_d"]

    raw_feature, classif_res, bkg_flag = ARveto(
        res_net_updated, triggers, bkg_ref, fap_c, alpha_d
    )

    if seqarima_mode == "on|on":
        deno_parlist_upd = {det: None for det in dets}
        fit_status = {det: "skip" for det in dets}
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=max_workers)(
                delayed(update_deno_params)(
                    batch_net[det], config=config,
                    deno_params=deno_parlist[det], isbkg=bkg_flag[det],
                    proc=res_net_updated[det]["proc"],
                    coinc_clust=coinc_clust,
                    classif_res=classif_res,
                    det=det,
                )
                for det in dets
            )
        deno_parlist_upd = {det: r[0] for det, r in zip(dets, results)}
        fit_status = {det: r[1] for det, r in zip(dets, results)}

    return (res_net_updated, prev_batch, coinc_list,
            classif_res, deno_parlist_upd, raw_feature, bkg_flag,
            fit_status)


# Streaming batch data into pipe_net
def _get_summary_schema(dets: List[str]) -> pa.Schema:
    """PyArrow schema for summary.parquet (includes bkg_flag)."""
    return pa.schema([
        ("batch_id", pa.int32()),
        ("detector", pa.string()),
        ("t_batch", pa.float64()),
        ("N_cl", pa.int32()),
        ("N_anom", pa.int32()),
        ("lambda_a", pa.float64()),
        ("lambda_c", pa.float64()),
        ("lambda_a_upd", pa.float64()),
        ("lambda_c_upd", pa.float64()),
        ("bkg_flag", pa.bool_()),
        ("fit_status", pa.string()),
        ("eta", pa.float64()),
    ])


def _build_summary_rows(
    batch_id: int,
    dets: List[str],
    res_net: Rist,
    bkg_flag: dict,
    eta: float,
    fit_status: dict = None,
) -> List[dict]:
    """Build summary rows for current batch (one row per detector, includes bkg_flag)."""
    rows = []
    for det in dets:
        current_stat = res_net[det]["stat"][-1] if len(res_net[det]["stat"]) > 0 else None
        current_lamb = res_net[det]["lamb"][-1] if len(res_net[det]["lamb"]) > 0 else None

        if current_stat is not None and not is_all_nan(current_stat):
            stats = current_stat["stats"]
            t_batch_val = stats["t_batch"]
            N_cl = stats["N_cl"]
            N_anom = stats["N_anom"]
            lambda_a = stats["lambda_a"]
            lambda_c = stats["lambda_c"]
        else:
            t_batch_val = np.nan
            N_cl = 0
            N_anom = 0
            lambda_a = np.nan
            lambda_c = np.nan

        if current_lamb is not None and not is_all_nan(current_lamb):
            lambda_a_upd = current_lamb["a"]
            lambda_c_upd = current_lamb["c"]
        else:
            lambda_a_upd = np.nan
            lambda_c_upd = np.nan

        rows.append({
            "batch_id": batch_id,
            "detector": det,
            "t_batch": float(t_batch_val) if not np.isnan(t_batch_val) else None,
            "N_cl": int(N_cl) if N_cl is not None else None,
            "N_anom": int(N_anom) if N_anom is not None else None,
            "lambda_a": float(lambda_a) if not np.isnan(lambda_a) else None,
            "lambda_c": float(lambda_c) if not np.isnan(lambda_c) else None,
            "lambda_a_upd": float(lambda_a_upd) if not np.isnan(lambda_a_upd) else None,
            "lambda_c_upd": float(lambda_c_upd) if not np.isnan(lambda_c_upd) else None,
            "bkg_flag": bkg_flag[det],
            "fit_status": fit_status[det] if fit_status else None,
            "eta": float(eta),
        })
    return rows


def stream(
    batch_set: Rist,
    config: Rist,
    checkpoint_dir: str,
    use_model: Rist = None,
    verbose: bool = True,
) -> dict:
    """Non-generator batch loop with mandatory checkpoint.

    Phase 1 (config['bkg_ref'] is None):
        All batches assumed BKG (bkg_flag=True). Extracts XH/XL features
        from coinc_clust triggers and accumulates them. After the loop,
        calls fit_bkg_reference -> saves bkg_ref.npz + bkg_fts.pkl.
        Inserts config dict into bkg_ref (for Phase 2 consistency check).
    Phase 2 (config['bkg_ref'] provided):
        Verifies config consistency (bkg_ref['config'] vs current config).
        Performs ARveto classification, saves classif_res per batch.

    Common:
        Saves config.json, proc (with bin_id), coinc_clust, summary
        per batch. Saves deno_params only for BKG-flagged detectors.
        Clears memory after each batch (proc/stat/lamb/coinc).

    Args:
        batch_set: sequence of batches, each a Rist of detectors.
        config: pipeline configuration parameters.
        Must include 'feat_dim' key (AR feature dimension).
        checkpoint_dir: directory for checkpoint output (mandatory).
        use_model: pretrained ustat per detector.
        verbose: print progress information.

    Returns:
        dict with paths and (Phase 1 only) bkg_ref/bkg_fts objects.
    """
    dets = batch_set[0].names
    feat_dim = config["feat_dim"]
    bkg_ref = config["bkg_ref"]
    is_phase1 = bkg_ref is None
    window_size = config["window_size"]
    overlap = config["overlap"]

    if not is_phase1:
        _check_config_consistency(bkg_ref, config)

    os.makedirs(checkpoint_dir, exist_ok=True)
    proc_dir = os.path.join(checkpoint_dir, "proc")
    coinc_dir = os.path.join(checkpoint_dir, "coinc")
    classif_dir = os.path.join(checkpoint_dir, "classif")
    deno_dir = os.path.join(checkpoint_dir, "deno_params")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(coinc_dir, exist_ok=True)
    if not is_phase1:
        os.makedirs(classif_dir, exist_ok=True)
    os.makedirs(deno_dir, exist_ok=True)

    config_path = os.path.join(checkpoint_dir, "config.json")
    config_dict = save_config(config, config_path)

    summary_path = os.path.join(checkpoint_dir, "summary.parquet")
    summary_schema = _get_summary_schema(dets)
    summary_writer = pq.ParquetWriter(summary_path, summary_schema)

    prev_batch, res_net, coinc_lis, deno_parlist = init_pipe(dets)
    if use_model is not None:
        for det in dets:
            res_net[det]["ustat"] = Rist(use_model[det])

    if is_phase1:
        xh_accum, xl_accum = [], []

    eta_lis = []
    prev_classif_res = None
    prev_coinc_clust = None
    for i in range(len(batch_set)):
        batch_id = i + 1
        if verbose:
            print(f"{batch_id}-th batch:")
        start = time.time()

        (
            res_net, prev_batch, coinc_lis,
            classif_res, deno_parlist, raw_feature, bkg_flag,
            fit_status,
        ) = pipe_net(
            batch_net=batch_set[i],
            prev_batch=prev_batch,
            res_net=res_net,
            coinc_list=coinc_lis,
            config=config,
            deno_parlist=deno_parlist,
            prev_classif_res=prev_classif_res,
            prev_coinc_clust=prev_coinc_clust,
            verbose=verbose,
        )

        # n_former 결정용으로 다음 iter에 carry
        prev_classif_res = classif_res
        prev_coinc_clust = coinc_lis[-1] if len(coinc_lis) > 0 else None

        eta = time.time() - start
        eta_lis.append(eta)

        if is_phase1:
            coinc_last = coinc_lis[-1] if len(coinc_lis) > 0 else None
            if coinc_last is not None:
                triggers_p1 = filter_centroid_time(coinc_last)
                if len(triggers_p1) > 0:
                    XH_i, XL_i, _, _ = extract_raw_features(
                        res_net, triggers_p1)
                    xh_accum.append(XH_i)
                    xl_accum.append(XL_i)

        for det in dets:
            proc = res_net[det]["proc"]
            if proc is not None and not is_all_nan(proc):
                proc_with_bid = assign_bin_ids_to_proc(
                    proc, window_size, overlap)
                proc_with_bid.write_parquet(
                    os.path.join(proc_dir,
                                 f"batch_{batch_id:04d}_{det}.parquet"))

        coinc_last = coinc_lis[-1] if len(coinc_lis) > 0 else None
        if coinc_last is not None:
            coinc_last.write_parquet(
                os.path.join(coinc_dir, f"batch_{batch_id:04d}.parquet"))

        if not is_phase1 and classif_res is not None:
            classif_res.write_parquet(
                os.path.join(classif_dir, f"batch_{batch_id:04d}.parquet"))

        for det in dets:
            if fit_status[det] != "skip" and deno_parlist[det] is not None:
                with open(os.path.join(
                        deno_dir,
                        f"batch_{batch_id:04d}_{det}.pkl"), "wb") as f:
                    pickle.dump(deno_parlist[det], f)

        summary_rows = _build_summary_rows(
            batch_id, dets, res_net, bkg_flag, eta, fit_status)
        summary_writer.write_table(
            pa.Table.from_pylist(summary_rows, schema=summary_schema))

        for det in dets:
            res_net[det]["proc"] = None
            res_net[det]["stat"] = Rist()
            res_net[det]["lamb"] = Rist()
            last_ustat = res_net[det]["ustat"][-1]
            res_net[det]["ustat"] = Rist(last_ustat)
        coinc_lis = Rist()

    summary_writer.close()

    model_path = os.path.join(checkpoint_dir, "model.pkl")
    model_state = {det: res_net[det]["ustat"][-1] for det in dets}
    with open(model_path, "wb") as f:
        pickle.dump(model_state, f)

    result = {
        "summary_path": summary_path,
        "model_path": model_path,
        "config_path": config_path,
    }

    if is_phase1:
        XH_pool = (np.vstack(xh_accum) if xh_accum
                   else np.empty((0, feat_dim)))
        XL_pool = (np.vstack(xl_accum) if xl_accum
                   else np.empty((0, feat_dim)))
        print(f"\n[Phase 1 done] Feature pool: "
              f"XH={XH_pool.shape}, XL={XL_pool.shape}")

        bkg_ref_out, bkg_fts_out = fit_bkg_reference(
            XH_pool, XL_pool, n_feat=feat_dim)

        bkg_ref_out["config"] = config_dict

        bkg_ref_path = os.path.join(checkpoint_dir, "bkg_ref.npz")
        np.savez(bkg_ref_path,
                 **{k: v for k, v in bkg_ref_out.items()
                    if isinstance(v, np.ndarray)},
                 **{k: np.array([v]) for k, v in bkg_ref_out.items()
                    if isinstance(v, (int, float,
                                      np.integer, np.floating))})

        bkg_fts_path = os.path.join(checkpoint_dir, "bkg_fts.pkl")
        with open(bkg_fts_path, "wb") as f:
            pickle.dump(bkg_fts_out, f)

        bkg_ref_pkl_path = os.path.join(checkpoint_dir, "bkg_ref.pkl")
        with open(bkg_ref_pkl_path, "wb") as f:
            pickle.dump(bkg_ref_out, f)

        result.update({
            "bkg_ref": bkg_ref_out,
            "bkg_fts": bkg_fts_out,
            "bkg_ref_path": bkg_ref_pkl_path,
            "bkg_fts_path": bkg_fts_path,
        })

    print(f"\n{len(batch_set)} batches done. "
          f"Mean eta={np.mean(eta_lis):.3f}s")
    return result


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
    coinc_res: pl.DataFrame = coincide_P0(
        shift_proc=h1_proc,
        ref_proc=l1_proc,
        n_shift=None,
        window_size=ws,
        overlap=ov,
        step_size=None,
        mean_func=mean_func,
        p_col="P0",
    )

    # --- 6) Return Rist result ---
    return Rist(
        res_net=updated_res_net,
        model_at=model,
        coinc_res=coinc_res,
        batch_num=i_bch,  # keep 1-based index for user-facing consistency
    )


def Significance(P: ArrayLike, a: float = 1.0) -> Union[float, NDArray[np.float64]]:
    """
    Detection Significance from Probability.

    Compute detection significance on a logarithmic scale:
    S = -a * log10(P)

    Parameters
    ----------
    P : ArrayLike
        Probability values (typically 0 <= P <= 1). NaN is preserved.
    a : float, default 1.0
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


# ---------------------------------------------------------------------------
# AR veto helper functions
# ---------------------------------------------------------------------------

def proc2ts(proc, value_col="observed", time_col="time",
            sampling_freq=4096):
    """Convert BEACON proc DataFrame to a beacon.ts time series object.

    Args:
        proc: Polars DataFrame with time and value columns.
        value_col: column name for observed values.
        time_col: column name for GPS time.
        sampling_freq: sampling frequency in Hz.
    """
    return ts(proc[value_col], start=proc[time_col][0],
              sampling_freq=sampling_freq)


def filter_centroid_time(coinc_clust):
    """Select the peak-S row per coincidence cluster (centroid).

    Args:
        coinc_clust: output of cluster_coinc (must have coincl_id column).

    Returns:
        pl.DataFrame with one row per cluster, sorted by time_bin.
    """
    return (
        coinc_clust
        .filter(pl.col("coincl_id").is_not_null())
        .filter(pl.col("S") == pl.col("S").max().over("coincl_id"))
        .unique(subset=["coincl_id"])
        .sort("time_bin")
    )


def cluster_coinc(coinc_res, eps, min_samples, p0_thresh=0.05):
    """Add Significance column and DBSCAN-cluster the coincidence result.

    Args:
        coinc_res: coincidence DataFrame from coincide_P0.
        eps: DBSCAN epsilon (seconds).
        min_samples: DBSCAN min_samples.

    Returns:
        pl.DataFrame with S and coincl_id columns added.
    """
    coinc_res = coinc_res.with_columns(
        [
            pl.col("P0_net")
            .map_elements(lambda x: Significance(x, a=1),
                          return_dtype=pl.Float64)
            .alias("S")
            .fill_nan(0)
        ]
    )
    mask = pl.col("S") > Significance(p0_thresh, a=1)
    db_res = run_dbscan(
        coinc_res, eps, min_samples, time_col="time_bin", anomaly_col="S", mask=mask
    )
    return db_res.rename({"cluster": "coincl_id"})


def cluster_coinc_triggers(coinc_res, eps=32 * 15 / 4096, min_samples=1, p0_thresh=0.05):
    """Cluster coincidence result and extract centroid triggers.

    Args:
        coinc_res: coincidence DataFrame from coincide_P0.
        eps: DBSCAN epsilon (seconds).
        min_samples: DBSCAN min_samples.

    Returns:
        (coinc_clust, triggers) — coinc_clust is the full clustered
        DataFrame; triggers is the centroid rows (or empty DataFrame).
    """
    coinc_clust = cluster_coinc(coinc_res, eps, min_samples, p0_thresh)
    triggers = filter_centroid_time(coinc_clust)
    return coinc_clust, triggers


def collect_bkg_pool(batch_iter, feat_dim=32):
    """Collect BKG trigger AR features across multiple batches.

    Args:
        batch_iter: iterable of (res_net, coinc_clust) tuples.
            coinc_clust should already have S/coincl_id columns.
        feat_dim: maximum AR order (feature dimension).

    Returns:
        (XH_pool, XL_pool) — numpy arrays, shape (n_total, feat_dim).
    """
    xh_list, xl_list = [], []
    for i, (res_net, coinc_clust) in enumerate(batch_iter):
        if coinc_clust is None:
            print(f"  batch {i+1}: coinc_clust=None, skip")
            continue
        triggers = filter_centroid_time(coinc_clust)
        if len(triggers) == 0:
            print(f"  batch {i+1}: +0 triggers")
            continue
        XH, XL, _, _ = extract_raw_features(res_net, triggers)
        xh_list.append(XH)
        xl_list.append(XL)
        n_total = sum(len(x) for x in xh_list)
        print(f"  batch {i+1}: +{len(XH)} -> total {n_total}")

    XH_pool = (np.vstack(xh_list) if xh_list
               else np.empty((0, feat_dim)))
    XL_pool = (np.vstack(xl_list) if xl_list
               else np.empty((0, feat_dim)))
    return XH_pool, XL_pool


def fit_bkg_reference(XH_pool, XL_pool, n_feat=32):
    """Fit BKG reference: whitening + 2D GMM Normal separation + per-IFO chi2 MLE.

    Args:
        XH_pool: H1 AR feature pool, shape (n, n_feat).
        XL_pool: L1 AR feature pool, shape (n, n_feat).
        n_feat: feature dimension (for Beta alpha/beta parameter).

    Returns:
        (bkg_ref, bkg_fts) — bkg_ref contains whitening params and
        distribution parameters; bkg_fts contains diagnostic data.
    """
    ZH, (mu_H, S_H) = whiten_with_bkg(XH_pool, XH_pool)
    ZL, (mu_L, S_L) = whiten_with_bkg(XL_pool, XL_pool)

    dH, uH = decompose_vector(ZH)
    dL, uL = decompose_vector(ZL)

    gmm_input = np.column_stack([np.log10(dH**2), np.log10(dL**2)])
    gmm = GaussianMixture(n_components=2, random_state=42).fit(gmm_input)
    normal_idx = int(np.argmin(gmm.means_.sum(axis=1)))
    mask_normal = gmm.predict(gmm_input) == normal_idx

    neg_loglik = lambda df, data: -np.sum(chi2.logpdf(data, df))
    d2n_H = dH[mask_normal] ** 2
    d2n_L = dL[mask_normal] ** 2
    df_mle_H = float(minimize_scalar(
        neg_loglik, bounds=(1, 200),
        method="bounded", args=(d2n_H,)).x)
    df_mle_L = float(minimize_scalar(
        neg_loglik, bounds=(1, 200),
        method="bounded", args=(d2n_L,)).x)

    alpha_beta = (n_feat - 1) / 2
    n_norm = int(mask_normal.sum())
    print(f"Pool: {len(XH_pool)} triggers")
    print(f"GMM: Normal n={n_norm}, "
          f"Abnormal n={len(XH_pool)-n_norm}")
    print(f"chi2 MLE: H1 df={df_mle_H:.2f}, "
          f"L1 df={df_mle_L:.2f} (theory={n_feat})")

    return (
        {
            "mu_H": mu_H, "S_H": S_H,
            "mu_L": mu_L, "S_L": S_L,
            "df_mle_H": df_mle_H, "df_mle_L": df_mle_L,
            "alpha_beta": alpha_beta,
        },
        {
            "dH": dH, "dL": dL,
            "uH": uH, "uL": uL,
            "gmm": gmm, "mask_normal": mask_normal,
        },
    )


def classify_triggers(summary_feature, bkg_ref, times, coincl_ids,
                      fap_c=0.053, alpha_d=0.05):
    """3-stage classification: C -> GW/NOS, d2 -> BKG/GLC.

    Args:
        summary_feature: dict with d2H, d2L, C arrays.
        bkg_ref: output of fit_bkg_reference (whitening + distribution params).
        times: trigger GPS times.
        coincl_ids: coincidence cluster IDs (for traceability).
        fap_c: significance level for C (iFAR=10yr -> 0.053).
        alpha_d: significance level for d2.

    Returns:
        pl.DataFrame with columns: coincl_id, times, C, p_C,
        d2H, p_dH, d2L, p_dL, label, glc_detail.
    """
    d2H = summary_feature["d2H"]
    d2L = summary_feature["d2L"]
    C = summary_feature["C"]

    ab = bkg_ref["alpha_beta"]
    tau_C = 2 * beta_dist.ppf(1 - fap_c, ab, ab) - 1
    tau_H = chi2.ppf(1 - alpha_d, bkg_ref["df_mle_H"])
    tau_L = chi2.ppf(1 - alpha_d, bkg_ref["df_mle_L"])

    is_gw = C > tau_C
    is_glc_H = d2H > tau_H
    is_glc_L = d2L > tau_L

    p_C = beta_dist.sf((C + 1) / 2, ab, ab)
    p_dH = chi2.sf(d2H, bkg_ref["df_mle_H"])
    p_dL = chi2.sf(d2L, bkg_ref["df_mle_L"])

    labels = np.where(is_gw, "GW",
                      np.where(is_glc_H | is_glc_L, "GLC", "BKG"))

    glc_detail = np.array([
        f"({''.join(p for p, f in [('H', is_glc_H[i]), ('L', is_glc_L[i])] if f)})"
        if labels[i] == "GLC" else ""
        for i in range(len(C))
    ])

    return pl.DataFrame({
        "coincl_id": coincl_ids,
        "times": times,
        "C": C, "p_C": p_C,
        "d2H": d2H, "p_dH": p_dH,
        "d2L": d2L, "p_dL": p_dL,
        "label": labels,
        "glc_detail": glc_detail,
    })


def is_all_bkg(classif_res):
    """Determine if the entire batch is BKG per IFO.

    Args:
        classif_res: pl.DataFrame from classify_triggers.

    Returns:
        dict with H1/L1 boolean values (True = all BKG for that IFO).
    """
    labels = classif_res["label"].to_numpy()
    details = classif_res["glc_detail"].to_numpy()
    is_gw = labels == "GW"
    is_glc = labels == "GLC"
    unsafe_H = is_gw | (is_glc & np.array(["H" in d for d in details]))
    unsafe_L = is_gw | (is_glc & np.array(["L" in d for d in details]))
    return {"H1": not unsafe_H.any(), "L1": not unsafe_L.any()}


def ARveto(res_net, triggers, bkg_ref, fap_c=0.053, alpha_d=0.05):
    """AR feature extraction + classification + BKG determination.

    Skips if no triggers or bkg_ref is None.

    Args:
        res_net: BEACON pipeline result (Rist with H1/L1 proc).
        triggers: centroid trigger DataFrame (from filter_centroid_time).
        bkg_ref: BKG reference from fit_bkg_reference (or None).
        fap_c: C significance level.
        alpha_d: d2 significance level.

    Returns:
        (raw_feature, classif_res, bkg_flag)
    """
    if len(triggers) == 0 or bkg_ref is None:
        return None, None, {"H1": True, "L1": True}

    XH, XL, times, coincl_ids = extract_raw_features(res_net, triggers)
    summ_feat = get_summary_feature(XH, XL, bkg_ref)
    classif_res = classify_triggers(
        summ_feat, bkg_ref, times, coincl_ids, fap_c, alpha_d)
    bkg_flag = is_all_bkg(classif_res)
    return {"H1": XH, "L1": XL}, classif_res, bkg_flag


def _find_unsafe_time_ranges(classif_res, coinc_clust, proc, det,
                             window_size, overlap):
    """Identify time ranges occupied by non-BKG (GW/GLC) triggers.

    Traces: classif_res[coincl_id] -> coinc_clust[bin_id] -> proc[time].

    Args:
        classif_res: classification result (pl.DataFrame).
        coinc_clust: coincidence cluster DataFrame with coincl_id, bin_id.
        proc: per-detector proc DataFrame.
        det: detector name ('H1' or 'L1').
        window_size: coincide_P0 window size (in samples).
        overlap: coincide_P0 overlap fraction.

    Returns:
        Sorted, merged list of (t_start, t_end) GPS time tuples.
    """
    labels = classif_res["label"].to_numpy()
    details = classif_res["glc_detail"].to_numpy()
    coincl_ids = classif_res["coincl_id"].to_numpy()

    ifo_char = det[0]  # 'H' or 'L'
    is_gw = labels == "GW"
    is_glc_det = (labels == "GLC") & np.array([ifo_char in d for d in details])
    unsafe_mask = is_gw | is_glc_det

    if not unsafe_mask.any():
        return []

    unsafe_ids = set(coincl_ids[unsafe_mask].tolist())

    # bin_ids occupied by unsafe coincl_ids
    unsafe_bins = (
        coinc_clust
        .filter(pl.col("coincl_id").is_in(list(unsafe_ids)))
        ["bin_id"].unique().to_list()
    )
    if not unsafe_bins:
        return []

    # Compute time ranges from bin geometry (same as coincide_P0 binning)
    n = proc.height
    step_size = max(1, int((1 - overlap) * window_size))
    start_indices = np.arange(0, n - window_size + 1, step_size)
    times = proc["time"].to_numpy()

    ranges = []
    for bid in sorted(unsafe_bins):
        idx = bid - 1  # 1-indexed -> 0-indexed
        if 0 <= idx < len(start_indices):
            row_start = start_indices[idx]
            row_end = min(row_start + window_size - 1, n - 1)
            ranges.append((times[row_start], times[row_end]))

    # Merge overlapping ranges
    ranges.sort()
    merged = []
    for t_min, t_max in ranges:
        if merged and t_min <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t_max))
        else:
            merged.append((t_min, t_max))

    return merged


def _longest_clean_ts(curr_ts, gated_ranges, min_duration_s=1.5):
    """Find the longest trigger-free segment in the current batch.

    Args:
        curr_ts: current batch time series (ts object).
        gated_ranges: sorted, merged list of (t_start, t_end) to exclude.
        min_duration_s: minimum acceptable segment duration in seconds.

    Returns:
        ts object of the longest clean segment, or None if all < min_duration_s.
    """
    fs = curr_ts.sampling_freq
    t_start = curr_ts.start
    t_end = t_start + len(curr_ts.data) / fs

    if not gated_ranges:
        return curr_ts

    # Build clean gaps: regions between gated ranges
    gaps = []
    prev_end = t_start
    for g_start, g_end in gated_ranges:
        if g_start > prev_end:
            gaps.append((prev_end, g_start))
        prev_end = max(prev_end, g_end)
    if prev_end < t_end:
        gaps.append((prev_end, t_end))

    if not gaps:
        return None

    durations = [g[1] - g[0] for g in gaps]
    best_idx = int(np.argmax(durations))

    if durations[best_idx] < min_duration_s:
        return None

    g_start, g_end = gaps[best_idx]
    i_start = max(0, int(round((g_start - t_start) * fs)))
    i_end = min(len(curr_ts.data), int(round((g_end - t_start) * fs)))

    return ts(curr_ts.data[i_start:i_end], start=g_start, sampling_freq=fs)


def update_deno_params(curr, config, deno_params, isbkg,
                       proc=None, coinc_clust=None, classif_res=None,
                       det=None, min_clean_factor=1.5):
    """Refit seqARIMA parameters based on BKG classification.

    - isbkg=True: refit on full batch.
    - isbkg=False:
        - If the batch contains any GW trigger, skip the refit entirely
          so signal is never absorbed into the noise model.
        - Otherwise gate the GLC-occupied ranges via coinc_clust tracing
          and refit on the longest clean segment, provided it is at least
          min_clean_factor * (p / fs) seconds long. If none qualifies,
          keep previous params unchanged.

    Args:
        curr: current batch time series.
        config: pipeline configuration.
        deno_params: current seqARIMA parameters.
        isbkg: whether this detector's batch is all-BKG.
        proc: per-detector proc DataFrame (needed when isbkg=False).
        coinc_clust: coincidence cluster DataFrame (needed when isbkg=False).
        classif_res: classification result DataFrame (needed when isbkg=False).
        det: detector name, e.g. 'H1' (needed when isbkg=False).
        min_clean_factor: minimum clean segment duration as a multiple of the
            AR filter length p / fs. Scaling with the AR order preserves a
            fixed samples-per-coefficient margin (default 1.5).

    Returns:
        (deno_params, fit_status) where fit_status is
        "full" | "gated" | "skip".
    """
    if isbkg:
        _, deno_params = fit_seqarima(
            curr, d=config["d"], p=config["p"],
            q=config["q"], fl=config["fl"],
            fu=config["fu"], verbose=False,
        )
        return deno_params, "full"

    if (proc is not None and coinc_clust is not None
          and classif_res is not None and det is not None):
        # GW-containing batch: skip the refit entirely (conservative).
        if (classif_res["label"].to_numpy() == "GW").any():
            return deno_params, "skip"

        min_clean_s = min_clean_factor * config["p"] / config["sampling_freq"]
        gated_ranges = _find_unsafe_time_ranges(
            classif_res, coinc_clust, proc, det,
            config["window_size"], config["overlap"],
        )
        clean_seg = _longest_clean_ts(curr, gated_ranges, min_clean_s)
        if clean_seg is not None:
            _, deno_params = fit_seqarima(
                clean_seg, d=config["d"], p=config["p"],
                q=config["q"], fl=config["fl"],
                fu=config["fu"], verbose=False,
            )
            return deno_params, "gated"

    return deno_params, "skip"


def assign_bin_ids_to_proc(proc, window_size, overlap):
    """Add bin_id column to proc (reproduces coincide_P0 binning).

    Each proc row is assigned to the latest bin that contains it
    (unique 1:1 mapping). If proc height < window_size, bin_id = None.

    Args:
        proc: Polars DataFrame from the pipeline.
        window_size: coincidence window size in samples.
        overlap: fractional overlap in [0, 1).

    Returns:
        pl.DataFrame with bin_id column appended.
    """
    n = proc.height
    step_size = max(1, int((1 - overlap) * window_size))
    start_indices = np.arange(0, n - window_size + 1, step_size)
    n_bins = len(start_indices)

    if n_bins == 0:
        return proc.with_columns(
            pl.lit(None).cast(pl.Int32).alias("bin_id"))

    indices = np.arange(n)
    bin_assignments = (
        np.searchsorted(start_indices, indices, side="right") - 1)
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1) + 1

    return proc.with_columns(
        pl.Series("bin_id", bin_assignments).cast(pl.Int32))


# ---------------------------------------------------------------------------
# Config save / load / consistency
# ---------------------------------------------------------------------------

def _to_json_safe(val):
    """Recursively convert a value to a JSON-safe type."""
    if val is None:
        return None
    if isinstance(val, range):
        return list(val)
    if isinstance(val, Rist):
        return {k: _to_json_safe(v)
                for k, v in zip(val.names, val.values)
                if k is not None}
    if isinstance(val, dict):
        return {k: _to_json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_json_safe(v) for v in val]
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _serialize_config(config):
    """Convert config (Rist) to a JSON-safe dict (no file I/O).

    Excludes: bkg_ref (non-serializable numpy arrays).
    """
    CONFIG_EXCLUDE_KEYS = {"bkg_ref"}
    d = {}
    for name in config.names:
        if name is None or name in CONFIG_EXCLUDE_KEYS:
            continue
        d[name] = _to_json_safe(config[name])
    return d


def save_config(config, path):
    """Save config as a human-readable JSON file.

    Args:
        config: pipeline configuration Rist.
        path: output JSON file path.

    Returns:
        dict: the saved JSON-safe config dict.
    """
    config_dict = _serialize_config(config)
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    return config_dict


def load_config(path):
    """Load JSON config and restore types.

    Args:
        path: path to config.json.

    Returns:
        dict with properly typed values.
    """
    with open(path, "r") as f:
        d = json.load(f)

    for k in ("d_max", "p", "fl", "nmax", "window_size", "q_max"):
        if k in d and d[k] is not None:
            d[k] = int(d[k])

    for k in ("sampling_freq", "scale", "eps", "P_update", "overlap",
              "fap_c", "alpha_d"):
        if k in d and d[k] is not None:
            d[k] = float(d[k])

    if "d" in d and d["d"] is not None and d["d"] != "auto":
        d["d"] = int(d["d"])
    if "fu" in d and d["fu"] is not None:
        d["fu"] = int(d["fu"])
    if "q" in d and isinstance(d["q"], list):
        d["q"] = [int(v) for v in d["q"]]
    if "n_missed" in d and isinstance(d["n_missed"], dict):
        d["n_missed"] = {k: int(v) for k, v in d["n_missed"].items()}
    if "smooth_params" in d and isinstance(d["smooth_params"], dict):
        d["smooth_params"] = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in d["smooth_params"].items()
        }

    return d


def _extract_consistency_dict(config_dict):
    """Extract consistency-relevant keys from a config dict."""
    CONFIG_CONSISTENCY_KEYS = [
        "d", "d_max", "p", "q", "fl", "fu",
        "seqarima_mode",
        "tbch", "sampling_freq",
        "nmax", "scale", "method", "eps",
        "P_update", "smooth", "smooth_params",
        "window_size", "overlap",
    ]
    return {k: config_dict[k]
            for k in CONFIG_CONSISTENCY_KEYS if k in config_dict}


def _check_config_consistency(bkg_ref, config):
    """Verify Phase 1 <-> Phase 2 config consistency.

    Raises ValueError if any consistency key differs between
    bkg_ref['config'] and the current config.
    """
    if "config" not in bkg_ref:
        print("[WARNING] bkg_ref has no 'config' key — "
              "skipping consistency check (legacy bkg_ref)")
        return
    CONFIG_CONSISTENCY_KEYS = [
        "d", "d_max", "p", "q", "fl", "fu",
        "seqarima_mode",
        "tbch", "sampling_freq",
        "nmax", "scale", "method", "eps",
        "P_update", "smooth", "smooth_params",
        "window_size", "overlap",
    ]
    ref_config = bkg_ref["config"]
    ref_consistency = _extract_consistency_dict(ref_config)
    curr_config = _serialize_config(config)
    curr_consistency = _extract_consistency_dict(curr_config)

    mismatches = []
    for k in CONFIG_CONSISTENCY_KEYS:
        ref_val = ref_consistency.get(k)
        curr_val = curr_consistency.get(k)
        if ref_val != curr_val:
            mismatches.append(
                f"  {k}: bkg_ref={ref_val!r}  vs  current={curr_val!r}")

    if mismatches:
        msg = ("Phase 1 (BKG pool) <-> Phase 2 (main search) "
               "config mismatch:\n")
        msg += "\n".join(mismatches)
        raise ValueError(msg)
