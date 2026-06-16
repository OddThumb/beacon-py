from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.text import Text

# from .QT import qtransform
from matplotlib.colors import Normalize
import seaborn as sns
import polars as pl
from typing import Literal, Optional, Callable, Sequence, Mapping
from .etc import Rist  # For R-style list container
from cycler import cycler
from cmap import Colormap
import matplotlib.colors as mcolors
from adjustText import adjust_text
from pathlib import Path

try:
    from pycbc.types import TimeSeries

    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

# Okabe–Ito colormap
cm = Colormap("okabeito:okabeito")
colors = [mcolors.to_hex(c) for c in cm(np.arange(cm.num_colors))]
plt.rcParams["axes.prop_cycle"] = cycler("color", colors)


# Plot oscillogram
def plot_oscillo(
    ts_obj,
    tzero=None,
    trange=None,
    ylim="pm",
    title=None,
    lw=0.5,
    ax=None,
    color=None,
    label=None,
    figsize=(8, 3),
    xticks=7,
    **kwargs,
):
    """
    Plot oscillogram on provided axis (or create new one if None).

    Args:
        ts_obj: ts object
        tzero: float or None
        trange: (start, end) tuple or None
        ylim: "pm" or (ymin, ymax)
        title: str or None
        lw: line width
        ax: matplotlib axis object (optional)
        color: line color
        label: legend label
        **kwargs: additional plot kwargs
    """
    # Extract data
    times = ts_obj.times
    data = ts_obj.data
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]

    # Apply trange
    if trange is None:
        t_start, t_end = times[0], times[-1]
    else:
        t_start, t_end = trange
    mask = (times >= t_start) & (times <= t_end)
    times, data = times[mask], data[mask]

    # Time zero adjustment
    tzero_adj = 0 if tzero is None else tzero
    times = times - tzero_adj

    # Axis prep
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Amplitude scaling
    value_order = int(np.floor(np.log10(np.max(np.abs(data)))))
    scale_factor = 10**value_order

    # Plot
    ax.plot(times, data, lw=lw, color=color, label=label, **kwargs)

    # Axis labels
    ax.set_xlabel(f"Time {'- $t_0$ ' if tzero else ''}(s)")
    ax.set_xlim(times[0], times[-1])
    ax.xaxis.set_major_locator(MaxNLocator(xticks))

    # Y ticks
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / scale_factor:.1f}"))
    ax.set_ylabel(f"$h~(10^{{{value_order}}})$")

    if title:
        ax.set_title(title)

    if isinstance(ylim, str) and ylim == "pm":
        limit = np.max(np.abs(data))
        ax.set_ylim(-limit, limit)
    elif isinstance(ylim, (tuple, list)) and len(ylim) == 2:
        ax.set_ylim(ylim)

    ax.grid(True, which="both", ls="--", alpha=0.3)
    return ax


def plot_freq(
    fs_obj,
    frange=None,
    logf=False,
    logy=False,
    title=None,
    ylabel=None,
    lw=0.5,
    ax=None,
    figsize=(8, 4),
    **kwargs,
):
    """
    Plot frequency series data showing real and imaginary components.

    This is a general plotting function that displays the frequency series
    content as-is. Works for any fs object (Fourier coefficients, PSD, etc.).

    Args:
        fs_obj (fs): Frequency series object to plot.
        frange (tuple, optional): Frequency range (f_min, f_max) in Hz to display.
        logf (bool): Use logarithmic frequency axis. Default: False.
        logy (bool): Use logarithmic amplitude axis. Default: False.
        title (str, optional): Plot title.
        lw (float): Line width. Default: 0.5.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Creates new if None.
        figsize (tuple): Figure size if ax is None. Default: (8, 4).
        **kwargs: Additional matplotlib plot kwargs.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.

    Examples:
        >>> # Plot Fourier transform
        >>> fs_obj = ts_obj.to_fs()
        >>> ax = plot_freq(fs_obj)

        >>> # Plot with frequency range
        >>> ax = plot_freq(fs_obj, frange=(10, 500))

        >>> # Log-log plot
        >>> ax = plot_freq(fs_obj, logf=True, logy=True)

        >>> # Plot PSD (after computing it)
        >>> psd_fs = psd(ts_obj)
        >>> ax = plot_freq(psd_fs, logf=True, logy=True)
    """
    # Get frequency axis and data
    freqs = fs_obj.freqs()
    data = fs_obj

    # Apply frequency range mask if specified
    if frange is not None:
        mask = (freqs >= frange[0]) & (freqs <= frange[1])
        freqs = freqs[mask]
        data = data[mask]

    # Check if data is complex or real based on dtype
    is_complex = np.iscomplexobj(data)

    # Extract real and imaginary parts
    real_part = np.real(data)
    imag_part = np.imag(data)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot based on whether data is real or complex
    if is_complex:
        # Data is complex - plot both parts with legend
        ax.plot(freqs, real_part, lw=lw, label="Real", **kwargs)
        ax.plot(freqs, imag_part, lw=lw, label="Imaginary", **kwargs)
        ax.legend()
    else:
        # Data is real - plot only real part without legend
        ax.plot(freqs, real_part, lw=lw, **kwargs)

    # Set logarithmic axes if requested
    if logf:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    # Labels
    ax.set_xlabel("Frequency (Hz)")

    if ylabel is None:
        # Check if this is a PSD/ASD and set appropriate y-label
        if hasattr(fs_obj, "is_psd") and fs_obj.is_psd:
            ax.set_ylabel("PSD (1/Hz)")
        elif hasattr(fs_obj, "is_asd") and fs_obj.is_asd:
            ax.set_ylabel(r"ASD (1/$\sqrt{Hz}$)")
        else:
            ax.set_ylabel("Amplitude")
    else:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    # Grid
    ax.grid(True, which="both", ls="--", alpha=0.3)

    return ax


# Plot Q-transform spectrogram
def plot_spectro(
    ts_obj,
    tzero=0,
    trange=None,
    frange=(30, 512),
    qrange=(40, 1),
    crange=None,
    tres=1000,
    fres=1000,
    logf=True,
    title=None,
    show_xlabel=False,
    show_ylabel=True,
    grid="none",
    cmap="viridis",
    transform=None,
    show_osci_xlabel=True,
    show_osci_ylabel=True,
    stack=True,
    figsize=(8, 5),
    figsize_spec=(8, 3),
    figsize_osci=(8, 2),
    show_colorbar=True,
    label_colorbar=None,
    xticks=7,
):
    """
    Plot Q-transform spectrogram and oscillogram with optional inset colorbar.

    Args:
        ts_obj (ts): Time series object.
        tzero (float): Time to align time axis (in seconds).
        trange (tuple): Time range to show (absolute GPS).
        frange (tuple): Frequency range (Hz).
        qrange (tuple): Q factor range.
        crange (tuple or None): Color range to clip power.
        tres (int): Time resolution for Q-transform.
        fres (int): Frequency resolution for Q-transform.
        logf (bool): Whether to use logarithmic frequency spacing.
        title (str): Title of the spectrogram.
        show_xlabel (bool): Show x-axis label on spectrogram.
        show_ylabel (bool): Show y-axis label on spectrogram.
        grid (str): Grid setting: 'none', 'x', 'y', 'xy'.
        cmap (str): Colormap.
        transform (function or None): Optional transform to apply to power.
            Common: np.log10, np.sqrt, np.log, lambda x: x**2, etc.
        show_osci_xlabel (bool): Show x-axis label on oscillogram.
        show_osci_ylabel (bool): Show y-axis label on oscillogram.
        stack (bool): Whether to stack plots vertically.
        figsize (tuple): Figure size for stacked mode. Default: (8, 5).
        figsize_spec (tuple): Figure size for spectrogram when stack=False. Default: (8, 3).
        figsize_osci (tuple): Figure size for oscillogram when stack=False. Default: (8, 2).
        show_colorbar (bool): Whether to show colorbar. Default: True.
        loc_colorbar (str): Deprecated - colorbar is always on the right.
        dir_colorbar (str): Deprecated - colorbar is always vertical.
        label_colorbar (str or None): Colorbar label. If None, auto-detects:
            - No transform: "Normalized energy"
            - np.log10: r"$\\log_{10}$(Normalized energy)"
            - np.sqrt: r"$\\sqrt{\\text{Normalized energy}}$"
            - np.log: r"$\\ln$(Normalized energy)"
            Default: None (auto-detect).
        xticks (int): Number of x-axis ticks. Default: 7.

    Returns:
        matplotlib.figure.Figure or tuple of Figures

    Notes:
        - Colorbar is attached to the right side of the spectrogram
        - Default label 'Normalized energy' follows GW community standard
        - Label auto-detection recognizes common transform functions
        - Y-axis ticks are automatically formatted for readability (log or linear scale)
        - When stack=False, use figsize_spec and figsize_osci for better proportions

    References:
        - PyCBC Q-transform: https://pycbc.org/pycbc/latest/html/_modules/pycbc/filter/qtransform.html
    """

    if not PYCBC_AVAILABLE:
        raise ImportError(
            "PyCBC is required for this function. " "Install with: pip install pycbc"
        )

    # Time crop
    if trange is None:
        trange = ts_obj.trange
    ts_crop = ts_obj.window(*trange)

    # Q-transform
    times, freqs, power = TimeSeries.qtransform(
        # qres = qtransform(
        # ts_crop,
        ts_crop.to_pycbc(),
        delta_t=1 / tres,
        delta_f=None if logf else 1 / fres,
        logfsteps=fres if logf else None,
        frange=frange,
        qrange=qrange,
        mismatch=0.2,
        return_complex=False,
    )
    times = times - tzero
    # times = qres["times"] - tzero
    # freqs = qres["freqs"]
    # power = qres["q_plane"].T

    if transform:
        power = transform(power)

    norm = None
    if crange:
        power = np.clip(power, crange[0], crange[1])
        norm = Normalize(vmin=crange[0], vmax=crange[1])

    # Stack or separate
    if stack:
        fig, (ax_spec, ax_osci) = plt.subplots(
            2,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [0.7, 0.3], "hspace": 0.05},
            sharex=True,
        )
    else:
        fig_spec, ax_spec = plt.subplots(figsize=figsize_spec)
        fig_osci, ax_osci = plt.subplots(figsize=figsize_osci)

    # Spectrogram
    im = ax_spec.pcolormesh(times, freqs, power, shading="auto", cmap=cmap, norm=norm)

    # Set up y-axis scale and ticks
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    if logf:
        ax_spec.set_yscale("log")
        # Manually set tick positions based on the frequency range
        # Calculate nice tick values
        fmin, fmax = frange
        # Generate tick positions: powers of 10 and their multiples
        ticks = []
        for exp in range(
            int(np.floor(np.log10(fmin))), int(np.ceil(np.log10(fmax))) + 1
        ):
            base = 10**exp
            for mult in [1, 2, 3, 5, 7]:
                val = base * mult
                if fmin <= val <= fmax:
                    ticks.append(val)

        ax_spec.set_yticks(ticks)

        # Format labels as clean integers
        def freq_formatter(x, pos):
            if x >= 1000:
                return f"{int(x/1000)}k"
            else:
                return f"{int(x)}"

        ax_spec.yaxis.set_major_formatter(FuncFormatter(freq_formatter))
    else:
        # For linear scale, use MaxNLocator for pretty breaks
        ax_spec.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

    # X-label: show automatically when stack=False, otherwise respect show_xlabel
    if not stack:
        ax_spec.set_xlabel("Time (s)")
    elif show_xlabel:
        ax_spec.set_xlabel("Time (s)")
    else:
        ax_spec.tick_params(axis="x", labelbottom=False)

    if show_ylabel:
        ax_spec.set_ylabel("Frequency (Hz)")
    else:
        ax_spec.tick_params(axis="y", labelleft=False)

    # X-axis tick count
    ax_spec.xaxis.set_major_locator(MaxNLocator(xticks))

    if grid in ["x", "xy"]:
        ax_spec.xaxis.grid(True, linestyle="--", alpha=0.3)
    if grid in ["y", "xy"]:
        ax_spec.yaxis.grid(True, linestyle="--", alpha=0.3)

    if title:
        ax_spec.set_title(title)

    # Colorbar attached to right side
    if show_colorbar:
        # Auto-detect label if not provided
        if label_colorbar is None:
            if transform is None:
                label_colorbar = "Normalized energy"
            elif transform == np.log10 or (
                hasattr(transform, "__name__") and transform.__name__ == "log10"
            ):
                label_colorbar = r"$\log_{10}$(Normalized energy)"
            elif transform == np.sqrt or (
                hasattr(transform, "__name__") and transform.__name__ == "sqrt"
            ):
                label_colorbar = r"$\sqrt{\text{Normalized energy}}$"
            elif transform == np.log or (
                hasattr(transform, "__name__") and transform.__name__ == "log"
            ):
                label_colorbar = r"$\ln$(Normalized energy)"
            else:
                # Generic fallback
                label_colorbar = "Normalized energy"

        # Use make_axes_locatable to create colorbar with proper alignment
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider_spec = make_axes_locatable(ax_spec)
        cax = divider_spec.append_axes("right", size="3%", pad=0.0)
        cbar = plt.colorbar(im, cax=cax)

        # Configure colorbar label - match y-axis label font size
        cbar.set_label(label_colorbar, rotation=90, labelpad=10, fontsize=12)

        # Adjust tick label size
        cbar.ax.tick_params(labelsize=10)

        # Also adjust oscillogram width to match spectrogram width
        # Create dummy axes on the right of oscillogram to match colorbar space
        divider_osci = make_axes_locatable(ax_osci)
        dummy_ax = divider_osci.append_axes("right", size="3%", pad=0.0)
        dummy_ax.axis("off")  # Hide the dummy axes

    # Oscillogram
    ax = plot_oscillo(ts_crop, tzero=tzero, trange=trange, ax=ax_osci, lw=0.5, xticks=xticks)

    if not show_osci_xlabel:
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False)

    if not show_osci_ylabel:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)

    if stack:
        return fig
    else:
        return fig_spec, fig_osci


def plot_anomaly(
    anom_df: pl.DataFrame,
    tzero: float | None = None,
    val_col: str = "observed",
    time_col: str = "time",
    p_crit: float = 0.05,
    p_col: str | None = "P0",
    title: str = "Anomaly Plot",
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple[float, float] = (8, 3),
    lw: float = 0.5,
    ax=None,
) -> None:
    """
    Plot anomaly results with oscillogram-style formatting (polars-only).

    Rules (no implicit column renaming/modification):
    - If 'cluster' exists and p_col is provided, color anomalies by cluster only when (p_col < p_crit).
      Otherwise anomalies are gray.
    - If p_col is None or missing, all anomaly points are gray.
    - Optional error ribbon is drawn if columns f"{val_col}_l1" and f"{val_col}_l2" exist.
    """

    if time_col not in anom_df.columns or val_col not in anom_df.columns:
        raise KeyError(f"Required columns missing: '{time_col}' and/or '{val_col}'.")

    if tzero is None:
        tzero = anom_df.select(pl.col(time_col).first()).item()

    time_shifted = (
        (anom_df.select(pl.col(time_col) - pl.lit(tzero))).to_series().to_numpy()
    )
    y = anom_df.get_column(val_col).to_numpy()

    err_lwr = f"{val_col}_l1"
    err_upr = f"{val_col}_l2"
    has_ribbon = (err_lwr in anom_df.columns) and (err_upr in anom_df.columns)
    if has_ribbon:
        y_lwr = anom_df.get_column(err_lwr).to_numpy()
        y_upr = anom_df.get_column(err_upr).to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        is_standalone = True
    else:
        is_standalone = False

    if has_ribbon:
        ax.fill_between(
            time_shifted, y_lwr, y_upr, color="gray", alpha=0.5, label="IQR range"
        )

    ax.plot(time_shifted, y, color="black", lw=lw, label="Observed")

    if "anomaly" in anom_df.columns:
        anoms = anom_df.filter(pl.col("anomaly") == 1)

        if anoms.height > 0:
            use_signif = (
                (p_col is not None)
                and (p_col in anoms.columns)
                and (p_crit is not None)
            )

            if "cluster" in anoms.columns:
                if use_signif:
                    label_series = anoms.select(
                        pl.when(
                            (pl.col(p_col) < pl.lit(p_crit))
                            & (~pl.col("cluster").is_null())
                        )
                        .then(pl.format("cluster_{}", pl.col("cluster").cast(pl.Int64)))
                        .otherwise(pl.lit("gray"))
                        .alias("label")
                    )["label"]
                    labels = label_series.unique().to_list()

                    non_gray = [lab for lab in labels if lab != "gray"]
                    palette = sns.color_palette("tab10", len(non_gray))
                    color_map = {lab: palette[i] for i, lab in enumerate(non_gray)}
                    color_map["gray"] = "gray"

                    colors = [color_map[l] for l in label_series.to_list()]
                else:
                    colors = "gray"
            else:
                colors = "red" if use_signif else "gray"

            tx = (anoms.select(pl.col(time_col) - pl.lit(tzero))).to_series().to_numpy()
            ty = anoms.get_column(val_col).to_numpy()

            ax.scatter(tx, ty, color=colors, s=20, alpha=0.35)
            ax.scatter(
                tx,
                ty,
                facecolors="none",
                edgecolors=colors,
                s=60,
                alpha=0.35,
                linewidths=1,
            )

    if time_shifted.size > 0:
        ax.set_xlim(time_shifted[0], time_shifted[-1])
    ax.xaxis.set_major_locator(MaxNLocator(10))

    ymax = np.max(np.abs(y)) if y.size > 0 else 0.0
    value_order = int(np.floor(np.log10(ymax))) if ymax > 0 else 0
    scale_factor = (10**value_order) if value_order != 0 else 1

    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / scale_factor:.1f}"))

    if xlabel is None:
        xlabel = r"Time$ - t_0$ (s)"
    if ylabel is None:
        ylabel = f"$h~(10^{{{value_order}}})$" if value_order != 0 else r"$h$"
    else:
        ylabel = f"{ylabel} $(10^{{{value_order}}})$" if value_order != 0 else r"$h$"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    if is_standalone:
        plt.show()


def plot_lambda(
    summary: pl.DataFrame,
    lambda_type: Literal["a", "c"] = "a",
    use_batch: Literal["raw", "upd"] = "upd",
    figsize: tuple = (8, 4),
) -> None:
    """
    Plot the update history of lambda_a or lambda_c from summary DataFrame.

    Args:
        summary: DataFrame with columns: batch_id, detector, lambda_a, lambda_c,
                 lambda_a_upd, lambda_c_upd. Can be from:
                 - stream() return: result["summary"]
                 - checkpoint mode: pl.read_parquet("checkpoint/summary.parquet")
        lambda_type: Either "a" for λ_a or "c" for λ_c.
        use_batch: "raw" for per-batch lambda, "upd" for cumulative updated lambda.
        figsize: Figure size.
    """
    if lambda_type not in {"a", "c"}:
        raise ValueError("lambda_type must be either 'a' or 'c'")
    if use_batch not in {"raw", "upd"}:
        raise ValueError("use_batch must be either 'raw' or 'upd'")

    # Select column
    col = f"lambda_{lambda_type}" if use_batch == "raw" else f"lambda_{lambda_type}_upd"

    if col not in summary.columns:
        raise ValueError(f"Column '{col}' not found in summary DataFrame")

    y_label = rf"$\lambda_{lambda_type}$"
    suffix = "(per batch)" if use_batch == "raw" else "(cumulative)"
    title = rf"Update history of $\lambda_{lambda_type}$ {suffix}"

    fig, ax = plt.subplots(figsize=figsize)

    for det in summary["detector"].unique().sort():
        sub = summary.filter(pl.col("detector") == det).sort("batch_id")
        ax.plot(
            sub["batch_id"].to_numpy(),
            sub[col].to_numpy(),
            marker="o",
            markersize=3,
            label=det,
        )

    ax.set_xlabel("Batch")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(title="Detector", loc="upper right")
    plt.tight_layout()
    plt.show()
    return fig


def plot_coinc(
    df: pl.DataFrame,
    *,
    tzero: Optional[float] = None,
    p_crit: float = 0.05,
    a: float = 1,
    alpha_det: float = 0.3,
    annotate_vals: bool = False,
    annotate_thresh: Optional[float] = None,
    time_col: str = "time_bin",
    prob_cols: Sequence[str] = ("P0_net", "P0_H1_bin", "P0_L1_bin"),
    utc2gps: Optional[Callable[[object], float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot coincidence significance values over time.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame returned from coincide_P0(). Must include `time_col` and `prob_cols`.
        Existing column names are used as-is (no renaming inside).
    tzero : float, optional
        GPS time used to align the x-axis (plots `GPS(time_col) - tzero`).
        If None, uses the first timestamp (converted by `utc2gps` if provided).
    p_crit : float
        Critical p-value to draw a horizontal significance threshold line.
    a : float
        Scaling factor for Significance.
    alpha_det : float
        Transparency for single-detector series.
    annotate_vals : bool
        If True, annotate points exceeding the annotation threshold.
    annotate_thresh : float, optional
        p-value threshold for annotations; defaults to `p_crit`.
    time_col : str
        Name of the time column (kept unchanged).
    prob_cols : Sequence[str]
        Names of probability columns (kept unchanged).
    utc2gps : callable, optional
        Converter from the values in `time_col` to GPS seconds (float).
        If None, `time_col` must already be numeric GPS seconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new one is created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the coincidence significance plot.
    """
    from .Pipe import Significance

    # --- Validate required columns (no renaming) ---
    needed = [time_col, *prob_cols]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if annotate_thresh is None:
        annotate_thresh = p_crit

    # --- Melt to long format (adds 'variable' and 'value'; original columns intact) ---
    melted = (
        df.select([pl.col(time_col), *[pl.col(c) for c in prob_cols]]).melt(
            id_vars=[time_col],
            value_vars=list(prob_cols),
            variable_name="variable",
            value_name="value",
        )
        # Replace only NaN with 1.0 (to avoid +inf in log10); keep Nulls as Nulls
        .with_columns(
            pl.when(pl.col("value").is_nan())
            .then(1.0)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    )

    # --- Prepare legend/appearance maps (labels only; no column renaming) ---
    label_map: Mapping[str, str] = {
        "P0_net": "net",
        "P0_H1_bin": "H1",
        "P0_L1_bin": "L1",
    }
    color_map: Mapping[str, str] = {
        "P0_net": "black",
        "P0_H1_bin": "red",
        "P0_L1_bin": "blue",
    }
    alpha_map: Mapping[str, float] = {
        "P0_net": 1.0,
        "P0_H1_bin": alpha_det,
        "P0_L1_bin": alpha_det,
    }
    ls_map: Mapping[str, str] = {
        "P0_net": "-",
        "P0_H1_bin": "--",
        "P0_L1_bin": "--",
    }

    # --- Create axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    # --- Determine tzero (use first timestamp converted to GPS if needed) ---
    first_time = melted.get_column(time_col).to_list()[0] if melted.height > 0 else None
    if tzero is None:
        if first_time is None:
            raise ValueError("Empty DataFrame; cannot infer `tzero`.")
        tzero = float(utc2gps(first_time) if utc2gps else first_time)

    # --- Draw threshold line in significance space ---
    s_threshold = float(Significance(np.array([p_crit], dtype=float), a)[0])
    ax.axhline(s_threshold, linestyle="--", linewidth=1)

    # --- Plot each variable as line + points using its own time series ---
    for var in prob_cols:
        sub = melted.filter(pl.col("variable") == var)
        if sub.is_empty():
            continue

        # Convert times to GPS seconds per subset
        t_vals_raw = sub.get_column(time_col).to_list()
        if utc2gps is not None:
            x = np.array([utc2gps(v) for v in t_vals_raw], dtype=float)
        else:
            # Expect numeric GPS seconds; polars will cast Null -> np.nan in to_numpy()
            x = np.asarray(sub.get_column(time_col).to_numpy(), dtype=float)

        x_rel = x - tzero

        # Probability values (float array; Null -> np.nan)
        p_vals = sub.get_column("value").to_numpy()
        # Significance (NaN preserved; zeros will yield +inf as in the R definition)
        s_vals = Significance(p_vals, a)

        # Line and points
        ax.plot(
            x_rel,
            s_vals,
            linestyle=ls_map.get(var, "-"),
            color=color_map.get(var, "gray"),
            alpha=alpha_map.get(var, 1.0),
            label=label_map.get(var, var),
        )
        ax.scatter(
            x_rel,
            s_vals,
            s=12,
            color=color_map.get(var, "gray"),
            alpha=min(1.0, alpha_map.get(var, 1.0) * 0.85),
            linewidths=0,
        )

        # Optional annotations for values exceeding annotation threshold
        if annotate_vals:
            s_annot = float(
                Significance(np.array([annotate_thresh], dtype=float), a)[0]
            )
            mask = np.isfinite(s_vals) & (s_vals > s_annot)
            for xi, yi in zip(x_rel[mask], s_vals[mask]):
                # Minimal offset to avoid overlapping the marker
                ax.annotate(
                    f"{yi:.2f}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                    color=color_map.get(var, "gray"),
                    alpha=alpha_map.get(var, 1.0),
                )

    # --- Labels, legend, grid ---
    ax.set_xlabel(f"Time (s) from {tzero}")
    ax.set_ylabel(r"$\mathcal{S}$")
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    return ax


# Print only if verbose is True
def message_verb(message, verb):
    if verb:
        print(message)


def summary(x):
    """
    R style summary function, but vertically aligned.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    nan_count = np.isnan(x).sum()
    valid = x[~np.isnan(x)]

    if valid.size == 0:
        raise ValueError("All values are NaN.")

    # Collect statistics
    stats = {
        "Min.": np.min(valid),
        "1st Qu.": np.percentile(valid, 25),
        "Median": np.median(valid),
        "Mean": np.mean(valid),
        "Std.": np.std(valid, ddof=1),
        "3rd Qu.": np.percentile(valid, 75),
        "Max.": np.max(valid),
        "NA's": nan_count,
    }

    # Maximum label width
    label_width = max(len(k) for k in stats.keys())

    # Formatting values
    formatted_values = {}
    for k, v in stats.items():
        if k == "NA's":
            value_str = f"{int(v):d}"
        else:
            value_str = f"{v:+.3e}"
        formatted_values[k] = value_str

    # Maximum value width
    value_width = max(len(s) for s in formatted_values.values())

    # Printing
    for k in stats.keys():
        print(f"{k:>{label_width}} {formatted_values[k]:>{value_width}}")


# ──────────────────────────────────────────────────────────────────────────────
# AR veto diagnostics & classification plots
# ──────────────────────────────────────────────────────────────────────────────
from scipy.stats import chi2, beta as beta_dist, multivariate_normal, gaussian_kde


def plot_coinc_clust(coinc_res, tzero=None, dt=None, figsize=(8, 5), ax=None):
    """Plot coincidence clusters with colored spans and markers.

    Args:
        coinc_res: DataFrame with 'time_bin', 'S', 'coincl_id' columns.
        tzero: reference GPS time (subtracted from time axis).
        dt: ignored (computed from data).
        figsize: figure size if standalone.
        ax: optional axes to plot on.
    """
    if tzero is None:
        tzero = 0

    dt = coinc_res["time_bin"][1] - coinc_res["time_bin"][0]
    half_dt = dt / 2

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        is_standalone = True
    else:
        is_standalone = False

    plot_coinc(coinc_res, tzero=tzero, a=1, ax=ax)

    unique_clusters = (
        coinc_res.filter(pl.col("coincl_id").is_not_null())["coincl_id"].unique().sort()
    )
    cmap_ = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "*", "p", "v", "<", ">", "h"]

    for i, c_id in enumerate(unique_clusters):
        subset = coinc_res.filter(pl.col("coincl_id") == c_id)

        t_min = (subset["time_bin"].min() - tzero) - half_dt
        t_max = (subset["time_bin"].max() - tzero) + half_dt

        cluster_color = cmap_(i % 10)
        cluster_marker = markers[i % len(markers)]

        ax.axvspan(t_min, t_max, color=cluster_color, alpha=0.2, zorder=1)
        ax.scatter(
            subset["time_bin"] - tzero,
            subset["S"],
            color=cluster_color,
            marker=cluster_marker,
            s=40,
            label=f"Cluster {c_id}",
            zorder=5,
        )
    if is_standalone:
        plt.show()


def plot_diag(res_net, coinc):
    """3x1 (time series) + 2x1 (statistics) diagnostic dashboard.

    Args:
        res_net: BEACON pipeline result Rist (H1/L1).
        coinc: coincidence DataFrame for the current batch.
    """
    from beacon.Pipe import cluster_coinc

    plot_data = pl.DataFrame(
        [
            {
                "index": i,
                "Detector": det,
                "lambda_c": it["stats"]["lambda_c"],
                "lambda_a": it["stats"]["lambda_a"],
            }
            for det in res_net.names
            for i, it in enumerate(res_net[det]["ustat"])
        ]
    )

    t0 = min(
        res_net["H1"]["proc"]["time"][0],
        res_net["L1"]["proc"]["time"][0],
    )

    plt.rcdefaults()
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)

    ax_l1 = fig.add_subplot(gs[0:2, 0:2])
    ax_l2 = fig.add_subplot(gs[2:4, 0:2], sharex=ax_l1)
    ax_l3 = fig.add_subplot(gs[4:6, 0:2], sharex=ax_l1)

    ax_r1 = fig.add_subplot(gs[0:3, 2])
    ax_r2 = fig.add_subplot(gs[3:6, 2], sharex=ax_r1)

    plot_anomaly(
        res_net["H1"]["proc"],
        tzero=t0, ax=ax_l1, title=None, xlabel="",
        ylabel=r"$h_{\rm H1}$",
    )
    plot_anomaly(
        res_net["L1"]["proc"],
        tzero=t0, ax=ax_l2, title=None, xlabel="",
        ylabel=r"$h_{\rm L1}$",
    )

    coinc_clustered = cluster_coinc(coinc, eps=480 / 4096, min_samples=1)
    plot_coinc_clust(coinc_clustered, tzero=t0, ax=ax_l3)
    ax_l3.set_xlabel(f"Time (s) from {t0}")

    detectors = plot_data["Detector"].unique().to_list()
    det_colors = {"H1": "red", "L1": "blue"}

    for det in detectors:
        subset = plot_data.filter(pl.col("Detector") == det)
        ax_r1.plot(
            subset["index"] - len(subset["index"]) + 1,
            subset["lambda_c"],
            marker="o", color=det_colors.get(det), label=det, alpha=0.5,
        )
        ax_r2.plot(
            subset["index"] - len(subset["index"]) + 1,
            subset["lambda_a"],
            marker="o", color=det_colors.get(det), label=det, alpha=0.5,
        )

    ax_r1.set_ylabel(r"$\lambda_c$")
    ax_r2.set_ylabel(r"$\lambda_a$")
    ax_r2.set_xlabel("Relative batch index")
    ax_r2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_r2.legend(loc="best", frameon=True)

    plt.setp(ax_l1.get_xticklabels(), visible=False)
    plt.setp(ax_l2.get_xticklabels(), visible=False)
    plt.setp(ax_r1.get_xticklabels(), visible=False)

    for ax in [ax_l1, ax_l2, ax_l3, ax_r1, ax_r2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_dist_chi2(d2, df_mle, ifo_label, alpha=0.05, ax=None):
    """Histogram of d^2 with chi^2(df_mle) PDF overlay and threshold line.

    Args:
        d2: d^2 values (normal BKG subset).
        df_mle: MLE degrees of freedom.
        ifo_label: detector label string ('H1' or 'L1').
        alpha: significance level for threshold.
        ax: optional axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    bins = np.logspace(np.log10(max(d2.min(), 1)), np.log10(d2.max()), 60)
    ax.hist(d2, bins=bins, density=True, alpha=0.4, color="tab:blue",
            label=f"Normal BKG (n={len(d2)})")

    x_grid = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 300)
    ax.plot(x_grid, chi2.pdf(x_grid, df_mle), "k--", lw=1.5,
            label=rf"$\chi^2$(df={df_mle:.1f})")

    tau = chi2.ppf(1 - alpha, df_mle)
    ax.axvline(tau, color="red", ls=":", lw=1.5,
               label=rf"$\alpha$={alpha} ($\tau$={tau:.1f})")

    ax.set(xscale="log", xlabel=rf"$d^2_{{\rm {ifo_label}}}$", ylabel="density")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def plot_dist_beta(C, alpha_beta, fap=0.053, ax=None):
    """Histogram of C with Beta PDF overlay and threshold line.

    Args:
        C: signed cosine coherence values.
        alpha_beta: Beta distribution shape parameter.
        fap: false alarm probability for threshold.
        ax: optional axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(C, bins=60, density=True, alpha=0.4, color="tab:blue",
            label=f"BKG (n={len(C)})")

    y = np.linspace(-1, 1, 300)
    pdf_vals = beta_dist.pdf((y + 1) / 2, alpha_beta, alpha_beta) / 2
    ax.plot(y, pdf_vals, "k--", lw=1.5,
            label=rf"Beta({alpha_beta:.1f}, {alpha_beta:.1f})")

    tau_C = 2 * beta_dist.ppf(1 - fap, alpha_beta, alpha_beta) - 1
    ax.axvline(tau_C, color="red", ls=":", lw=1.5,
               label=rf"FAP={fap:.3f} ($\tau_C$={tau_C:.3f})")

    ax.set(xlabel="C", ylabel="density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_3d(new_det, bkg_fts, bkg_ref, fap_c=0.053, alpha_d=0.05):
    """Interactive 3D feature space visualization with BKG sigma-contours.

    Requires plotly and scikit-image.

    Args:
        new_det: dict with 'd2H', 'd2L', 'C' arrays + 'labels' for new triggers.
        bkg_fts: dict with 'dH', 'dL', 'uH', 'uL', 'mask_normal'.
        bkg_ref: dict with 'df_mle_H', 'df_mle_L', 'alpha_beta'.
        fap_c: FAP for C threshold.
        alpha_d: significance level for d^2 threshold.
    """
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    d2H_new = new_det["d2H"]
    d2L_new = new_det["d2L"]
    C_new = new_det["C"]
    labels = new_det["labels"]

    ab = bkg_ref["alpha_beta"]
    tau_C = 2 * beta_dist.ppf(1 - fap_c, ab, ab) - 1
    tau_H = chi2.ppf(1 - alpha_d, bkg_ref["df_mle_H"])
    tau_L = chi2.ppf(1 - alpha_d, bkg_ref["df_mle_L"])

    mask = bkg_fts["mask_normal"]
    d2H_norm = bkg_fts["dH"][mask] ** 2
    d2L_norm = bkg_fts["dL"][mask] ** 2
    C_norm = (bkg_fts["uH"][mask] * bkg_fts["uL"][mask]).sum(axis=1)

    log_d2H_range = [
        np.log10(d2H_norm.min()) - 0.3,
        np.log10(max(d2H_norm.max(), d2H_new.max())) + 0.3,
    ]
    log_d2L_range = [
        np.log10(d2L_norm.min()) - 0.3,
        np.log10(max(d2L_norm.max(), d2L_new.max())) + 0.3,
    ]
    C_range = [-1, 1]

    N_GRID = 60
    d2H_grid = np.logspace(*log_d2H_range, N_GRID)
    d2L_grid = np.logspace(*log_d2L_range, N_GRID)
    C_grid = np.linspace(*C_range, N_GRID)

    D2H, D2L, CG = np.meshgrid(d2H_grid, d2L_grid, C_grid, indexing="ij")

    log_pdf = (
        chi2.logpdf(D2H, bkg_ref["df_mle_H"])
        + np.log(D2H) + np.log(np.log(10))
        + chi2.logpdf(D2L, bkg_ref["df_mle_L"])
        + np.log(D2L) + np.log(np.log(10))
        + np.log(np.clip(
            beta_dist.pdf((CG + 1) / 2, ab, ab) / 2,
            1e-300, None,
        ))
    )

    log_pdf_max = np.nanmax(log_pdf)
    neg2llr = -2 * (log_pdf - log_pdf_max)

    sigma_levels = {
        r"1$\sigma$": chi2.ppf(0.6827, 3),
        r"2$\sigma$": chi2.ppf(0.9545, 3),
        r"3$\sigma$": chi2.ppf(0.9973, 3),
    }

    log_d2H_grid = np.log10(d2H_grid)
    log_d2L_grid = np.log10(d2L_grid)
    spacing = [
        (log_d2H_grid[-1] - log_d2H_grid[0]) / (N_GRID - 1),
        (log_d2L_grid[-1] - log_d2L_grid[0]) / (N_GRID - 1),
        (C_grid[-1] - C_grid[0]) / (N_GRID - 1),
    ]
    origin = [log_d2H_grid[0], log_d2L_grid[0], C_grid[0]]

    fig = go.Figure()

    sigma_colors = {
        r"1$\sigma$": "rgba(0,100,255,0.35)",
        r"2$\sigma$": "rgba(0,100,255,0.20)",
        r"3$\sigma$": "rgba(0,100,255,0.10)",
    }

    for name in [r"3$\sigma$", r"2$\sigma$", r"1$\sigma$"]:
        level = sigma_levels[name]
        try:
            verts, faces, _, _ = marching_cubes(neg2llr, level, spacing=spacing)
            verts[:, 0] += origin[0]
            verts[:, 1] += origin[1]
            verts[:, 2] += origin[2]
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=sigma_colors[name], opacity=0.3,
                name=f"BKG {name}",
            ))
        except Exception as e:
            print(f"{name} surface failed: {e}")

    fig.add_trace(go.Scatter3d(
        x=np.log10(d2H_norm), y=np.log10(d2L_norm), z=C_norm,
        mode="markers",
        marker=dict(size=2.5, color="steelblue", opacity=0.4),
        name=f"Normal BKG (n={len(d2H_norm)})",
    ))

    marker_cfg = {
        "GW": ("limegreen", "diamond"),
        "GLC": ("darkorange", "cross"),
        "BKG": ("gray", "circle"),
    }
    for lab in ["BKG", "GLC", "GW"]:
        m = labels == lab
        if not m.any():
            continue
        col, sym = marker_cfg[lab]
        fig.add_trace(go.Scatter3d(
            x=np.log10(d2H_new[m]), y=np.log10(d2L_new[m]), z=C_new[m],
            mode="markers+text",
            marker=dict(size=7, color=col, symbol=sym,
                        line=dict(width=1, color="black")),
            text=[f"#{i}" for i in np.where(m)[0]],
            textposition="top center", textfont=dict(size=9),
            name=f"new {lab} (n={m.sum()})",
        ))

    N_SURF = 30
    gx = np.linspace(*log_d2H_range, N_SURF)
    gy = np.linspace(*log_d2L_range, N_SURF)
    GX, GY = np.meshgrid(gx, gy)
    fig.add_trace(go.Surface(
        x=GX, y=GY, z=np.full_like(GX, tau_C),
        colorscale=[[0, "rgba(255,215,0,0.25)"],
                     [1, "rgba(255,215,0,0.25)"]],
        showscale=False, name=f"C = tau_C = {tau_C:.3f}",
    ))
    gy2 = np.linspace(*log_d2L_range, N_SURF)
    gz2 = np.linspace(*C_range, N_SURF)
    GY2, GZ2 = np.meshgrid(gy2, gz2)
    fig.add_trace(go.Surface(
        x=np.full_like(GY2, np.log10(tau_H)), y=GY2, z=GZ2,
        colorscale=[[0, "rgba(255,0,0,0.15)"],
                     [1, "rgba(255,0,0,0.15)"]],
        showscale=False, name=f"d2_H1 = tau_H = {tau_H:.1f}",
    ))
    gx3 = np.linspace(*log_d2H_range, N_SURF)
    gz3 = np.linspace(*C_range, N_SURF)
    GX3, GZ3 = np.meshgrid(gx3, gz3)
    fig.add_trace(go.Surface(
        x=GX3, y=np.full_like(GX3, np.log10(tau_L)), z=GZ3,
        colorscale=[[0, "rgba(0,0,255,0.15)"],
                     [1, "rgba(0,0,255,0.15)"]],
        showscale=False, name=f"d2_L1 = tau_L = {tau_L:.1f}",
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="log10(d2_H1)"),
            yaxis=dict(title="log10(d2_L1)"),
            zaxis=dict(title="C"),
            aspectmode="cube",
        ),
        title="3D Feature Space",
        width=900, height=750,
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()


# ============================================================
# BKG null-reference & classification diagnostics
# (registered 2026-06-15; schema key bkg_fts['mcd'])
# ============================================================

def plot_d2_plane(d2_H, d2_L, mcd, mask_normal, iso_dsq=None, ax=None,
               show_abnormal=True):
    n_norm, n_abn = int(mask_normal.sum()), int((~mask_normal).sum())
    normal_idx = int(np.argmin(mcd.means_.sum(axis=1)))

    pad = 0.2
    log_h = np.linspace(np.log10(d2_H.min())-pad, np.log10(d2_H.max())+pad, 200)
    log_l = np.linspace(np.log10(d2_L.min())-pad, np.log10(d2_L.max())+pad, 200)
    Gh, Gl = np.meshgrid(log_h, log_l)
    pos = np.stack([Gh, Gl], axis=-1)
    xlo, xhi = 10**log_h[0], 10**log_h[-1]
    ylo, yhi = 10**log_l[0], 10**log_l[-1]
    lo, hi = min(xlo, ylo), max(xhi, yhi)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    comp_colors = [colors[5], colors[1]]
    comp_labels = ["Normal", "Abnormal"]
    comp_order = [normal_idx, 1 - normal_idx]

    # Normal scatter and contour share colors[5] (blue); a white halo around the
    # contour lines keeps the Gaussian core distinct from the dense scatter.
    contour_color = colors[5]
    for i, ci in enumerate(comp_order):
        if i == 1:  # Abnormal contour omitted (MCD: only Normal Gaussianity matters)
            continue
        rv = multivariate_normal(mcd.means_[ci], mcd.covariances_[ci])
        pdf_vals = rv.pdf(pos) * mcd.weights_[ci]
        levels = [pdf_vals.max() * np.exp(-0.5 * n**2) for n in [3, 2, 1]]
        cs = ax.contour(10**Gh, 10**Gl, pdf_vals,
                        levels=levels, colors=contour_color, linewidths=1.5,
                        alpha=0.95, zorder=20)
        halo = [pe.withStroke(linewidth=2.0, foreground="white")]
        try:
            cs.set_path_effects(halo)            # matplotlib >= 3.8 (ContourSet is a Collection)
        except AttributeError:
            for col in cs.collections:           # older matplotlib
                col.set_path_effects(halo)

    if iso_dsq:
        for S in iso_dsq:
            d2h_line = np.linspace(0.01, S-0.01, 500)
            d2l_line = S - d2h_line
            ax.plot(d2h_line, d2l_line, color="green", ls="--", lw=1.5, alpha=0.6)
            ax.text(S, lo, rf"$\Sigma={S}$", fontsize=7, color="green",
                    rotation=-80, ha="center", va="bottom", clip_on=True)

    w_norm = mcd.weights_[normal_idx]
    w_abn = mcd.weights_[1 - normal_idx]
    ax.scatter(d2_H[mask_normal], d2_L[mask_normal],
               alpha=0.25, s=5, color=colors[5],
               label=f"Normal (w={w_norm:.2f}, n={n_norm:,})")
    if show_abnormal:
        ax.scatter(d2_H[~mask_normal], d2_L[~mask_normal],
                   alpha=0.25, s=5, color=colors[1],
                   label=f"Abnormal (w={w_abn:.2f}, n={n_abn:,})")
    else:
        ax.scatter(d2_H[~mask_normal], d2_L[~mask_normal],
                   alpha=0.1, s=3, color="gray")

    ax.set(xscale="log", yscale="log",
           xlabel=r"$d_{\rm H1}^2$", ylabel=r"$d_{\rm L1}^2$",
           xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    if standalone:
        plt.tight_layout(); plt.show()


def plot_null_ref(bkg_fts, bkg_ref, figsize=(10, 4.5), save_path=None):
    mask = bkg_fts["mask_normal"]
    dH_all = bkg_fts["dH"]; dL_all = bkg_fts["dL"]
    d2H_all = dH_all ** 2; d2L_all = dL_all ** 2
    d2H_norm = d2H_all[mask]; d2L_norm = d2L_all[mask]
    C = (bkg_fts["uH"] * bkg_fts["uL"]).sum(axis=1)

    ab = bkg_ref["alpha_beta"]
    df_H = bkg_ref["df_mle_H"]
    df_L = bkg_ref["df_mle_L"]

    n_total = len(dH_all)
    n_norm = int(mask.sum())
    n_abn = n_total - n_norm
    print(f"Null Reference: {n_total} triggers")
    print(f"  MCD: Normal={n_norm}, Abnormal={n_abn}")
    print(f"  chi2 MLE: H1 df={df_H:.2f}, L1 df={df_L:.2f}")
    print(f"  Beta: a=b={ab:.1f}")

    fig = plt.figure(figsize=figsize)
    gs_outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.2)
    ax_C = fig.add_subplot(gs_outer[0])
    gs_corner = gs_outer[1].subgridspec(
        2, 2, wspace=0.04, hspace=0.04,
        width_ratios=[3, 1], height_ratios=[1, 3],
    )
    ax_dH = fig.add_subplot(gs_corner[0, 0])
    ax_2d = fig.add_subplot(gs_corner[1, 0])
    ax_dL = fig.add_subplot(gs_corner[1, 1])

    # ── ax_2d: 2D MCD ──
    plot_d2_plane(d2H_all, d2L_all, bkg_fts["mcd"], mask, ax=ax_2d)
    ax_2d.set_aspect("auto")

    # ── ax_C: outline hist + Beta PDF ──
    c_bins = np.linspace(-1, 1, 61)
    y_grid = np.linspace(-1, 1, 300)
    ax_C.plot(y_grid, beta_dist.pdf((y_grid + 1) / 2, ab, ab) / 2,
              "k--", lw=1.5, zorder=10,
              label=rf"Beta({ab:.1f}, {ab:.1f})")
    ax_C.hist(C, bins=c_bins, density=True,
              histtype="step", color=colors[5], linewidth=2.0, zorder=4,
              label=f"NOS (n={len(C):,})")
    ax_C.set(xlabel="C", ylabel="density")
    ax_C.legend(fontsize=8)
    ax_C.grid(True, alpha=0.3)

    def _draw_marginal(ax, d2, df_mle, ifo_label, orientation):
        horiz = (orientation == "horizontal")
        bins = np.logspace(np.log10(max(d2.min(), 1)),
                           np.log10(d2.max()), 60)
        x_grid = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 300)
        pdf_vals = chi2.pdf(x_grid, df_mle)
        if horiz:
            chi2_line, = ax.plot(pdf_vals, x_grid, "k--", lw=1.5, zorder=10,
                                 label=rf"$\chi^2$(df={df_mle:.1f})")
            ax.hist(d2, bins=bins, density=True,
                    histtype="step", orientation="horizontal",
                    color=colors[5], linewidth=1.5, zorder=4)
            ax.set(yscale="log", ylabel=rf"$d^2_{{\rm {ifo_label}}}$",
                   xlabel="density")
            ax.set_ylim(bins[0], bins[-1])

            # Real rotated legend: build it already vertical.
            # (AuxTransformBox rotates the line path but NOT the glyphs, so the
            #  text must carry rotation itself; line is drawn vertical.)
            tbox = AuxTransformBox(Affine2D())
            tbox.add_artist(Line2D([0, 0], [8, 24], color="k", ls="--", lw=1.5))
            tbox.add_artist(Text(0, -4, chi2_line.get_label(),
                                 rotation=270, rotation_mode="anchor",
                                 ha="left", va="center", fontsize=8))
            anchored = AnchoredOffsetbox(
                loc="upper right", child=tbox, frameon=True,
                borderpad=0.3, bbox_to_anchor=(1.0, 1.0),
                bbox_transform=ax.transAxes,
            )
            anchored.patch.set(facecolor="white", edgecolor="0.7", alpha=0.7)
            anchored.set_zorder(11)
            ax.add_artist(anchored)
        else:
            chi2_line, = ax.plot(x_grid, pdf_vals, "k--", lw=1.5, zorder=10,
                                 label=rf"$\chi^2$(df={df_mle:.1f})")
            ax.hist(d2, bins=bins, density=True,
                    histtype="step", color=colors[5], linewidth=1.5, zorder=4)
            ax.set(xscale="log", xlabel=rf"$d^2_{{\rm {ifo_label}}}$",
                   ylabel="density")
            ax.set_xlim(bins[0], bins[-1])
            ax.legend([chi2_line], [chi2_line.get_label()],
                      fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
    _draw_marginal(ax_dH, d2H_norm, df_H, "H1", "vertical")
    _draw_marginal(ax_dL, d2L_norm, df_L, "L1", "horizontal")

    xlim_2d = ax_2d.get_xlim()
    ylim_2d = ax_2d.get_ylim()
    ax_dH.sharex(ax_2d)
    ax_dL.sharey(ax_2d)
    ax_dH.set_xlim(xlim_2d)
    ax_dL.set_ylim(ylim_2d)
    plt.setp(ax_dH.get_xticklabels(), visible=False)
    plt.setp(ax_dL.get_yticklabels(), visible=False)
    ax_dH.set_xlabel("")
    ax_dL.set_ylabel("")

    fig.suptitle(f"Null Reference (n={n_norm:,})", fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def _draw_d2_panel(ax, d2_train, d2_new, labels, df_mle, ifo_label,
                   alpha_d, tau, label_colors, orientation="vertical",
                   show_legend=True):
    horiz = (orientation == "horizontal")
    all_d2 = np.concatenate([d2_train, d2_new])
    bins = np.logspace(np.log10(max(all_d2.min(), 1)),
                       np.log10(all_d2.max()), 60)
    stack_data, stack_colors, stack_labels = [], [], []
    for lab in ["BKG", "GLC"]:
        m = labels == lab
        n_lab = int(m.sum())
        if n_lab < 5:
            continue
        stack_data.append(d2_new[m])
        stack_colors.append(label_colors[lab])
        stack_labels.append(f"new {lab} (n={n_lab:,})")
    if stack_data:
        fill_cols = [(*mcolors.to_rgb(c), 0.55) for c in stack_colors]
        ax.hist(stack_data, bins=bins, density=True, stacked=True,
                histtype="stepfilled", orientation=orientation,
                color=fill_cols, edgecolor=stack_colors, linewidth=1.5,
                label=stack_labels, zorder=2)
    x_grid = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 300)
    pdf_vals = chi2.pdf(x_grid, df_mle)
    if horiz:
        ax.plot(pdf_vals, x_grid, "k--", lw=1.5, zorder=4,
                label=rf"$\chi^2$(df={df_mle:.1f})")
        ax.axhline(tau, color=colors[6], ls=":", lw=1.5, zorder=4,
                   label=rf"$\alpha$={alpha_d} ($\tau$={tau:.1f})")
        ax.hist(d2_train, bins=bins, density=True,
                histtype="step", orientation="horizontal",
                color=colors[5], linewidth=2.0, zorder=10,
                label=f"BKG train (n={len(d2_train):,})")
        ax.set(yscale="log", ylabel=rf"$d^2_{{\rm {ifo_label}}}$",
               xlabel="density")
        ax.set_ylim(bins[0], bins[-1])
    else:
        ax.plot(x_grid, pdf_vals, "k--", lw=1.5, zorder=4,
                label=rf"$\chi^2$(df={df_mle:.1f})")
        ax.axvline(tau, color=colors[6], ls=":", lw=1.5, zorder=4,
                   label=rf"$\alpha$={alpha_d} ($\tau$={tau:.1f})")
        ax.hist(d2_train, bins=bins, density=True,
                histtype="step", color=colors[5], linewidth=2.0, zorder=10,
                label=f"BKG train (n={len(d2_train):,})")
        ax.set(xscale="log", xlabel=rf"$d^2_{{\rm {ifo_label}}}$",
               ylabel="density")
        ax.set_xlim(bins[0], bins[-1])
    if show_legend == "chi2":
        loc = "upper right"# if horiz else "upper right"
        for line in ax.get_lines():
            lbl = line.get_label()
            if lbl.startswith(r"$\chi^2$"):
                ax.legend([line], [lbl], fontsize=8, loc=loc)
                break
    elif show_legend:
        ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def plot_classif_summary(classif_res, bkg_fts, bkg_ref, fap_c=0.053, alpha_d=0.05,
                 kde_grid=80, kde_max_n=10000, seed=42):
    d2H_new = classif_res['d2H'].to_numpy()
    d2L_new = classif_res['d2L'].to_numpy()
    C_new = classif_res['C'].to_numpy()
    labels = classif_res['label'].to_numpy()
    ab = bkg_ref["alpha_beta"]
    tau_H = chi2.ppf(1 - alpha_d, bkg_ref["df_mle_H"])
    tau_L = chi2.ppf(1 - alpha_d, bkg_ref["df_mle_L"])

    mask_bkg_norm = bkg_fts["mask_normal"]
    d2H_bkg = bkg_fts["dH"][mask_bkg_norm] ** 2
    d2L_bkg = bkg_fts["dL"][mask_bkg_norm] ** 2
    C_bkg = (bkg_fts["uH"] * bkg_fts["uL"]).sum(axis=1)

    label_colors = {"GW": colors[3], "GLC": colors[1], "BKG": "gray"}
    n_triggers = len(C_new)
    rng = np.random.default_rng(seed)

    def _filled_layers(rgb, alphas=(0.18, 0.32, 0.50)):
        return [(*rgb, a) for a in alphas]

    # ── GridSpec: C panel (좌) + corner block (우), 크기 비슷하게 ──
    fig = plt.figure(figsize=(10, 4.5))
    gs_outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.2)
    ax_C = fig.add_subplot(gs_outer[0])
    gs_corner = gs_outer[1].subgridspec(
        2, 2, wspace=0.04, hspace=0.04,
        width_ratios=[3, 1], height_ratios=[1, 3],
    )
    ax_dH = fig.add_subplot(gs_corner[0, 0])
    ax_2d = fig.add_subplot(gs_corner[1, 0])
    ax_dL = fig.add_subplot(gs_corner[1, 1])
    #ax_2d.set_box_aspect(1)

    # ── ax_2d: 2D combined KDE + label dominance mask ──
    ax = ax_2d
    mcd = bkg_fts["mcd"]
    normal_idx = int(np.argmin(mcd.means_.sum(axis=1)))

    all_log_h = np.log10(np.concatenate([bkg_fts["dH"] ** 2, d2H_new]))
    all_log_l = np.log10(np.concatenate([bkg_fts["dL"] ** 2, d2L_new]))
    pad = 0.1
    h_lin = np.linspace(all_log_h.min() - pad, all_log_h.max() + pad, kde_grid)
    l_lin = np.linspace(all_log_l.min() - pad, all_log_l.max() + pad, kde_grid)
    Hg, Lg = np.meshgrid(h_lin, l_lin)
    grid_pts = np.vstack([Hg.ravel(), Lg.ravel()])

    rv = multivariate_normal(mcd.means_[normal_idx], mcd.covariances_[normal_idx])
    pos = np.stack([Hg, Lg], axis=-1)
    pdf_vals = rv.pdf(pos) * mcd.weights_[normal_idx]
    normal_levels = sorted(
        pdf_vals.max() * np.exp(-0.5 * n ** 2) for n in (1, 2, 3)
    )
    ax.contour(10 ** Hg, 10 ** Lg, pdf_vals,
               levels=normal_levels, colors=colors[5],
               alpha=0.9, linewidths=1.4)
    ax.plot([], [], color=colors[5], lw=1.5,
            label=f"BKG train (n={int(mask_bkg_norm.sum()):,})")

    sigma_p = {1: 0.6827, 2: 0.9545, 3: 0.9973}
    m_bkg = labels == "BKG"
    m_glc = labels == "GLC"
    n_bkg = int(m_bkg.sum())
    n_glc = int(m_glc.sum())

    def _maybe_subsample(arr_h, arr_l, n):
        if n > kde_max_n:
            idx = rng.choice(n, kde_max_n, replace=False)
            return arr_h[idx], arr_l[idx]
        return arr_h, arr_l

    if n_bkg + n_glc >= 5:
        log_h_all = np.log10(np.concatenate([d2H_new[m_bkg], d2H_new[m_glc]]))
        log_l_all = np.log10(np.concatenate([d2L_new[m_bkg], d2L_new[m_glc]]))
        log_h_all, log_l_all = _maybe_subsample(log_h_all, log_l_all,
                                                len(log_h_all))
        kde_all = gaussian_kde(np.vstack([log_h_all, log_l_all]))
        kde_all_vals = kde_all(grid_pts).reshape(Hg.shape)
        sorted_v = np.sort(kde_all_vals.ravel())[::-1]
        cum = np.cumsum(sorted_v) / sorted_v.sum()
        sig_levels = sorted(
            sorted_v[min(int(np.searchsorted(cum, sigma_p[s])),
                         len(sorted_v) - 1)]
            for s in (1, 2, 3)
        )
        fill_levels = sig_levels + [kde_all_vals.max() * 1.01]

        if n_bkg >= 5 and n_glc >= 5:
            h_b, l_b = _maybe_subsample(np.log10(d2H_new[m_bkg]),
                                         np.log10(d2L_new[m_bkg]), n_bkg)
            h_g, l_g = _maybe_subsample(np.log10(d2H_new[m_glc]),
                                         np.log10(d2L_new[m_glc]), n_glc)
            kde_b = gaussian_kde(np.vstack([h_b, l_b]))
            kde_g = gaussian_kde(np.vstack([h_g, l_g]))
            w_b = kde_b(grid_pts) * n_bkg
            w_g = kde_g(grid_pts) * n_glc
            bkg_dominant = (w_b >= w_g).reshape(Hg.shape)
        else:
            bkg_dominant = np.full_like(kde_all_vals, n_bkg >= n_glc, dtype=bool)

        kde_bkg_part = np.where(bkg_dominant, kde_all_vals, np.nan)
        kde_glc_part = np.where(~bkg_dominant, kde_all_vals, np.nan)

        rgb_bkg = mcolors.to_rgb(label_colors["BKG"])
        rgb_glc = mcolors.to_rgb(label_colors["GLC"])
        ax.contourf(10 ** Hg, 10 ** Lg, kde_bkg_part,
                    levels=fill_levels, colors=_filled_layers(rgb_bkg))
        ax.contourf(10 ** Hg, 10 ** Lg, kde_glc_part,
                    levels=fill_levels,
                    colors=_filled_layers(rgb_glc))#, alphas=(0.08, 0.16, 0.25)))
        ax.contour(10 ** Hg, 10 ** Lg, kde_bkg_part,
                   levels=sig_levels, colors=label_colors["BKG"],
                   alpha=0.9, linewidths=1.0)
        ax.contour(10 ** Hg, 10 ** Lg, kde_glc_part,
                   levels=sig_levels, colors=label_colors["GLC"],
                   alpha=0.9, linewidths=1.0)
        ax.plot([], [], color=label_colors["BKG"], lw=4, alpha=0.5,
                label=f"new BKG (n={n_bkg:,})")
        ax.plot([], [], color=label_colors["GLC"], lw=4, alpha=0.5,
                label=f"new GLC (n={n_glc:,})")

    ax.axvline(tau_H, color=colors[6], ls=":", lw=1.5, alpha=0.85,
               label=rf"$\tau_{{\rm H1}}$={tau_H:.1f}")
    ax.axhline(tau_L, color=colors[6], ls=":", lw=1.5, alpha=0.85,
               label=rf"$\tau_{{\rm L1}}$={tau_L:.1f}")
    ax.set(xscale="log", yscale="log",
           xlabel=r"$d_{\rm H1}^2$", ylabel=r"$d_{\rm L1}^2$",
           xlim=(10 ** h_lin[0], 10 ** h_lin[-1]),
           ylim=(10 ** l_lin[0], 10 ** l_lin[-1]))
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    # ── ax_C: C 1D ──
    ax = ax_C
    tau_C = 2 * beta_dist.ppf(1 - fap_c, ab, ab) - 1
    c_bins = np.linspace(-1, 1, 61)
    stack_data, stack_colors, stack_labels = [], [], []
    for lab in ["BKG", "GLC", "GW"]:
        m = labels == lab
        n_lab = int(m.sum())
        if n_lab < 5:
            continue
        stack_data.append(C_new[m])
        stack_colors.append(label_colors[lab])
        stack_labels.append(f"new {lab} (n={n_lab:,})")
    if stack_data:
        fill_cols = [(*mcolors.to_rgb(c), 0.55) for c in stack_colors]
        ax.hist(stack_data, bins=c_bins, density=True, stacked=True,
                histtype="stepfilled",
                color=fill_cols, edgecolor=stack_colors, linewidth=1.5,
                label=stack_labels, zorder=2)
    y_grid = np.linspace(-1, 1, 300)
    ax.plot(y_grid, beta_dist.pdf((y_grid + 1) / 2, ab, ab) / 2,
            "k--", lw=1.5, zorder=4,
            label=rf"Beta({ab:.1f}, {ab:.1f})")
    ax.axvline(tau_C, color=colors[3], ls="-.", lw=1.5, zorder=4,
               label=rf"FAP={fap_c:.3f} ($\tau_C$={tau_C:.3f})")
    ax.hist(C_bkg, bins=c_bins, density=True,
            histtype="step", color=colors[5], linewidth=2.0, zorder=10,
            label=f"BKG train (n={len(C_bkg):,})")
    ax.set(xlabel="C", ylabel="density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── marginals (no legend, share axis with 2D) ──
    _draw_d2_panel(ax_dH, d2H_bkg, d2H_new, labels,
                   bkg_ref["df_mle_H"], "H1", alpha_d,
                   tau_H, label_colors, orientation="vertical",
                   show_legend="chi2")

    _draw_d2_panel(ax_dL, d2L_bkg, d2L_new, labels,
                   bkg_ref["df_mle_L"], "L1", alpha_d,
                   tau_L, label_colors, orientation="horizontal",
                   show_legend="chi2")

    xlim_2d = ax_2d.get_xlim()
    ylim_2d = ax_2d.get_ylim()
    ax_dH.sharex(ax_2d)
    ax_dL.sharey(ax_2d)
    ax_dH.set_xlim(xlim_2d)
    ax_dL.set_ylim(ylim_2d)
    plt.setp(ax_dH.get_xticklabels(), visible=False)
    plt.setp(ax_dL.get_yticklabels(), visible=False)
    ax_dH.set_xlabel("")
    ax_dL.set_ylabel("")

    fig.suptitle(f"Classification (n={n_triggers} triggers)", fontsize=13)
    fig.text(
        0.5, -0.05,
        "Note: BKG and GLC are drawn as stacked histograms — "
        "bar heights are cumulative, not independent.",
        ha="center", va="bottom", fontsize=9, style="italic", color="dimgray",
    )
    plt.tight_layout()
    plt.show()


def plot_classif_summary_3d(classif_res, label_colors,
                    grid_size=50, kde_max_n=8000, seed=42,
                    sigma_levels=(1, 2),
                    sigma_alphas=(0.55, 0.25)):
    import plotly.graph_objects as go
    from skimage import measure
    sigma_p = {1: 0.6827, 2: 0.9545, 3: 0.9973}
    rng = np.random.default_rng(seed)
    d2H = classif_res["d2H"].to_numpy()
    d2L = classif_res["d2L"].to_numpy()
    C   = classif_res["C"].to_numpy()
    labels = classif_res["label"].to_numpy()

    h_lin = np.linspace(np.log10(d2H.min()) - 0.1,
                        np.log10(d2H.max()) + 0.1, grid_size)
    l_lin = np.linspace(np.log10(d2L.min()) - 0.1,
                        np.log10(d2L.max()) + 0.1, grid_size)
    c_lin = np.linspace(-1, 1, grid_size)
    Hg, Lg, Cg = np.meshgrid(h_lin, l_lin, c_lin, indexing="ij")
    pts = np.vstack([Hg.ravel(), Lg.ravel(), Cg.ravel()])

    kdes, label_counts = {}, {}
    for lab in ["BKG", "GLC", "GW"]:
        m = labels == lab
        n_lab = int(m.sum())
        if n_lab < 20:
            continue
        h = np.log10(d2H[m]); l = np.log10(d2L[m]); c = C[m]
        if n_lab > kde_max_n:
            idx = rng.choice(n_lab, kde_max_n, replace=False)
            h, l, c = h[idx], l[idx], c[idx]
        kdes[lab] = gaussian_kde(np.vstack([h, l, c]))
        label_counts[lab] = n_lab
    label_list = list(kdes.keys())
    if not label_list:
        print("No labels with sufficient data."); return

    raw_grid = {lab: kdes[lab](pts) for lab in label_list}
    union = np.zeros_like(raw_grid[label_list[0]])
    for lab in label_list:
        norm = raw_grid[lab] / raw_grid[lab].max()
        union = np.maximum(union, norm)
    vol = union.reshape(Hg.shape)

    sv = np.sort(union)[::-1]
    cum = np.cumsum(sv) / sv.sum()
    iso_levels = [
        sv[min(int(np.searchsorted(cum, sigma_p[s])), len(sv) - 1)]
        for s in sigma_levels
    ]

    spacing = (
        (h_lin[-1] - h_lin[0]) / (grid_size - 1),
        (l_lin[-1] - l_lin[0]) / (grid_size - 1),
        (c_lin[-1] - c_lin[0]) / (grid_size - 1),
    )
    origin = np.array([h_lin[0], l_lin[0], c_lin[0]])

    fig = go.Figure()
    for j, (s, iso) in enumerate(zip(sigma_levels, iso_levels)):
        try:
            verts, faces, _, _ = measure.marching_cubes(
                vol, level=iso, spacing=spacing
            )
        except (RuntimeError, ValueError):
            continue
        verts = verts + origin

        face_centers = verts[faces].mean(axis=1)
        face_dens = np.stack([
            kdes[lab](face_centers.T) * label_counts[lab]
            for lab in label_list
        ], axis=0)
        face_lab = np.argmax(face_dens, axis=0)

        for i, lab in enumerate(label_list):
            sel = face_lab == i
            if not sel.any():
                continue
            faces_sel = faces[sel]
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces_sel[:, 0], j=faces_sel[:, 1], k=faces_sel[:, 2],
                color=label_colors[lab],
                opacity=sigma_alphas[j],
                flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.5),
                legendgroup=lab,
                name=(f"{lab} ({','.join(f'{x}σ' for x in sigma_levels)}, "
                      f"n={label_counts[lab]:,})"),
                showlegend=(j == 0),
                hoverinfo="name",
            ))

    axis_style = dict(
        backgroundcolor="white",
        showbackground=True,
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="log₁₀ d²_H1", **axis_style),
            yaxis=dict(title="log₁₀ d²_L1", **axis_style),
            zaxis=dict(title="C", **axis_style),
            aspectmode="cube",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=(f"Classification 3D distribution "
               f"({','.join(f'{x}σ' for x in sigma_levels)} surfaces, "
               f"face color = label dominance)"),
        width=820, height=750,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()


def place_trigger_labels(ax, fig, items, base_x, base_y, row_gap_frac=0.35,
                         x_margin_frac=0.02, data_gap_frac=0.03, top_limit=0.90,
                         expand=1.5, n_iter=10):
    """Label boxes: bounded rows (interval scheduling) at local-curve heights.

    Row index = interval scheduling on x (depth = max simultaneous overlap, so a
    long chain of pairwise overlaps still fits in a few rows — no runaway stack).
    Vertical position = local coinc-curve height under the box + row offset, in
    axes-fraction (stable on any y-scale). ymax grows only if the tallest box
    would exceed top_limit. Use with a symlog S axis.

    items: list of dict(x=trigger_x_rel, txt=str, color=facecolor).
    base_x, base_y: coinc net-S curve (x rel to t0, S data units).
    """
    from matplotlib.patches import BoxStyle

    class _RoundTop(BoxStyle.Round):           # 상단 패딩만 추가 (top-only pad)
        # mathtext 라인이 섞이면 블록 윗줄 ascent 예약폭이 ~0.2*fontsize 깎여
        # 텍스트가 박스 안에서 위로 쏠린다. 상단에만 그만큼 패딩을 더해 잉크를 중앙정렬.
        def __init__(self, pad=0.3, rounding_size=None, top_pad=0.0):
            super().__init__(pad=pad, rounding_size=rounding_size)
            self.top_pad = top_pad

        def __call__(self, x0, y0, width, height, mutation_size):
            return super().__call__(
                x0, y0, width, height + self.top_pad * mutation_size, mutation_size)

    box = _RoundTop(pad=0.3, top_pad=0.5)      # 실측 튜닝값 (mathtext 라벨 중앙정렬)

    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    x0, x1 = ax.get_xlim()
    xr = x1 - x0
    xat = ax.get_xaxis_transform()

    def fracY(ys, xc):                            # data-y -> axes-fraction y
        ab = ax.get_window_extent(rend)
        py = ax.transData.transform(
            np.column_stack([np.full(np.shape(ys), xc), ys]))[:, 1]
        return (py - ab.y0) / ab.height

    box_hf, hw = 0.0, []
    for it in items:
        t = ax.text(it["x"], 0.5, it["txt"], transform=xat, fontsize=7,
                    ha="center", va="center",
                    bbox=dict(pad=2, boxstyle=box))
        fig.canvas.draw()
        ext = t.get_window_extent(rend)
        (xa, _), (xb, _) = inv.transform([[ext.x0, ext.y0], [ext.x1, ext.y1]])
        hw.append((xb - xa) / 2)
        box_hf = max(box_hf, ext.height / ax.get_window_extent(rend).height)
        t.remove()

    base_x = np.asarray(base_x, float)
    base_y = np.asarray(base_y, float)
    mx = x_margin_frac * xr
    cx = [min(max(items[i]["x"], x0 + mx + hw[i]), x1 - mx - hw[i])
          for i in range(len(items))]
    L = [cx[i] - hw[i] for i in range(len(items))]
    R = [cx[i] + hw[i] for i in range(len(items))]

    # bounded rows: interval scheduling (lowest free row)
    rows_right, row_of = [], [0] * len(items)
    for i in sorted(range(len(items)), key=lambda k: L[k]):
        for r in range(len(rows_right)):
            if rows_right[r] + 0.012 * xr <= L[i]:
                row_of[i] = r
                rows_right[r] = R[i]
                break
        else:
            row_of[i] = len(rows_right)
            rows_right.append(R[i])
    row_hf = box_hf * (1 + row_gap_frac)

    # local curve height (data) under each box
    cbase = []
    for i in range(len(items)):
        m = (base_x >= L[i]) & (base_x <= R[i])
        cbase.append(float(np.nanmax(base_y[m])) if m.any() else ax.get_ylim()[0])

    def layout():
        bottoms = [float(fracY([cbase[i]], cx[i])[0]) + data_gap_frac
                   + row_of[i] * row_hf for i in range(len(items))]
        return bottoms, max(b + box_hf for b in bottoms)

    for _ in range(n_iter):                       # grow ymax only if tallest overflows
        bottoms, mtop = layout()
        if mtop <= top_limit:
            break
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * expand)
    bottoms, mtop = layout()

    for i, it in enumerate(items):
        ax.annotate(
            it["txt"], xy=(it["x"], cbase[i]), xycoords=ax.transData,
            xytext=(cx[i], bottoms[i] + box_hf / 2), textcoords=xat,
            ha="center", va="center", fontsize=8,
            bbox=dict(fc=it["color"], alpha=0.4, pad=2, boxstyle=box),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5,
                            shrinkA=0, shrinkB=0),
            zorder=10,
        )


def plot_batch_dashboard(
    seg_dir, batch_id, figsize=(12, 9), ar_seg_dur=32 * 15 / 4096, save_path=None
):
    """배치별 진단 대시보드 + 분류 오버레이 (좌측 3패널: H1 / L1 / coinc).

    저장된 parquet(proc/coinc/classif)로부터 자동 로드.
    분류 음영: 고정폭 ar_seg_dur (= 특징추출 윈도우), 색 = 라벨.
    coinc 마커: 색 = 분류 라벨, 모양 = 클러스터 구분.

    Args:
        seg_dir: beacon_results 디렉토리 (Path 또는 str).
        batch_id: 배치 번호.
        figsize: figure 크기 (가로, 세로).
        ar_seg_dur: 분류 shading 폭 (초).
    """
    seg_dir = Path(seg_dir)
    label_colors = {"GW": "limegreen", "GLC": "darkorange", "BKG": "gray"}
    cluster_markers = ["o", "s", "^", "D", "*", "p", "v", "<", ">", "h"]

    # ── 데이터 로드 ───────────────────────────────────────────────
    proc_h1 = pl.read_parquet(
        str(seg_dir / "proc" / f"batch_{batch_id:04d}_H1.parquet")
    )
    proc_l1 = pl.read_parquet(
        str(seg_dir / "proc" / f"batch_{batch_id:04d}_L1.parquet")
    )
    coinc = pl.read_parquet(str(seg_dir / "coinc" / f"batch_{batch_id:04d}.parquet"))

    classif_path = seg_dir / "classif" / f"batch_{batch_id:04d}.parquet"
    classif = pl.read_parquet(str(classif_path)) if classif_path.exists() else None

    # ── t0 기준 ──────────────────────────────────────────────────
    t0 = min(proc_h1["time"][0], proc_l1["time"][0])

    # ── Figure 구성 (좌측 3패널) ─────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, hspace=0.05)
    ax_l1 = fig.add_subplot(gs[0])
    ax_l2 = fig.add_subplot(gs[1], sharex=ax_l1)
    ax_l3 = fig.add_subplot(gs[2], sharex=ax_l1)
    left_axes = [ax_l1, ax_l2, ax_l3]

    # anomaly (strain)
    plot_anomaly(
        proc_h1, tzero=t0, ax=ax_l1, title=None, xlabel="", ylabel=r"$h_{\rm H1}$"
    )
    plot_anomaly(
        proc_l1, tzero=t0, ax=ax_l2, title=None, xlabel="", ylabel=r"$h_{\rm L1}$"
    )

    # coinc: S-곡선 (net/H1/L1) + 라벨색 마커 (모양=클러스터)
    plot_coinc(coinc, tzero=t0, ax=ax_l3)
    ax_l3.set_yscale("symlog", linthresh=3)   # S: 0~3 linear, 그 위 log (스파이크 가시화)
    ax_l3.set_ylim(bottom=0)
    ax_l3.set_xlabel(f"Time (s) from {t0}")

    # coinc 마커: 색 = 2차 DBSCAN 클러스터(tab10), 모양 = 클러스터 구분
    cmap_clusters = plt.get_cmap("tab10")
    unique_clusters = (
        coinc.filter(pl.col("coincl_id").is_not_null())["coincl_id"].unique().sort()
    )
    for i, c_id in enumerate(unique_clusters):
        subset = coinc.filter(pl.col("coincl_id") == c_id)
        ax_l3.scatter(
            (subset["time_bin"] - t0).to_numpy(),
            subset["S"].to_numpy(),
            color=cmap_clusters(i % 10),
            marker=cluster_markers[i % len(cluster_markers)],
            s=40,
            zorder=5,
        )

    # ── 분류 오버레이 (고정폭 박스 = 특징추출 윈도우) ────────────
    if classif is not None and len(classif) > 0:
        times = classif["times"].to_numpy()
        labels = classif["label"].to_numpy()
        glc_detail = classif["glc_detail"].to_numpy()
        p_C = classif["p_C"].to_numpy()
        p_dH = classif["p_dH"].to_numpy()
        p_dL = classif["p_dL"].to_numpy()

        # shading
        for i in range(len(times)):
            t_rel = times[i] - t0
            t_min = t_rel - ar_seg_dur / 2
            t_max = t_rel + ar_seg_dur / 2
            col = label_colors[labels[i]]
            for ax in left_axes:
                ax.axvspan(t_min, t_max, color=col, alpha=0.25, zorder=0)

        # 라벨 박스: GW=p_C / GLC=flagged 검출기 p_d (HL이면 둘 다) / BKG=박스 생략
        def _mt(p):                               # p -> mathtext "m\times10^{e}"
            if not np.isfinite(p) or p <= 0:
                return "0"
            e = int(np.floor(np.log10(p))); m = p / 10.0 ** e
            return rf"{m:.1f}\times10^{{{e}}}"
        items = []
        for i in range(len(times)):
            if labels[i] == "BKG":
                continue                          # background: 음영/마커만, 박스 없음
            if labels[i] == "GW":
                txt = f"GW\n$p_C={_mt(p_C[i])}$"  # GW: 결정한 양은 coherence p_C
            else:                                 # GLC: 글리치로 판정된 검출기의 p_d만
                det = glc_detail[i]
                lines = [f"GLC{det}"]
                if "H" in det:
                    lines.append(rf"$p_{{d,\mathrm{{H}}}}={_mt(p_dH[i])}$")
                if "L" in det:
                    lines.append(rf"$p_{{d,\mathrm{{L}}}}={_mt(p_dL[i])}$")
                txt = "\n".join(lines)
            items.append(dict(x=times[i] - t0, txt=txt,
                              color=label_colors[labels[i]]))

        base_x = (coinc["time_bin"] - t0).to_numpy()
        base_y = coinc["S"].to_numpy()
        finite = np.isfinite(base_y)
        base_y = np.where(finite, base_y, base_y[finite].max() if finite.any() else 0.0)
        place_trigger_labels(ax_l3, fig, items, base_x, base_y, data_gap_frac=0.1)

    # ── 공통 스타일 ──────────────────────────────────────────────
    leg = ax_l3.get_legend()
    if leg is not None:
        ax_l3.legend(loc="center right", bbox_to_anchor=(1.0, -0.25),
                     ncol=3, fontsize=10, framealpha=0.8)
    
    plt.setp(ax_l1.get_xticklabels(), visible=False)
    plt.setp(ax_l2.get_xticklabels(), visible=False)
    
    for ax in left_axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        ax.yaxis.label.set_size(12)
        ax.title.set_size(14)    
    ax_l3.xaxis.label.set_size(12)
        
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_refit_status(seg_dir, batch_id, window=128):
    """AR model refit status timeline: full / gated / skip per H1/L1.

    Args:
        seg_dir: beacon_results directory (Path or str).
        batch_id: current batch number (window end).
        window: number of batches to display (default 128 = 512s).
    """
    seg_dir = Path(seg_dir)
    summary = pl.read_parquet(str(seg_dir / "summary.parquet"))
    batch_lo = max(1, batch_id - window + 1)
    sw = summary.filter(
        (pl.col("batch_id") >= batch_lo) & (pl.col("batch_id") <= batch_id)
    )

    status_colors = {"full": "seagreen", "gated": "orange", "skip": "whitesmoke"}
    status_edge = {"full": "seagreen", "gated": "orange", "skip": "lightgray"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 3), sharex=True)

    for ax, det in zip(axes, ["H1", "L1"]):
        det_rows = (sw.filter(pl.col("detector") == det)
                      .sort("batch_id"))
        batch_ids = det_rows["batch_id"].to_numpy()
        statuses = det_rows["fit_status"].to_numpy()
        rel_idx = batch_ids - batch_id

        for st in ["full", "gated", "skip"]:
            m = statuses == st
            spans = [(x - 0.5, 1) for x in rel_idx[m]]
            if spans:
                alpha = 0.3 if st == "skip" else 0.7
                ax.broken_barh(spans, (0, 1),
                               facecolors=status_colors[st], alpha=alpha,
                               edgecolors=status_edge[st], linewidth=0.3)

        n_total = len(statuses)
        n_full = int((statuses == "full").sum())
        n_gated = int((statuses == "gated").sum())
        n_skip = int((statuses == "skip").sum())
        n_refit = n_full + n_gated
        frac = n_refit / n_total if n_total > 0 else 0
        ax.set_ylabel(det, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(1.01, 0.5,
                f"refit {n_refit}/{n_total} ({frac:.0%})\n"
                f"full {n_full} | gated {n_gated} | skip {n_skip}",
                transform=ax.transAxes, va="center", fontsize=8)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_visible(True)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="seagreen", alpha=0.7, label="full"),
        Patch(facecolor="orange", alpha=0.7, label="gated"),
        Patch(facecolor="whitesmoke", edgecolor="lightgray", label="skip"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper left",
                   ncol=3, fontsize=8, frameon=True)

    axes[1].set_xlabel("Relative batch index")
    fig.suptitle(f"AR Model Refit Status — batch {batch_lo}–{batch_id}",
                 fontsize=12)
    plt.tight_layout()
    plt.show()
