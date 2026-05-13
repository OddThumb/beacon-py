from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

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
from scipy.stats import chi2, beta as beta_dist, multivariate_normal


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


def plot_gmm2d(d2_H, d2_L, gmm, mask_normal, iso_dsq=None, ax=None,
               show_abnormal=True, thresholds=None):
    """2D GMM contours + data scatter on (d^2_H, d^2_L) plane.

    Args:
        d2_H: d^2 values for H1.
        d2_L: d^2 values for L1.
        gmm: fitted GaussianMixture (2 components, log10 space).
        mask_normal: boolean mask for normal component.
        iso_dsq: list of iso-d^2_joint values for contour lines.
        ax: optional axes.
        show_abnormal: whether to show abnormal component contours.
        thresholds: dict with 'tau_H' and/or 'tau_L' threshold values.
    """
    n_norm, n_abn = int(mask_normal.sum()), int((~mask_normal).sum())
    normal_idx = int(np.argmin(gmm.means_.sum(axis=1)))

    pad = 0.2
    log_range_h = np.linspace(
        np.log10(d2_H.min()) - pad, np.log10(d2_H.max()) + pad, 200
    )
    log_range_l = np.linspace(
        np.log10(d2_L.min()) - pad, np.log10(d2_L.max()) + pad, 200
    )
    grid_h, grid_l = np.meshgrid(log_range_h, log_range_l)
    pos = np.stack([grid_h, grid_l], axis=-1)

    xlo, xhi = 10 ** log_range_h[0], 10 ** log_range_h[-1]
    ylo, yhi = 10 ** log_range_l[0], 10 ** log_range_l[-1]
    lo, hi = min(xlo, ylo), max(xhi, yhi)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    comp_colors = ["red", "blue"]
    comp_labels = ["Normal", "Abnormal"]
    comp_order = [normal_idx, 1 - normal_idx]

    for i, ci in enumerate(comp_order):
        if i == 1 and not show_abnormal:
            continue
        rv = multivariate_normal(gmm.means_[ci], gmm.covariances_[ci])
        pdf_vals = rv.pdf(pos) * gmm.weights_[ci]
        levels = [pdf_vals.max() * np.exp(-0.5 * n**2) for n in [3, 2, 1]]
        ax.contour(
            10**grid_h, 10**grid_l, pdf_vals,
            levels=levels, colors=comp_colors[i], alpha=0.7,
        )
        ax.scatter(
            10 ** gmm.means_[ci, 0], 10 ** gmm.means_[ci, 1],
            color=comp_colors[i], marker="x", s=100,
            label=f"{comp_labels[i]} (w={gmm.weights_[ci]:.2f})",
        )

    if iso_dsq:
        for S in iso_dsq:
            d2h_line = np.linspace(0.01, S - 0.01, 500)
            d2l_line = S - d2h_line
            ax.plot(d2h_line, d2l_line, color="green", ls="--", lw=1.5, alpha=0.6)
            ax.text(
                S, lo, rf"$\Sigma={S}$", fontsize=7, color="green",
                rotation=-80, ha="center", va="bottom", clip_on=True,
            )

    if thresholds:
        if "tau_H" in thresholds:
            ax.axvline(thresholds["tau_H"], color="red", ls=":", lw=1.5,
                       alpha=0.7, label=rf"$\tau_H$={thresholds['tau_H']:.1f}")
        if "tau_L" in thresholds:
            ax.axhline(thresholds["tau_L"], color="blue", ls=":", lw=1.5,
                       alpha=0.7, label=rf"$\tau_L$={thresholds['tau_L']:.1f}")

    ax.scatter(
        d2_H[mask_normal], d2_L[mask_normal],
        alpha=0.25, s=5, color="red", label=f"Normal (n={n_norm})",
    )
    if show_abnormal:
        ax.scatter(
            d2_H[~mask_normal], d2_L[~mask_normal],
            alpha=0.25, s=5, color="blue", label=f"Abnormal (n={n_abn})",
        )
    else:
        ax.scatter(
            d2_H[~mask_normal], d2_L[~mask_normal],
            alpha=0.1, s=3, color="gray",
        )

    ax.set(
        xscale="log", yscale="log",
        xlabel=r"$d_{\rm H1}^2$", ylabel=r"$d_{\rm L1}^2$",
        xlim=(lo, hi), ylim=(lo, hi), aspect="equal",
    )
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    if standalone:
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


def plot_bkg_summary(bkg_fts, bkg_ref, fap_c=0.053, alpha_d=0.05):
    """BKG pool summary: 2x2 diagnostic plot.

    Layout:
        [GMM 2D (d^2_H vs d^2_L) | C vs Beta        ]
        [d^2_H vs chi^2(df_mle_H) | d^2_L vs chi^2(df_mle_L)]

    Args:
        bkg_fts: dict with 'dH', 'dL', 'uH', 'uL', 'gmm', 'mask_normal'.
        bkg_ref: dict with 'df_mle_H', 'df_mle_L', 'alpha_beta'.
        fap_c: FAP for C threshold.
        alpha_d: significance level for d^2 threshold.
    """
    mask = bkg_fts["mask_normal"]
    d2H = bkg_fts["dH"][mask] ** 2
    d2L = bkg_fts["dL"][mask] ** 2
    C = (bkg_fts["uH"] * bkg_fts["uL"]).sum(axis=1)

    n_total = len(bkg_fts["dH"])
    n_norm = int(mask.sum())
    print(f"BKG Pool: {n_total} triggers")
    print(f"  GMM: Normal={n_norm}, Abnormal={n_total - n_norm}")
    print(f"  chi2 MLE: H1 df={bkg_ref['df_mle_H']:.2f}, "
          f"L1 df={bkg_ref['df_mle_L']:.2f}")
    print(f"  Beta: a=b={bkg_ref['alpha_beta']:.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax in axes.flat:
        ax.set_box_aspect(1)

    plot_gmm2d(
        bkg_fts["dH"] ** 2, bkg_fts["dL"] ** 2,
        bkg_fts["gmm"], bkg_fts["mask_normal"], ax=axes[0, 0],
    )
    axes[0, 0].set_aspect("auto")

    plot_dist_beta(C, bkg_ref["alpha_beta"], fap=fap_c, ax=axes[0, 1])
    plot_dist_chi2(d2H, bkg_ref["df_mle_H"], "H1", alpha=alpha_d, ax=axes[1, 0])
    plot_dist_chi2(d2L, bkg_ref["df_mle_L"], "L1", alpha=alpha_d, ax=axes[1, 1])

    fig.suptitle(f"BKG Pool (n={n_total})", fontsize=13)
    plt.tight_layout()
    plt.show()


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


def plot_diag_classified(res_net, coinc_clust, classif_res, ar_seg_dur=32 * 15 / 4096):
    """Diagnostic dashboard with classification overlay.

    Extends plot_diag with trigger shading (3 panels) and classification
    labels with p-value annotations.

    Args:
        res_net: BEACON pipeline result Rist (H1/L1).
        coinc_clust: clustered coincidence DataFrame.
        classif_res: classification result DataFrame.
        ar_seg_dur: AR segment duration in seconds (for shading width).
    """

    plot_data = pl.DataFrame([
        {"index": i, "Detector": det,
         "lambda_c": it["stats"]["lambda_c"],
         "lambda_a": it["stats"]["lambda_a"]}
        for det in res_net.names
        for i, it in enumerate(res_net[det]["ustat"])
    ])
    t0 = min(res_net["H1"]["proc"]["time"][0],
             res_net["L1"]["proc"]["time"][0])

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
    ax_l1 = fig.add_subplot(gs[0:2, 0:2])
    ax_l2 = fig.add_subplot(gs[2:4, 0:2], sharex=ax_l1)
    ax_l3 = fig.add_subplot(gs[4:6, 0:2], sharex=ax_l1)
    ax_r1 = fig.add_subplot(gs[0:3, 2])
    ax_r2 = fig.add_subplot(gs[3:6, 2], sharex=ax_r1)
    left_axes = [ax_l1, ax_l2, ax_l3]

    plot_anomaly(res_net["H1"]["proc"], tzero=t0, ax=ax_l1,
                 title=None, xlabel="", ylabel=r"$h_{\rm H1}$")
    plot_anomaly(res_net["L1"]["proc"], tzero=t0, ax=ax_l2,
                 title=None, xlabel="", ylabel=r"$h_{\rm L1}$")
    plot_coinc_clust(coinc_clust, tzero=t0, ax=ax_l3)
    ax_l3.set_xlabel(f"Time (s) from {t0}")

    det_colors = {"H1": "red", "L1": "blue"}
    for det in plot_data["Detector"].unique().to_list():
        subset = plot_data.filter(pl.col("Detector") == det)
        ax_r1.plot(subset["index"] - len(subset["index"]) + 1, subset["lambda_c"],
                   marker="o", color=det_colors.get(det), label=det, alpha=0.5)
        ax_r2.plot(subset["index"] - len(subset["index"]) + 1, subset["lambda_a"],
                   marker="o", color=det_colors.get(det), label=det, alpha=0.5)
    ax_r1.set_ylabel(r"$\lambda_c$")
    ax_r2.set_ylabel(r"$\lambda_a$")
    ax_r2.set_xlabel("Relative batch index")
    ax_r2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_r2.legend(loc="best", frameon=True)

    times = classif_res['times'].to_numpy()
    labels = classif_res["label"].to_numpy()
    glc_detail = classif_res["glc_detail"].to_numpy()
    p_C = classif_res["p_C"].to_numpy()
    p_dH = classif_res["p_dH"].to_numpy()
    p_dL = classif_res["p_dL"].to_numpy()

    label_colors = {"GW": "limegreen", "GLC": "darkorange", "BKG": "gray"}
    for i in range(len(times)):
        t_rel = times[i] - t0
        t_min = t_rel - ar_seg_dur / 2
        t_max = t_rel + ar_seg_dur / 2
        col = label_colors[labels[i]]
        for ax in left_axes:
            ax.axvspan(t_min, t_max, color=col, alpha=0.25, zorder=0)

    plt.setp(ax_l1.get_xticklabels(), visible=False)
    plt.setp(ax_l2.get_xticklabels(), visible=False)
    plt.setp(ax_r1.get_xticklabels(), visible=False)
    for ax in left_axes + [ax_r1, ax_r2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    ymin3, ymax3 = ax_l3.get_ylim()
    renderer = fig.canvas.get_renderer()
    inv = ax_l3.transData.inverted()
    tmp = ax_l3.text(0, 0, "GW\npC=1.0e-01", fontsize=7,
                     bbox=dict(pad=2, boxstyle="round,pad=0.3"))
    bb_data = inv.transform(tmp.get_window_extent(renderer).get_points())
    box_h = bb_data[1, 1] - bb_data[0, 1]
    tmp.remove()

    margin = box_h * 0.3
    y_text = ymax3 + margin + box_h / 2
    new_ymax = y_text + box_h / 2 + margin
    ax_l3.set_ylim(ymin3, new_ymax)

    texts = []
    target_x, target_y = [], []
    for i in range(len(times)):
        t_rel = times[i] - t0
        col = label_colors[labels[i]]
        if labels[i] == "GW":
            txt = f"GW\npC={p_C[i]:.1e}"
        else:
            txt = (f"{labels[i]}{glc_detail[i]}\n"
                   f"pdH={p_dH[i]:.1e}\npdL={p_dL[i]:.1e}")
        txt_obj = ax_l3.text(t_rel, y_text, txt,
                             fontsize=7, ha="center", va="center",
                             bbox=dict(fc=col, alpha=0.4, pad=2,
                                       boxstyle="round,pad=0.3"),
                             zorder=10)
        texts.append(txt_obj)
        target_x.append(t_rel)
        target_y.append(ymax3)

    fig.canvas.draw()
    adjust_text(texts, ax=ax_l3,
                target_x=target_x, target_y=target_y,
                force_text=(0.5, 1),
                force_static=(2.0, 1.0),
                force_pull=(1e-2, 1e-2),
                force_explode=(0.5, 0.5),
                max_move=(50, 25),
                expand=(1.1, 1.0),
                ensure_inside_axes=True,
                min_arrow_len=5,
                iter_lim=5,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    plt.show()


def plot_det_bkg(classif_res, bkg_fts, bkg_ref, fap_c=0.053, alpha_d=0.05):
    """2x2 classification result plot with BKG reference overlay.

    Layout:
        [2D: Normal contour + NOS triggers | C vs Beta + markers]
        [d^2_H 1D + markers                | d^2_L 1D + markers]

    Args:
        classif_res: classification result DataFrame.
        bkg_fts: dict with 'dH', 'dL', 'uH', 'uL', 'gmm', 'mask_normal'.
        bkg_ref: dict with 'df_mle_H', 'df_mle_L', 'alpha_beta'.
        fap_c: FAP for C threshold.
        alpha_d: significance level for d^2 threshold.
    """
    d2H_new = classif_res['d2H'].to_numpy()
    d2L_new = classif_res['d2L'].to_numpy()
    C_new = classif_res['C'].to_numpy()
    labels = classif_res['label'].to_numpy()
    ab = bkg_ref["alpha_beta"]
    tau_C = 2 * beta_dist.ppf(1 - fap_c, ab, ab) - 1
    tau_H = chi2.ppf(1 - alpha_d, bkg_ref["df_mle_H"])
    tau_L = chi2.ppf(1 - alpha_d, bkg_ref["df_mle_L"])

    mask_bkg_norm = bkg_fts["mask_normal"]
    d2H_bkg = bkg_fts["dH"][mask_bkg_norm] ** 2
    d2L_bkg = bkg_fts["dL"][mask_bkg_norm] ** 2
    C_bkg = (bkg_fts["uH"] * bkg_fts["uL"]).sum(axis=1)

    label_colors = {"GW": "limegreen", "GLC": "darkorange", "BKG": "gray"}
    n_triggers = len(C_new)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax in axes.flat:
        ax.set_box_aspect(1)

    plot_gmm2d(
        bkg_fts["dH"] ** 2, bkg_fts["dL"] ** 2,
        bkg_fts["gmm"], bkg_fts["mask_normal"],
        ax=axes[0, 0], show_abnormal=False,
        thresholds={"tau_H": tau_H, "tau_L": tau_L},
    )
    axes[0, 0].set_aspect("auto")
    for lab in ["BKG", "GLC"]:
        m = labels == lab
        if m.any():
            axes[0, 0].scatter(
                d2H_new[m], d2L_new[m], s=80, marker="*",
                color=label_colors[lab], edgecolors="k", linewidths=0.5,
                label=f"new {lab} (n={m.sum()})", zorder=5,
            )
    axes[0, 0].legend(loc="upper left", fontsize=8)

    plot_dist_beta(C_bkg, bkg_ref["alpha_beta"], fap=fap_c, ax=axes[0, 1])
    for lab in ["BKG", "GLC", "GW"]:
        m = labels == lab
        if m.any():
            y_pdf = beta_dist.pdf((C_new[m] + 1) / 2, ab, ab) / 2
            axes[0, 1].scatter(
                C_new[m], y_pdf, s=80, marker="*",
                color=label_colors[lab], edgecolors="k", linewidths=0.5,
                label=f"new {lab}", zorder=5,
            )
    axes[0, 1].legend(fontsize=8)

    plot_dist_chi2(d2H_bkg, bkg_ref["df_mle_H"], "H1", alpha=alpha_d, ax=axes[1, 0])
    for lab in ["BKG", "GLC"]:
        m = labels == lab
        if m.any():
            y_pdf = chi2.pdf(d2H_new[m], bkg_ref["df_mle_H"])
            axes[1, 0].scatter(
                d2H_new[m], y_pdf, s=80, marker="*",
                color=label_colors[lab], edgecolors="k", linewidths=0.5,
                label=f"new {lab}", zorder=5,
            )
    axes[1, 0].legend(fontsize=8)

    plot_dist_chi2(d2L_bkg, bkg_ref["df_mle_L"], "L1", alpha=alpha_d, ax=axes[1, 1])
    for lab in ["BKG", "GLC"]:
        m = labels == lab
        if m.any():
            y_pdf = chi2.pdf(d2L_new[m], bkg_ref["df_mle_L"])
            axes[1, 1].scatter(
                d2L_new[m], y_pdf, s=80, marker="*",
                color=label_colors[lab], edgecolors="k", linewidths=0.5,
                label=f"new {lab}", zorder=5,
            )
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"Classification (n={n_triggers} triggers)", fontsize=13)
    plt.tight_layout()
    plt.show()
