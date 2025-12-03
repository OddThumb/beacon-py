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
    ax.xaxis.set_major_locator(MaxNLocator(10))

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
        >>> psd_fs = compute_psd(ts_obj)  # hypothetical PSD function
        >>> ax = plot_freq(psd_fs, logf=True, logy=True)
    """
    # Get frequency axis and data
    freqs = fs_obj.freqs()
    data = fs_obj.data

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
    ax.set_ylabel("Amplitude")

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
    loc_colorbar="ul",
    dir_colorbar="vertical",
    label_colorbar=None,
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
    ax = plot_oscillo(ts_crop, tzero=tzero, trange=trange, ax=ax_osci, lw=0.5)

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
    figsize: tuple[float, float] = (8, 3),
    lw: float = 0.5,
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

    fig, ax = plt.subplots(figsize=figsize)

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

    ax.set_xlabel(r"Time$ - t_0$ (s)")
    if time_shifted.size > 0:
        ax.set_xlim(time_shifted[0], time_shifted[-1])
    ax.xaxis.set_major_locator(MaxNLocator(10))

    ymax = np.max(np.abs(y)) if y.size > 0 else 0.0
    value_order = int(np.floor(np.log10(ymax))) if ymax > 0 else 0
    scale_factor = (10**value_order) if value_order != 0 else 1

    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / scale_factor:.1f}"))
    ax.set_ylabel(f"$h~(10^{{{value_order}}})$" if value_order != 0 else r"$h$")

    if title:
        ax.set_title(title)

    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_lambda(
    res_net: "Rist",
    lambda_type: Literal["a", "c"] = "a",
    chunk_len: float = 1.0,
    t_from: Optional[float] = None,
    figsize: tuple = (8, 4),
) -> None:
    """
    Plot the update history of lambda_a or lambda_c using res_net (Rist object).

    Args:
        res_net: Rist object containing detector-wise lambda update history.
        lambda_type: Either "a" for λ_a or "c" for λ_c.
        chunk_len: Batch duration to scale time axis.
        t_from: Optional float to label x-axis origin.
    """
    if lambda_type not in {"a", "c"}:
        raise ValueError("lambda_type must be either 'a' or 'c'")

    extract_key = lambda_type
    y_label = r"$\lambda_a$" if lambda_type == "a" else r"$\lambda_c$"
    title = (
        r"Update history of $\lambda_a$"
        if lambda_type == "a"
        else r"Update history of $\lambda_c$"
    )

    df_dict = {}
    for det in res_net.names:
        lamb_list = res_net[det].lamb  # This is a Rist of Rist(a=..., c=...)
        values = [l[extract_key] for l in lamb_list]
        df_dict[det] = values

    n_rows = len(next(iter(df_dict.values())))
    df_dict["tt"] = [chunk_len * (i + 1) for i in range(n_rows)]
    df = pl.DataFrame(df_dict)

    df_long = df.unpivot(index=["tt"], variable_name="detector", value_name="value")

    fig, ax = plt.subplots(figsize=figsize)
    for det in df_long["detector"].unique():
        sub = df_long.filter(pl.col("detector") == det)
        ax.plot(sub["tt"].to_numpy(), sub["value"].to_numpy(), label=det)

    xlabel = (
        r"$\mathit{t}~(s)$" if t_from is None else rf"$\mathit{{t}}~(s)$ from {t_from}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(title="Detector", loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_coinc(
    df: pl.DataFrame,
    *,
    tzero: Optional[float] = None,
    p_crit: float = 0.05,
    a: float = 2.3,  # e.g., b1p1k=2.3, b8p4k=1.6
    alpha_det: float = 0.3,
    legend_position: str = "tr",  # one of {"tr","tl","br","bl"}
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
    legend_position : {"tr","tl","br","bl"}
        Legend corner inside the axes.
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
    loc_map = {
        "tr": "upper right",
        "tl": "upper left",
        "br": "lower right",
        "bl": "lower left",
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
    ax.legend(loc=loc_map.get(legend_position, "upper right"), frameon=True)
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
