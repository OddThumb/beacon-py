import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Sequence, Optional


def amax(x: np.ndarray) -> float:
    return np.nanmax(np.abs(x))


def amin(x: np.ndarray) -> float:
    return np.nanmin(np.abs(x))


def get_order(x: np.ndarray) -> float:
    """
    Calculate the order of magnitude.

    Returns the order of magnitude of the maximum absolute value.
    For example: 1234 → 1000, 0.0056 → 0.01

    Args:
        x (np.ndarray): Input array.

    Returns:
        float: Order of magnitude (power of 10).
    """
    max_abs = amax(x)
    if max_abs == 0 or np.isnan(max_abs):
        return np.nan
    return 10 ** np.floor(np.log10(max_abs))


def invweight_prod(x: Sequence[float], w: Optional[float] = None) -> float:
    """
    Inverse-weighted product with optional NA removal.

    Args:
        x (Sequence[float]): Input data.
        w (float, optional): Weight factor. If None, use L / length(na.omit(x)).

    Returns:
        float: min(w * prod(x), 1.0)
    """
    x = np.asarray(x)
    x_omit = x[~np.isnan(x)]
    if w is None:
        L = len(x)
        w = L / len(x_omit) if len(x_omit) > 0 else np.nan
    result = w * np.prod(x_omit) if len(x_omit) > 0 else np.nan
    return min(result, 1.0) if not np.isnan(result) else np.nan


def ari_mean(x: Sequence[float], na_rm: bool = True) -> float:
    """
    Arithmetic mean with NA handling.

    Args:
        x (Sequence[float]): Input data.
        na_rm (bool): If True, remove np.nan values.

    Returns:
        float: Mean of x.
    """
    x = np.asarray(x)
    if na_rm:
        x = x[~np.isnan(x)]
    return np.mean(x) if len(x) > 0 else np.nan


def har_mean(x: Sequence[float], na_rm: bool = True) -> float:
    """
    Harmonic mean with NA handling.

    Args:
        x (Sequence[float]): Input data.
        na_rm (bool): If True, remove np.nan values.

    Returns:
        float: Harmonic mean of x.
    """
    x = np.asarray(x)
    if na_rm:
        x = x[~np.isnan(x)]
    # Floor at 1e-300 (not finfo.tiny): avoids 1/0 AND keeps sum(1/x) from overflowing
    # (1/1e-300=1e300, summable over a window; finfo.tiny gives 1/x~4.5e307 -> overflow).
    x = np.clip(x, 1e-300, None)
    return len(x) / np.sum(1.0 / x) if len(x) > 0 else np.nan


def geo_mean(x: Sequence[float], na_rm: bool = True) -> float:
    """
    Geometric mean with NA handling.

    Args:
        x (Sequence[float]): Input data.
        na_rm (bool): If True, remove np.nan values.

    Returns:
        float: Geometric mean of x.
    """
    x = np.asarray(x)
    if na_rm:
        x = x[~np.isnan(x)]
    return np.exp(np.sum(np.log(x)) / len(x)) if len(x) > 0 else np.nan


def get_limit(x: np.ndarray, mar_frac: float = 1.5) -> tuple[float, float]:
    """
    Symmetric y-limits centered at 0 with margin factor.

    Args:
        x (np.ndarray): Input data array.
        mar_frac (float): Margin multiplier.

    Returns:
        tuple: (min, max) y-axis limits
    """
    amax_val = amax(x)
    return (-amax_val * mar_frac, amax_val * mar_frac)


def uniqdif(x: np.ndarray, tol: float = 1e-8) -> float | np.ndarray:
    """
    Wrapper of unique(diff(x)) with tolerance check.

    Args:
        x (np.ndarray): Input numeric array.
        tol (float): Tolerance for detecting non-uniform spacing. Default is 1e-8.

    Returns:
        float: A single value if uniform spacing detected.
        np.ndarray: Array of unique differences if non-uniform spacing detected.
    """
    dx = np.diff(x)
    ref = dx[0]
    unique_dx = np.unique(dx[np.abs(dx - ref) > tol])

    if len(unique_dx) == 0:
        return ref
    else:
        import warnings

        warnings.warn("Non-uniform spacing detected.")
        return np.unique(dx)


def tukey_window(n: int, alpha: float) -> np.ndarray:
    """
    Generate a Tukey (tapered cosine) window.

    Computes a Tukey window of length n with taper parameter alpha.
    The Tukey window is rectangular when alpha = 0 and becomes a Hann
    window when alpha = 1.

    Args:
        n (int): Length of the window (number of points).
        alpha (float): Shape parameter in [0, 1] controlling the fraction
            of the window inside the cosine tapered regions.
            - alpha = 0: rectangular window
            - 0 < alpha < 1: Tukey window with cosine tapers
            - alpha = 1: Hann window

    Returns:
        np.ndarray: Array of length n containing the Tukey window values.
    """
    if n <= 1:
        return np.ones(n)

    # Rectangular
    if alpha <= 0:
        return np.ones(n)

    # Hann
    if alpha >= 1:
        m = np.arange(n)
        return 0.5 * (1 - np.cos(2 * np.pi * m / (n - 1)))

    r = n - 1
    m = np.arange(n)
    edge = int(np.floor(alpha * r / 2))

    w = np.zeros(n)

    left = m <= edge
    right = m >= (r - edge)
    middle = ~(left | right)

    w[left] = 0.5 * (1 + np.cos(np.pi * (2 * m[left] / (alpha * r) - 1)))
    w[middle] = 1
    w[right] = 0.5 * (1 + np.cos(np.pi * (2 * m[right] / (alpha * r) - 2 / alpha + 1)))

    return w


def welch_window(n: int) -> np.ndarray:
    """
    Generate a Welch (parabolic) window.

    Computes a Welch window of length n. The Welch window is a
    parabolic curve that tapers smoothly to zero at the edges.

    Args:
        n (int): Length of the window (number of points).

    Returns:
        np.ndarray: Array of length n containing the Welch window values.
    """
    if n <= 1:
        return np.ones(n)
    m = np.arange(n)
    mid = (n - 1) / 2
    return 1 - ((m - mid) / mid) ** 2


def cyclic(x: np.ndarray, n: int) -> np.ndarray:
    """
    Perform cyclic shift of array.

    Args:
        x (np.ndarray): Input array.
        n (int): Number of positions to shift.

    Returns:
        np.ndarray: Shifted array.
    """
    if n == 0:
        return x
    else:
        return np.concatenate([x[-n:], x[:-n]])


def interp2d(x, y, z, xout, yout, method="linear"):
    """
    Bilinear 2D interpolation.

    Args:
        x (np.ndarray): x-axis grid points.
        y (np.ndarray): y-axis grid points.
        z (2D np.ndarray): Value matrix (shape: [len(y), len(x)]).
        xout (np.ndarray): Interpolation x-points.
        yout (np.ndarray): Interpolation y-points.
        method (str): Interpolation method. ('linear' or 'nearest')

    Returns:
        dict: {'x': xout, 'y': yout, 'z': interpolated matrix}
    """
    interp = RegularGridInterpolator(
        (x, y), z, method=method, bounds_error=False, fill_value=0.0
    )
    xi, yi = np.meshgrid(xout, yout, indexing="ij")
    points = np.stack([xi.ravel(), yi.ravel()], axis=-1)
    zout = interp(points).reshape(len(xout), len(yout))

    return {"x": xout, "y": yout, "z": zout}
