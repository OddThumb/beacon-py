"""AR feature extraction and representation conversion utilities."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from beacon.seqARIMA import burgar


# ---------------------------------------------------------------------------
# AR representation conversions
# ---------------------------------------------------------------------------

def ar_to_lsf(ar_coef):
    """Convert AR coefficients to Line Spectral Frequencies (LSF).

    Decomposes the AR polynomial into symmetric/antisymmetric polynomials,
    then returns the phase angles of roots on the unit circle.
    """
    p = len(ar_coef)
    if p == 0:
        return np.array([])

    a = np.r_[1.0, -np.array(ar_coef)]

    p_poly = np.r_[a, 0] + np.r_[0, a[::-1]]
    q_poly = np.r_[a, 0] - np.r_[0, a[::-1]]

    p_roots = np.roots(p_poly)
    q_roots = np.roots(q_poly)

    p_angles = np.angle(p_roots[np.abs(np.abs(p_roots) - 1) < 0.01])
    q_angles = np.angle(q_roots[np.abs(np.abs(q_roots) - 1) < 0.01])

    p_lsf = np.sort(p_angles[(p_angles > 0) & (p_angles < np.pi)])
    q_lsf = np.sort(q_angles[(q_angles > 0) & (q_angles < np.pi)])

    return np.sort(np.r_[p_lsf, q_lsf])


def parcor_to_lar(parcor):
    """Convert PARCOR (reflection coefficients) to Log Area Ratios."""
    parcor = np.asarray(parcor)
    parcor_clipped = np.clip(parcor, -0.9999, 0.9999)
    return np.log((1 + parcor_clipped) / (1 - parcor_clipped))


def ar_to_cepstrum(ar_coef, var_pred, n_cepstrum=None):
    """Convert AR coefficients to cepstral coefficients (LPCC).

    Computes c[1]..c[n_cepstrum] via recursion (c[0]=ln(sigma^2) excluded).
    """
    ar = np.asarray(ar_coef)
    p = len(ar)
    if p == 0:
        return np.array([])

    if n_cepstrum is None:
        n_cepstrum = p

    c = np.zeros(n_cepstrum)
    for n in range(1, n_cepstrum + 1):
        if n <= p:
            c[n - 1] = ar[n - 1]
            for k in range(1, n):
                c[n - 1] += (k / n) * c[k - 1] * ar[n - k - 1]
        else:
            for k in range(n - p, n):
                if k > 0 and (n - k - 1) < p:
                    c[n - 1] += (k / n) * c[k - 1] * ar[n - k - 1]
    return c


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_segment(ts_obj, center_time, duration=0.25):
    """Extract a segment around center_time from a ts object.

    Args:
        ts_obj: beacon ts object.
        center_time: GPS time of the segment center.
        duration: segment length in seconds.

    Returns:
        np.ndarray or None if the segment is too short.
    """
    fs = ts_obj.sampling_freq
    data = ts_obj.data

    center_idx = int(round((center_time - ts_obj.start) * fs))
    half_samples = int(duration * fs / 2)

    start_idx = max(0, center_idx - half_samples)
    end_idx = min(len(data), center_idx + half_samples)

    segment = data[start_idx:end_idx]
    if len(segment) < 100:
        return None
    return segment


# ---------------------------------------------------------------------------
# Trigger-level AR feature extraction
# ---------------------------------------------------------------------------

def extract_trigger_features(
    denoised_ts,
    trigger_times,
    segment_duration=32 * 15 / 4096,
    order_max=32,
    ic=None,
    rep="lar",
):
    """Extract AR features at each trigger time for a single IFO.

    Args:
        denoised_ts: beacon ts time series (single IFO).
        trigger_times: array of trigger GPS times.
        segment_duration: AR segment length in seconds.
        order_max: maximum AR order.
        ic: information criterion (None = fixed order_max).
        rep: representation type ('lar', 'parcor', 'ar', 'lsf', 'cepstrum').

    Returns:
        np.ndarray, shape (n_triggers, order_max).
    """
    rep_extractors = {
        "ar":       lambda fit: fit.ar,
        "parcor":   lambda fit: (fit.partialacf[:fit.order]
                                 if fit.order > 0 else np.zeros(0)),
        "lsf":      lambda fit: ar_to_lsf(fit.ar),
        "lar":      lambda fit: parcor_to_lar(
                        fit.partialacf[:fit.order]
                        if fit.order > 0 else np.zeros(0)),
        "cepstrum": lambda fit: ar_to_cepstrum(fit.ar, fit.var_pred, fit.order),
    }
    extractor = rep_extractors[rep]
    features = np.empty((len(trigger_times), order_max))
    for i, t in enumerate(trigger_times):
        seg = extract_segment(denoised_ts, center_time=t,
                              duration=segment_duration)
        if seg is None:
            features[i] = np.nan
            continue
        try:
            fit = burgar(seg, ic=ic, order_max=order_max)
            features[i] = extractor(fit)
        except Exception as e:
            print(f"AR extraction failed at t={t}: {e}")
            features[i] = np.nan
    return features


def extract_raw_features(
    res_net,
    triggers,
    segment_duration=None,
    order_max=32,
    seg_factor=15,
    ic=None,
    sampling_freq=4096,
):
    """Extract per-IFO AR features at trigger locations.

    Args:
        res_net: BEACON pipeline result (Rist with H1/L1 proc).
        triggers: DataFrame with 'time_bin' and 'coincl_id' columns.
        segment_duration: AR segment length in seconds.
        order_max: maximum AR order.
        ic: information criterion (None = fixed order_max).
        sampling_freq: sampling frequency in Hz.

    Returns:
        (XH, XL, times, coincl_ids)
    """
    from beacon.Pipe import proc2ts

    times = triggers["time_bin"].to_numpy()
    coincl_ids = triggers["coincl_id"].to_numpy()
    if segment_duration is None:
        segment_duration = order_max * seg_factor / sampling_freq  # R * N_ARC / fs
    kw = dict(segment_duration=segment_duration,
              order_max=order_max, ic=ic, rep="lar")
    XH = extract_trigger_features(
        proc2ts(res_net["H1"]["proc"], sampling_freq=sampling_freq),
        times, **kw)
    XL = extract_trigger_features(
        proc2ts(res_net["L1"]["proc"], sampling_freq=sampling_freq),
        times, **kw)
    return XH, XL, times, coincl_ids


# ---------------------------------------------------------------------------
# Whitening and summary features
# ---------------------------------------------------------------------------

def whiten_with_bkg(target, bkg, mu=None, S=None):
    """LedoitWolf covariance whitening.

    If mu/S are None, estimate from bkg. Otherwise reuse provided values.

    Returns:
        (Z_target, (mu, S))
    """
    if mu is None or S is None:
        mu = bkg.mean(axis=0)
        cov = LedoitWolf().fit(bkg).covariance_
        eigvals, eigvecs = np.linalg.eigh(cov)
        nz = eigvals > 1e-10
        S = eigvecs[:, nz] @ np.diag(1.0 / np.sqrt(eigvals[nz])) @ eigvecs[:, nz].T
    return (target - mu) @ S, (mu, S)


def decompose_vector(Z):
    """Decompose whitened feature vectors into (d, u_hat).

    d = L2 norm, u_hat = unit vector.
    """
    d = np.linalg.norm(Z, axis=1)
    uhat = Z / (d[:, None] + 1e-30)
    return d, uhat


def get_summary_feature(XH, XL, bkg_ref):
    """Compute summary features (d^2_H, d^2_L, C) from raw features.

    Uses whitening parameters stored in bkg_ref.

    Returns:
        dict with keys 'd2H', 'd2L', 'C' — numpy arrays, length n_triggers.
    """
    ZH, _ = whiten_with_bkg(XH, None, mu=bkg_ref["mu_H"], S=bkg_ref["S_H"])
    ZL, _ = whiten_with_bkg(XL, None, mu=bkg_ref["mu_L"], S=bkg_ref["S_L"])
    dH, uH = decompose_vector(ZH)
    dL, uL = decompose_vector(ZL)
    C = (uH * uL).sum(axis=1)
    return {"d2H": dH**2, "d2L": dL**2, "C": C}
