# BEACON: Burst Event Anomaly Clustering and Outlier Notification (Python)

[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2+-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

BEACON is a fully data-driven, template-free pipeline for detecting unmodeled
gravitational-wave (GW) transients. It couples sequential autoregressive
denoising ([seqARIMA](https://doi.org/10.1103/PhysRevD.109.102003)) with anomaly
clustering and a cross-detector coherence veto by AR feature, giving a low-latency,
model-agnostic framework that runs on streaming or batched data.

## Key Features

- **Time / frequency series** via the `ts` and `fs` classes, with PyCBC- and
  GWpy-compatible conversions.
- **seqARIMA denoising** — off-source autoregressive fitting (Burg estimation)
  and on-source whitening, with optional zero-phase response.
- **Anomaly detection** using a robust, adaptive IQR threshold.
- **Clustering** of temporal outliers with DBSCAN.
- **Significance model** — per-cluster anomaly count (zero-truncated Poisson,
  `λ_a`) and inter-cluster waiting time (Exponential, `λ_c`) combined into a
  null probability `P0`.
- **Coincidence analysis** across a detector network.
- **Autoregressive veto / classification** — a cross-detector coherence
  statistic `C` separates GW candidates from noise, and a per-detector feature
  distance `d²` separates instrumental glitches (GLC) from background (BKG),
  against a background reference fitted with a Minimum Covariance Determinant
  (MCD) estimator.
- **Adaptive noise model** — the denoise and significance models are refitted
  on clean batches as the detector noise drifts.
- **Diagnostics** — oscillograms, spectrograms, anomaly/coincidence overlays,
  background-reference and classification dashboards.

## Pipeline Overview

```text
   seqARIMA denoising        off-source AR fit, on-source whitening
            │
   IQR anomaly detection     adaptive robust threshold
            │
   DBSCAN clustering         group temporally adjacent anomalies
            │
   Significance model        ZT-Poisson (λ_a) + Exponential (λ_c) → P0
            │
   Coincidence analysis      network P0 across detectors
            │
   AR veto / classification  coherence C : GW vs noise
                             distance d² : glitch (GLC) vs background (BKG)
            │
   Adaptive update           refit seqARIMA on clean batches
```

## Installation

```bash
git clone https://github.com/OddThumb/beacon-py.git
cd beacon-py
pip install .
```

A C compiler is required to build the Burg AR extension.

## Examples

Runnable notebooks live in [`examples/`](examples/). They fetch their data from GWOSC, so
nothing needs to be downloaded first.

| notebook | what it covers |
|---|---|
| [`seqarima_denoising.ipynb`](examples/seqarima_denoising.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OddThumb/beacon-py/blob/main/examples/seqarima_denoising.ipynb) | Fits the seqARIMA noise model off-source, whitens GW150914, and verifies the model against the measured spectrum. Denoising stage only. |

## Requirements

- Python 3.10+
- NumPy, SciPy (`==1.11.4`), pandas, Polars, PyArrow, h5py, statsmodels,
  scikit-learn, matplotlib, seaborn
- Optional: PyCBC / GWpy for data interop and Q-transform spectrograms

See [pyproject.toml](pyproject.toml) for the full dependency list.

## Quick Start

```python
import beacon as bc
import numpy as np

# Create a time series
ts_obj = bc.ts(np.random.randn(4096), start=1000.0, sampling_freq=4096)
ts_obj.plot(title="My Signal")     # oscillogram

# Frequency domain
fs_obj = ts_obj.to_fs()            # FFT (PyCBC-compatible normalization)
fs_obj.plot()
fs_obj.freqs()                     # frequency axis
fs_obj.to_ts()                     # inverse FFT
```

## Core Classes

| Class            | Purpose                                                        |
| ---------------- | ------------------------------------------------------------- |
| `bc.ts`          | Time series (`crop`, `window`, `to_fs`, `to_pycbc`, `to_gwpy`, `plot`, …) |
| `bc.fs`          | Frequency series (`freqs`, `to_ts`, `plot`); `psd` / `asd` helpers       |
| `bc.Rist`        | Named, list-like container used throughout the API                       |

## Module Map

| Module          | Contents                                                        |
| --------------- | --------------------------------------------------------------- |
| `bc.IO`         | `read_H5`, GWOSC helpers (`get_gwosc`, `get_gwosc_param`)        |
| `bc.seqARIMA`   | `seqarima`, `fit_seqarima`, `Autoregressive`, `BandPass`, PSD/response models |
| `bc.Pipe`       | `config_pipe`, `batching_network`, `arch`, `stream`, AR-veto (`fit_bkg_reference`, `classify_triggers`, `ARveto`) |
| `bc.features`   | trigger feature extraction (`extract_trigger_features`, `get_summary_feature`) |
| `bc.plot`       | `plot_oscillo`, `plot_spectro`, `plot_anomaly`, `plot_coinc`, `plot_null_ref`, `plot_classif_summary`, `plot_batch_dashboard` |
| `bc.Calc`       | numerical helpers (means, windows, interpolation)               |
| `bc.DQ`         | data-quality flag decoding                                      |

## Full Detection Pipeline

```python
import beacon as bc

# 1. Load strain for each detector
h1_ts = bc.IO.read_H5("H1.hdf5", sampling_freq=4096)
l1_ts = bc.IO.read_H5("L1.hdf5", sampling_freq=4096)
det_ts = bc.Rist(H1=h1_ts, L1=l1_ts)

# 2. Configure the pipeline (override defaults via a Rist)
cfg = bc.Pipe.config_pipe(
    replace=bc.Rist({"tbch": 4.0, "ar_ic": "HQIC", "DQ": None}),
    show_config=True,
)

# 3. Batch the network (batch-major across detectors)
batch_set = bc.Pipe.batching_network(det_ts, t_bch=4.0, has_DQ=False)

# 4. Stream the full pipeline
#    (denoise → anomaly/cluster → significance → coincidence →
#     AR veto/classification → adaptive update), checkpointed to disk
result = bc.Pipe.stream(
    batch_set=batch_set,
    config=cfg,
    checkpoint_dir="beacon_results",
)
```

Each trigger is labelled **GW**, **GLC** (glitch), or **BKG** (background) by the
autoregressive veto. Results are checkpointed to `checkpoint_dir`; inspect an
individual batch with `bc.plot.plot_batch_dashboard("beacon_results", batch_id=...)`.

> **Note:** a packaged end-to-end driver script (background training →
> streaming → classification → summary) is planned for a future release. The
> snippet above shows the underlying building blocks in the meantime.

## Denoising and Signal Processing

```python
import beacon as bc

ts_H1 = bc.IO.read_H5("H1.hdf5", sampling_freq=4096)

# seqARIMA denoising → whitened residual (a ts)
white = bc.seqARIMA.seqarima(ts_H1, p=1024, ar_ic="HQIC")

# Q-transform spectrogram
bc.plot.plot_spectro(white)
```

## PyCBC & GWpy Compatibility

```python
import beacon as bc

ts_obj = bc.IO.read_H5("strain.hdf5", sampling_freq=4096)

pycbc_ts = ts_obj.to_pycbc()       # → pycbc.types.TimeSeries
gwpy_ts  = ts_obj.to_gwpy()        # → gwpy.timeseries.TimeSeries

ts_back = bc.ts.from_pycbc(pycbc_ts)
ts_back = bc.ts.from_gwpy(gwpy_ts)
```

## Documentation

- **Repository:** <https://github.com/OddThumb/beacon-py>

Real GW strain data is available from [GWOSC](https://gwosc.org/).

## Publications

If you use BEACON, please cite:

> Kim et al., "Autoregressive Search of Gravitational Waves: Design of low-latency search pipeline for unmodeled transients: BEACON" (*under review*).

If you use only `seqarima`, please cite:

> [Kim et al., *Physical Review D* **109**, 102003 (2024), "Autoregressive
> Search of Gravitational Waves: Denoising"](https://doi.org/10.1103/PhysRevD.109.102003).

## License

GNU General Public License v2.0 or later (GPL-2.0+).

## Contributing

Contributions are welcome — please open an issue or pull request.

## Acknowledgments

- Built on the seqARIMA autoregressive denoising framework.
- Interoperable with PyCBC and GWpy time series.
