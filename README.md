# BEACON: Burst Event Anomaly Clustering and Outlier Notification (Python)

[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2+-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

BEACON is a fully data-driven pipeline designed to detect unmodeled gravitational wave (GW) transients. By combining sequential autoregressive modeling ([seqARIMA](https://doi.org/10.1103/PhysRevD.109.102003)) and anomaly clustering, BEACON provides a low-latency and model-agnostic framework for robust burst detection.

This is the **Python implementation** of the BEACON pipeline. For the R version, see [beacon](https://github.com/OddThumb/beacon).

## Key Features

-   **Time Series Analysis** via `ts` class with comprehensive signal processing tools
-   **Frequency Series Analysis** via `fs` class with FFT support (PyCBC-compatible normalization)
-   **Denoising** via seqARIMA modeling with Burg's AR estimation
-   **Anomaly Detection** using robust IQR-based statistical thresholds
-   **Clustering** of temporal outliers with DBSCAN
-   **Statistical Evaluation** via Poisson and Exponential models
-   **Coincidence Analysis** across multiple detectors
-   Fully compatible with streaming or batch-based analysis
-   **PyCBC integration** for gravitational wave data analysis workflows

## Pipeline Overview

``` text
                  ┌─────────────────────────┐
                  │        seqARIMA         │◀─── Denoising (seqarima)
                  └───────────┬─────────────┘
                              ↓
                  ┌─────────────────────────┐
                  │    Anomaly Detection    │◀─── IQR method
                  └───────────┬─────────────┘
                              ↓
                  ┌─────────────────────────┐
                  │       Clustering        │◀─── DBSCAN clustering
                  └───────────┬─────────────┘
                              ↓
                  ┌─────────────────────────┐
                  │ Significance Evaluation │◀─── Significance (λₐ, λ꜀)
                  └───────────┬─────────────┘
                              ↓
                  ┌─────────────────────────┐
                  │  Coincidence Analysis   │◀─── Across detectors
                  └─────────────────────────┘
```

## Installation

### From Source

```bash
git clone https://github.com/OddThumb/beacon-py.git
cd beacon-py
pip install .
```

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- C compiler (for building the Burg AR extension)

See [pyproject.toml](pyproject.toml) for full dependency list.

## Quick Start

```python
import beacon
import numpy as np

# Create a time series
data = np.random.randn(4096)
ts_obj = beacon.ts(data, start=1000.0, sampling_freq=4096)

# Plot the time series
ts_obj.plot(title="My Signal")

# Convert to frequency series (FFT)
fs_obj = ts_obj.to_fs()
fs_obj.plot()
```

## Core Classes

### Time Series (`ts`)

```python
# Create time series
ts_obj = beacon.ts(data, start=1000.0, sampling_freq=4096)

# Methods
ts_obj.plot()                    # Plot oscillogram
ts_obj.to_fs()                   # Convert to frequency series (FFT)
ts_obj.to_pycbc()                # Convert to PyCBC TimeSeries
ts_obj.to_df()                   # Convert to polars DataFrame
```

### Frequency Series (`fs`)

```python
# From FFT
fs_obj = ts_obj.to_fs()

# Methods
fs_obj.plot()                    # Plot frequency series
fs_obj.to_ts()                   # Convert back to time series (IFFT)
fs_obj.freqs()                   # Get frequency axis
```

## Usage Example

### Full Detection Pipeline

```python
# Python Example
import beacon

# Load GW strain data
ts_H1 = beacon.IO.read_H5("H1.hdf5", sampling_freq=4096)
ts_L1 = beacon.IO.read_H5("L1.hdf5", sampling_freq=4096)
ts_dict = {"H1": ts_H1, "L1": ts_L1}

# Convert to Rist (named list-like container inspired by list class in "R")
ts_list = beacon.Rist(ts_dict)

# Data batch preparation
batch_set = beacon.Pipe.batching_network(ts_list)

# Configure pipeline
cfg = beacon.Pipe.config_pipe()

# Run detection pipeline
result = beacon.Pipe.stream(batch_set=batch_set, arch_params=cfg)

# In console:
# 1-th batch:
#   H1: λ_c=6.403, λ_a=3.333
#   L1: λ_c=6.445, λ_a=3.337
# 2-th batch:
#   H1: λ_c=6.403, λ_a=3.333
#   L1: λ_c=6.445, λ_a=3.337
# ...
```

### Basic Signal Processing

```python
import beacon
import matplotlib.pyplot as plt

# Load GW strain data
ts_H1 = beacon.IO.read_H5("H1.hdf5", sampling_freq=4096)
ts_L1 = beacon.IO.read_H5("L1.hdf5", sampling_freq=4096)

# Apply seqARIMA denoising
denoised_H1 = beacon.seqarima(ts_H1, order_max=30)

# Compute Q-transform spectrogram
plot_spectro(denoised_H1)
plt.tight_layout()
plt.show()
```

## PyCBC Integration

BEACON is fully compatible with PyCBC & GWpy:

```python
import beacon

# Load data as BEACON time series
ts_obj = beacon.IO.read_H5("strain.hdf5", sampling_freq=4096)

# Convert to PyCBC TimeSeries
pycbc_ts = ts_obj.to_pycbc()

# Convert to GWpy TimeSeries
gwpy_ts = ts_obj.to_gwpy()

# Convert back to BEACON (if needed)
# ts_back = beacon.ts.from_pycbc(pycbc_ts)
# ts_back = beacon.ts.from_gwpy(gwpy_ts)
```

## Documentation

-   **Python Repository:** <https://github.com/OddThumb/beacon-py>
-   **R Version:** <https://github.com/OddThumb/beacon>
-   **Detailed Usage Guide:** <https://oddthumb.github.io/beacon/articles/>

## Example Datasets

See [GWOSC](https://www.gw-openscience.org/) for real gravitational wave event data.

## Publications

If you use BEACON in your work, please cite:

> Kim et al., "Autoregressive Search of Gravitational Waves: Design of low-latency search pipeline for unmodeled transients — BEACON", *submitted*.

If you use only `seqarima` in your work, please cite:

> [Kim et al., *Physical Review D*, 2024, "Autoregressive Search of Gravitational Waves: Denoising"](https://doi.org/10.1103/PhysRevD.109.102003).

## License

This project is licensed under the GNU General Public License v2.0 or later (GPL-2.0+).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This Python implementation is based on the original R version of BEACON
- Burg's AR estimation algorithm adapted from R's `ar.burg` function
- Compatible with PyCBC and GWpy for gravitational wave data analysis
