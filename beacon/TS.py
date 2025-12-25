import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from .Calc import get_order


# GPS epoch: January 6, 1980 00:00:00 UTC
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)


def gps2utc(gps_time, origin=None):
    """
    Convert GPS time to UTC datetime.

    GPS time is the number of seconds since January 6, 1980 00:00:00 UTC.

    Args:
        gps_time (float or array-like): GPS time in seconds.
        origin (datetime, optional): Origin datetime for GPS time.
            Defaults to January 6, 1980 00:00:00 UTC.

    Returns:
        datetime or np.ndarray: UTC datetime(s).
            If input is scalar, returns datetime object.
            If input is array-like, returns array of datetime objects.
    """
    if origin is None:
        origin = GPS_EPOCH

    # Handle scalar input
    if np.isscalar(gps_time):
        return origin + timedelta(seconds=float(gps_time))

    # Handle array input
    gps_arr = np.asarray(gps_time, dtype=np.float64)
    return np.array([origin + timedelta(seconds=float(t)) for t in gps_arr])


def utc2gps(utc_time, origin=None):
    """
    Convert UTC datetime to GPS time.

    GPS time is the number of seconds since January 6, 1980 00:00:00 UTC.

    Args:
        utc_time (datetime, str, or array-like): UTC time.
            Can be datetime object, ISO format string, or array of either.
        origin (datetime, optional): Origin datetime for GPS time.
            Defaults to January 6, 1980 00:00:00 UTC.

    Returns:
        float or np.ndarray: GPS time in seconds.
            If input is scalar, returns float.
            If input is array-like, returns float64 array.
    """
    if origin is None:
        origin = GPS_EPOCH

    def _convert_single(utc):
        # Handle string input
        if isinstance(utc, str):
            utc = datetime.fromisoformat(utc.replace("Z", "+00:00"))
        # Ensure timezone aware
        if utc.tzinfo is None:
            utc = utc.replace(tzinfo=timezone.utc)
        return (utc - origin).total_seconds()

    # Handle scalar input
    if isinstance(utc_time, (datetime, str)):
        return _convert_single(utc_time)

    # Handle array input
    return np.array([_convert_single(t) for t in utc_time], dtype=np.float64)


# Inherit ts attributes
def inherit_ts_attrs(source_ts, target_ts, exclude=("data", "start", "sampling_freq")):
    """
    Inherit custom attributes from one ts object to another, including dynamic attributes.

    Args:
        source_ts (ts): Original ts object to inherit attributes from.
        target_ts (ts): Target ts object to receive attributes.
        exclude (tuple): Attribute names to exclude from inheritance.
    """
    # Copy regular attributes
    for attr, val in source_ts.__dict__.items():
        if attr not in exclude and not hasattr(target_ts, attr):
            setattr(target_ts, attr, val)

    # Copy dynamic attributes
    if hasattr(source_ts, "_dynamic_attrs") and hasattr(target_ts, "_dynamic_attrs"):
        for k, v in source_ts._dynamic_attrs.items():
            if k not in exclude and k not in target_ts._dynamic_attrs:
                target_ts._dynamic_attrs[k] = v


# ts object
class ts:
    """
    Time series object for evenly sampled data.

    Attributes:
        data (np.ndarray): Numeric array of samples.
        start (float): Start time (GPS seconds).
        sampling_freq (float): Sampling frequency in Hz.

    Initialization:
        Can be initialized with either sampling_freq or deltat (1/sampling_freq).

    Properties:
        end (float): End time.
        trange (np.ndarray): Start and end time array.
        length (int): Number of samples.
        duration (float): Total duration in seconds.
        times (np.ndarray): Array of sample times.

    Methods:
        window(start=None, end=None, extend=False):
            Extract a windowed subset of the time series.
        __repr__():
            Return string representation.
    """

    def __init__(self, data, start, sampling_freq=None, deltat=None):
        """
        Initialize the time series object.

        Args:
            data (array-like): Sequence of samples.
            start (float): Start time (GPS seconds).
            sampling_freq (float, optional): Sampling frequency in Hz.
            deltat (float, optional): Time interval between samples (1/sampling_freq).

        Note:
            Exactly one of sampling_freq or deltat must be provided.
        """
        if sampling_freq is None and deltat is None:
            raise ValueError("Either sampling_freq or deltat must be provided.")
        if sampling_freq is not None and deltat is not None:
            raise ValueError("Cannot specify both sampling_freq and deltat.")

        self.data = np.asarray(data, dtype=np.float64)
        self.start = start

        if sampling_freq is not None:
            self.sampling_freq = sampling_freq
        else:
            self.sampling_freq = 1.0 / deltat

        self._dynamic_attrs = {}

    def __getattr__(self, name):
        if name in self._dynamic_attrs:
            return self._dynamic_attrs[name]
        raise AttributeError(f"{name} not found")

    def __setattr__(self, name, value):
        if name in {"data", "start", "sampling_freq", "_dynamic_attrs"}:
            super().__setattr__(name, value)
        else:
            self._dynamic_attrs[name] = value

    def __dir__(self):
        return list(super().__dir__()) + list(self._dynamic_attrs.keys())

    @property
    def end(self):
        return self.start + (len(self.data) - 1) / self.sampling_freq

    @property
    def trange(self):
        return np.array(
            [self.start, self.start + (len(self.data) - 1) / self.sampling_freq]
        )

    @property
    def length(self):
        return len(self.data)

    @property
    def duration(self):
        return len(self.data) / self.sampling_freq

    @property
    def times(self):
        return self.start + np.arange(len(self.data)) / self.sampling_freq

    def _binary_op(self, other, op):
        """
        Core logic for binary operations.

        Args:
            other: ts instance or scalar.
            op: function implementing the operation.

        Returns:
            ts: New time series with the result.
        """
        if isinstance(other, ts):
            if abs(self.sampling_freq - other.sampling_freq) > 1e-9:
                raise ValueError("Sampling frequencies do not match.")
            # Determine overlapping time range
            new_start = max(self.start, other.start)
            new_end = min(self.end, other.end)
            if new_start > new_end:
                raise ValueError("No overlapping time range.")
            # Compute indices for self
            i0 = int(round((new_start - self.start) * self.sampling_freq))
            i1 = int(round((new_end - self.start) * self.sampling_freq)) + 1
            # Compute indices for other
            j0 = int(round((new_start - other.start) * other.sampling_freq))
            j1 = int(round((new_end - other.start) * other.sampling_freq)) + 1
            # Align data
            data_self = self.data[i0:i1]
            data_other = other.data[j0:j1]
            result_data = op(data_self, data_other)
            return ts(result_data, start=new_start, sampling_freq=self.sampling_freq)
        else:
            # Scalar operation
            result_data = op(self.data, other)
            return ts(result_data, start=self.start, sampling_freq=self.sampling_freq)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        # For scalar - ts
        if isinstance(other, ts):
            return other.__sub__(self)
        return ts(other - self.data, start=self.start, sampling_freq=self.sampling_freq)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        # For scalar / ts
        if isinstance(other, ts):
            return other.__truediv__(self)
        return ts(other / self.data, start=self.start, sampling_freq=self.sampling_freq)

    def __repr__(self):
        """
        Return a string representation of the time series.

        Returns:
            str: String summary of the object.
        """
        data_str = np.array2string(self.data, threshold=20, max_line_width=80)
        return (
            f"Time Series:\n"
            f"  Start     = {self.start}\n"
            f"  End       = {self.end}\n"
            f"  Frequency = {self.sampling_freq}\n"
            f"  Length    = {len(self.data)}\n"
            f"  Data      = {data_str}\n"
        )

    def window(self, start=None, end=None, extend=False):
        """
        Extract a time window from the series.

        Args:
            start (float, optional): Start time of the window.
                Defaults to the series start.
            end (float, optional): End time of the window.
                Defaults to the series end.
            extend (bool, optional): If True, allow padding outside original range with NaN.
                Defaults to False.

        Returns:
            ts: New time series object containing the windowed data.
        """
        xtime = self.times
        xfreq = self.sampling_freq
        eps_ = 1e-7 / xfreq

        # Default start/end
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        # Start/end clipping
        if start < self.start - eps_ and not extend:
            start = self.start
        if end > self.end + eps_ and not extend:
            end = self.end
        if start > end + eps_:
            raise ValueError("'start' cannot be after 'end'")

        if not extend:
            # extend=False
            # Snap start
            if np.all(np.abs(start - xtime) > eps_):
                candidates = xtime[(xtime > start) & ((start + 1 / xfreq) > xtime)]
                if len(candidates) > 0:
                    start = candidates[0]
            # Snap end
            if np.all(np.abs(end - xtime) > eps_):
                candidates = xtime[(xtime < end) & ((end - 1 / xfreq) < xtime)]
                if len(candidates) > 0:
                    end = candidates[-1]

            i_start = int(np.trunc((start - self.start) * xfreq + 1.5)) - 1
            # i_end   = int(np.trunc((end   - self.start) * xfreq + 1.5))
            i_end = int(np.trunc((end - self.start) * xfreq + 1.5)) - 1

            i_start = max(0, i_start)
            i_end = min(len(self.data), i_end)

            sliced = self.data[i_start:i_end]

            ystart = xtime[i_start]
        else:
            # extend=True
            stoff = int(np.ceil((start - self.start) * xfreq - 1e-7))
            enoff = int(np.floor((end - self.end) * xfreq + 1e-7))

            ystart = stoff / xfreq + self.start
            yend = enoff / xfreq + self.end

            if ystart > yend and (ystart - yend) * xfreq < 1e-7:
                yend = ystart

            nold = len(self.data)
            new_len = int(round(xfreq * (yend - ystart))) + 1

            # Determine indices
            if start > self.end + eps_ or end < self.start - eps_:
                # Entirely out of range
                i = np.full(new_len, nold)
            else:
                i0 = 0 + max(0, stoff)
                i1 = nold - 1 + min(0, enoff)
                prefix = [nold] * max(0, -stoff)
                middle = list(range(i0, i1 + 1)) if i0 <= i1 else []
                suffix = [nold] * max(0, enoff)
                i = prefix + middle + suffix

            # Pad NA if needed
            extended_data = np.concatenate([self.data, [np.nan]])
            sliced = extended_data[i]

        out = ts(sliced, start=ystart, sampling_freq=xfreq)

        # Inherit ts attributes
        inherit_ts_attrs(self, out)

        return out

    def to_df(self, tzero: float = 0.0, val_name: str = "x") -> pd.DataFrame:
        """
        Convert ts object to a DataFrame for plotting.

        Args:
            tzero (float): Time-zero shift. All times will be offset by this value.
            val_name (str): Column name to assign to signal values.

        Returns:
            pd.DataFrame: DataFrame with 'time' and signal value column.
        """
        times = self.times - tzero
        data = self.data

        if data.ndim == 1:
            return pd.DataFrame({"time": times, val_name: data})
        elif data.ndim == 2 and data.shape[1] == 1:
            return pd.DataFrame({"time": times, val_name: data[:, 0]})
        else:
            raise ValueError("Only 1D ts data or single-column 2D data is supported.")

    def to_pycbc(self):
        """
        Convert ts object to PyCBC TimeSeries.

        Returns:
            pycbc.types.TimeSeries: PyCBC TimeSeries object with same data and metadata.

        Raises:
            ImportError: If pycbc is not installed.

        Examples:
            >>> x = ts([1, 2, 3, 4, 5], start=1000.0, sampling_freq=4096.0)
            >>> pycbc_ts = x.to_pycbc()
            >>> # pycbc_ts.delta_t = 1/4096.0, pycbc_ts.start_time = 1000.0

        References:
            - PyCBC TimeSeries: https://pycbc.org/pycbc/latest/html/pycbc.types.html
        """
        try:
            from pycbc.types import TimeSeries as PyCBCTimeSeries
        except ImportError:
            raise ImportError(
                "pycbc is required for this conversion. Install with: pip install pycbc"
            )

        # PyCBC uses delta_t (time step) instead of sampling_freq
        delta_t = 1.0 / self.sampling_freq

        # Create PyCBC TimeSeries
        # epoch is the GPS time of the first sample
        if self.start is not None:
            import lal

            epoch = lal.LIGOTimeGPS(float(self.start))
        else:
            epoch = None

        return PyCBCTimeSeries(
            initial_array=self.data, delta_t=delta_t, epoch=epoch, copy=True
        )

    @classmethod
    def from_pycbc(cls, pycbc_ts):
        """
        Create ts object from PyCBC TimeSeries.

        Args:
            pycbc_ts (pycbc.types.TimeSeries): PyCBC TimeSeries object to convert.

        Returns:
            ts: Time series object with same data and metadata.

        Raises:
            ImportError: If pycbc is not installed.
            TypeError: If input is not a PyCBC TimeSeries.

        Examples:
            >>> from pycbc.types import TimeSeries as PyCBCTimeSeries
            >>> pycbc_ts = PyCBCTimeSeries([1, 2, 3, 4, 5], delta_t=1/4096.0, epoch=1000.0)
            >>> x = ts.from_pycbc(pycbc_ts)
            >>> # x.sampling_freq = 4096.0, x.start = 1000.0

        References:
            - PyCBC TimeSeries: https://pycbc.org/pycbc/latest/html/pycbc.types.html
        """
        try:
            from pycbc.types import TimeSeries as PyCBCTimeSeries
        except ImportError:
            raise ImportError(
                "pycbc is required for this conversion. Install with: pip install pycbc"
            )

        if not isinstance(pycbc_ts, PyCBCTimeSeries):
            raise TypeError("Input must be a PyCBC TimeSeries object")

        # Extract data and metadata
        data = np.asarray(pycbc_ts.data, dtype=np.float64)
        delta_t = float(pycbc_ts.delta_t)
        sampling_freq = 1.0 / delta_t

        # Get start time (epoch)
        # PyCBC epoch can be None or a GPS time
        if pycbc_ts.start_time is not None:
            start = float(pycbc_ts.start_time)
        else:
            start = 0.0

        # Create ts object
        return cls(data, start=start, sampling_freq=sampling_freq)

    def to_gwpy(self):
        """
        Convert ts object to GWpy TimeSeries.

        Returns:
            gwpy.timeseries.TimeSeries: GWpy TimeSeries object with same data and metadata.

        Raises:
            ImportError: If gwpy is not installed.

        Examples:
            >>> x = ts([1, 2, 3, 4, 5], start=1000.0, sampling_freq=4096.0)
            >>> gwpy_ts = x.to_gwpy()
            >>> # gwpy_ts.sample_rate = 4096.0 Hz, gwpy_ts.t0 = 1000.0 s

        References:
            - GWpy TimeSeries: https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/
        """
        try:
            from gwpy.timeseries import TimeSeries as GWpyTimeSeries
        except ImportError:
            raise ImportError(
                "gwpy is required for this conversion. Install with: pip install gwpy"
            )

        # Create GWpy TimeSeries
        # t0 is the GPS start time, sample_rate is in Hz
        return GWpyTimeSeries(
            data=self.data, t0=self.start, sample_rate=self.sampling_freq
        )

    @classmethod
    def from_gwpy(cls, gwpy_ts):
        """
        Create ts object from GWpy TimeSeries.

        Args:
            gwpy_ts (gwpy.timeseries.TimeSeries): GWpy TimeSeries object to convert.

        Returns:
            ts: Time series object with same data and metadata.

        Raises:
            ImportError: If gwpy is not installed.
            TypeError: If input is not a GWpy TimeSeries.

        Examples:
            >>> from gwpy.timeseries import TimeSeries as GWpyTimeSeries
            >>> gwpy_ts = GWpyTimeSeries([1, 2, 3, 4, 5], t0=1000.0, sample_rate=4096.0)
            >>> x = ts.from_gwpy(gwpy_ts)
            >>> # x.sampling_freq = 4096.0, x.start = 1000.0

        References:
            - GWpy TimeSeries: https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/
        """
        try:
            from gwpy.timeseries import TimeSeries as GWpyTimeSeries
        except ImportError:
            raise ImportError(
                "gwpy is required for this conversion. Install with: pip install gwpy"
            )

        if not isinstance(gwpy_ts, GWpyTimeSeries):
            raise TypeError("Input must be a GWpy TimeSeries object")

        # Extract data and metadata
        data = np.asarray(gwpy_ts.value, dtype=np.float64)
        sampling_freq = float(gwpy_ts.sample_rate.value)

        # Get start time (t0/epoch)
        start = float(gwpy_ts.t0.value)

        # Create ts object
        return cls(data, start=start, sampling_freq=sampling_freq)

    def to_fs(self, delta_f=None):
        """
        Transform time series to frequency series via FFT.

        Args:
            delta_f (float, optional): Frequency resolution in Hz.
                Defaults to sampling_freq / length.

        Returns:
            fs: Frequency series object with embedded ts metadata.

        Raises:
            ValueError: If delta_f is too large (would cause undersampling).

        Examples:
            >>> x = ts([1, 2, 3, 4, 5], start=0, sampling_freq=4096.0)
            >>> freq_series = x.to_fs()
            >>> # or specify frequency resolution
            >>> freq_series = x.to_fs(delta_f=0.5)

        Notes:
            - Uses FFT to transform time domain to frequency domain
            - Zero-pads the time series if needed to achieve target delta_f
            - Only positive frequencies are stored (uses rfft)
            - Inherits all dynamic attributes from the ts object

        See Also:
            - fs.to_ts(): Inverse operation (frequency → time domain)
        """
        # Import here to avoid circular dependency
        from .FS import fs
        from .TS import inherit_ts_attrs

        if delta_f is None:
            delta_f = self.sampling_freq / self.length

        tlen = int(round(1.0 / delta_f / (1 / self.sampling_freq) + 0.5))
        flen = int(round(tlen / 2) + 1)

        if tlen < self.length:
            raise ValueError(
                f"The value of delta_f={delta_f} results in undersampling. "
                f"Maximum delta_f is {1.0 / self.duration}"
            )

        padded = np.zeros(tlen, dtype=np.float64)
        padded[: self.length] = self.data

        # Normalize FFT by sampling_freq to match PyCBC convention
        fft_res = np.fft.rfft(padded) / self.sampling_freq

        fs_obj = fs(
            fft_res[:flen],
            delta_f=delta_f,
            flen=flen,
            tlen=tlen,
            sampling_freq=self.sampling_freq,
            start=self.start,
        )

        inherit_ts_attrs(self, fs_obj)

        return fs_obj

    def plot(
        self,
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
        Plot the time series as an oscillogram.

        This is a convenience wrapper around plot_oscillo() that provides
        an object-oriented interface for plotting.

        Args:
            tzero (float, optional): Time-zero adjustment in seconds.
            trange (tuple, optional): Time range to display as (start, end) in GPS time.
            ylim (str or tuple): Y-axis limits. "pm" for ±max, or (ymin, ymax) tuple.
            title (str, optional): Plot title.
            lw (float): Line width. Default: 0.5.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Creates new figure if None.
            color (str, optional): Line color. Uses default color cycle if None.
            label (str, optional): Legend label.
            figsize (tuple): Figure size if ax=None. Default: (8, 3).
            **kwargs: Additional keyword arguments passed to matplotlib plot().

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.

        Examples:
            >>> # Basic plot
            >>> x = ts([1, 2, 3, 4, 5], start=1000.0, sampling_freq=4.0)
            >>> ax = x.plot()

            >>> # With customization
            >>> ax = x.plot(title="My Signal", color="red", lw=1.0)

            >>> # Plot multiple signals on same axes
            >>> fig, ax = plt.subplots()
            >>> signal1.plot(ax=ax, label="Signal 1", color="blue")
            >>> signal2.plot(ax=ax, label="Signal 2", color="orange")
            >>> ax.legend()

            >>> # With time window
            >>> ax = x.plot(trange=(1000, 1001), tzero=1000)
        """
        from .plot import plot_oscillo

        return plot_oscillo(
            self,
            tzero=tzero,
            trange=trange,
            ylim=ylim,
            title=title,
            lw=lw,
            ax=ax,
            color=color,
            label=label,
            figsize=figsize,
            **kwargs,
        )


def shift(x, t_shift):
    """
    Shift time series by time offset.

    Args:
        x (ts): Time series object to shift.
        t_shift (float): Amount to shift in seconds.

    Returns:
        ts: Time-shifted time series object.

    Raises:
        ValueError: If t_shift is not an integer multiple of 1/sampling_freq.
    """
    if not isinstance(x, ts):
        raise TypeError("Input must be a ts object")

    freq = x.sampling_freq
    k = t_shift * freq

    # Check if shift is an integer multiple of the sampling period
    if abs(k - round(k)) > np.finfo(np.float64).eps ** 0.5:
        raise ValueError("t_shift must be an integer multiple of 1/sampling_freq")

    # Create new ts with shifted start time
    out = ts(x.data.copy(), start=x.start + t_shift, sampling_freq=freq)

    # Inherit attributes
    inherit_ts_attrs(x, out)

    return out


def shift_cyclic(x, t_cyclic):
    """
    Cyclically shift a time series.

    Args:
        x (ts): Time series object to shift.
        t_cyclic (float): Time to shift cyclically in seconds.

    Returns:
        ts: Cyclically shifted time series object.

    Example:
        >>> x = ts([1, 2, 3, 4, 5], start=0, sampling_freq=1)
        >>> shifted = shift_cyclic(x, 2.0)  # Shift by 2 samples
        >>> shifted.data  # [3, 4, 5, 1, 2]
    """
    if not isinstance(x, ts):
        raise TypeError("Input must be a ts object")

    freq = x.sampling_freq
    N = len(x.data)

    if N == 0:
        return ts(x.data.copy(), start=x.start, sampling_freq=freq)

    # Convert time shift to number of samples
    n = int(round(t_cyclic * freq))
    # Ensure positive modulo
    n = ((n % N) + N) % N

    if n == 0:
        out = ts(x.data.copy(), start=x.start, sampling_freq=freq)
    else:
        # Cyclic shift: move first n elements to end
        shifted_data = np.concatenate([x.data[n:], x.data[:n]])
        out = ts(shifted_data, start=x.start, sampling_freq=freq)

    # Inherit attributes
    inherit_ts_attrs(x, out)

    return out


def pad(x, tstart, tend, at=0.0):
    """
    Pad a time series with zeros.

    Creates a zero-filled time series from tstart to tend, with the original
    time series x inserted starting at time 'at' (ignoring x's original start time).

    Args:
        x (ts): Time series object to pad.
        tstart (float): Start time of output series.
        tend (float): End time of output series.
        at (float, optional): Time position to inject original data. Defaults to 0.0.

    Returns:
        ts: Padded time series object.

    Raises:
        ValueError: If padded series would go out of bounds.

    Example:
        >>> x = ts([1, 2, 3, 4, 5], start=0, deltat=0.1)
        >>> padded = pad(x, tstart=1.0, tend=2.0, at=1.3)
        >>> # x is inserted at t=1.3, 1.4, 1.5, 1.6, 1.7
    """
    if not isinstance(x, ts):
        raise TypeError("Input must be a ts object")

    sampling_freq = x.sampling_freq

    # Time grid for the padded series
    times = np.arange(tstart, tend + 0.5 / sampling_freq, 1.0 / sampling_freq)
    ngrid = len(times)

    # Compute insertion range in samples
    shift_samples = round((at - tstart) * sampling_freq)
    n_rows = len(x.data)
    start_idx = int(shift_samples)
    end_idx = start_idx + n_rows

    if start_idx < 0 or end_idx > ngrid:
        raise ValueError(
            f"Padded series would go out of bounds. "
            f"start_idx={start_idx}, end_idx={end_idx}, ngrid={ngrid}. "
            f"Check 'at', 'tstart', 'tend'."
        )

    # Create zero-filled array and insert data
    zerobase = np.zeros(ngrid, dtype=np.float64)
    zerobase[start_idx:end_idx] = x.data

    out = ts(zerobase, start=tstart, sampling_freq=sampling_freq)

    # Inherit attributes
    inherit_ts_attrs(x, out)

    return out


def resize(x, nlen, align="left"):
    """
    Resize a time series to target length.

    If new length is shorter, truncates data. If longer, pads with zeros.
    The alignment determines how the data is positioned.

    Args:
        x (ts): Time series object to resize.
        nlen (int): Desired length (number of samples).
        align (str, optional): Alignment mode. One of:
            - "left": Keep/pad on right (default)
            - "center": Center the data
            - "right": Keep/pad on left

    Returns:
        ts: Resized time series object.

    Raises:
        ValueError: If align is not one of "left", "center", "right".
    """
    if not isinstance(x, ts):
        raise TypeError("Input must be a ts object")

    if align not in ("left", "center", "right"):
        raise ValueError("align must be one of 'left', 'center', 'right'")

    freq = x.sampling_freq
    st = x.start
    n = len(x.data)

    if n >= nlen:
        # Truncate
        if align == "left":
            x_new = x.data[:nlen]
        elif align == "right":
            x_new = x.data[(n - nlen) :]
        else:  # center
            s = (n - nlen) // 2
            e = s + nlen
            x_new = x.data[s:e]
    else:
        # Pad with zeros
        pad = nlen - n
        if align == "left":
            x_new = np.concatenate([x.data, np.zeros(pad, dtype=np.float64)])
        elif align == "right":
            x_new = np.concatenate([np.zeros(pad, dtype=np.float64), x.data])
        else:  # center
            lp = pad // 2
            rp = pad - lp
            x_new = np.concatenate(
                [np.zeros(lp, dtype=np.float64), x.data, np.zeros(rp, dtype=np.float64)]
            )

    out = ts(x_new, start=st, sampling_freq=freq)

    # Inherit attributes
    inherit_ts_attrs(x, out)

    return out


def evenify(x):
    """
    Ensure time series has even length.

    If the time series has odd length, truncates the last sample.

    Args:
        x (ts): Time series object.

    Returns:
        ts: Time series with even length.
    """
    if not isinstance(x, ts):
        raise TypeError("Input must be a ts object")

    if len(x.data) % 2 == 1:
        # Truncate last sample
        return x.window(start=x.start, end=x.end - 1.0 / x.sampling_freq)
    else:
        # Already even, return copy
        out = ts(x.data.copy(), start=x.start, sampling_freq=x.sampling_freq)
        inherit_ts_attrs(x, out)
        return out


def unit_normalize(x):
    """
    Normalize a time series to unit scale.

    Divides the time series by its order of magnitude (power of 10).
    The scale factor is stored as a dynamic attribute 'order'.

    Args:
        x (ts): Time series object to normalize.

    Returns:
        ts: Normalized time series object with 'order' attribute.

    Example:
        >>> x = ts([1234, 5678, 9012], start=0, sampling_freq=1)
        >>> normalized = unit_normalize(x)
        >>> # Divided by 10000 (order of magnitude of 9012)
    """
    if not isinstance(x, ts):
        raise TypeError("Input must be a ts object")

    # Calculate order of magnitude
    order_val = get_order(x.data)

    if order_val == 0 or np.isnan(order_val):
        raise ValueError("Cannot normalize: all values are zero or NaN")

    # Normalize
    norm_data = x.data / order_val
    out = ts(norm_data, start=x.start, sampling_freq=x.sampling_freq)

    # Store order as dynamic attribute
    out.order = order_val

    # Inherit other attributes
    inherit_ts_attrs(x, out, exclude=("data", "start", "sampling_freq", "order"))

    return out


def unit_denormalize(x, order=None):
    """
    Restore original scale of a normalized time series.

    Multiplies the time series by the scale factor to restore original magnitude.
    The scale factor is retrieved from the 'order' attribute or provided explicitly.

    Args:
        x (ts): Normalized time series object.
        order (float, optional): Scale factor. If None, uses x.order attribute.

    Returns:
        ts: Denormalized time series object.

    Raises:
        ValueError: If no order attribute found and none provided.
    """
    if not isinstance(x, ts):
        raise TypeError("Input must be a ts object")

    # Get scale factor
    if order is not None:
        scale_factor = order
    elif hasattr(x, "order"):
        scale_factor = x.order
    else:
        raise ValueError("No 'order' attribute found and no scale factor provided")

    # Denormalize
    denorm_data = x.data * scale_factor
    out = ts(denorm_data, start=x.start, sampling_freq=x.sampling_freq)

    # Inherit attributes (excluding order since we've used it)
    inherit_ts_attrs(x, out, exclude=("data", "start", "sampling_freq", "order"))

    return out


def tsref(data, ref=None, start=None, sampling_freq=None, deltat=None):
    """
    Convert numeric data to ts object using reference timing.

    Primary use case: Transform numeric data (same length as reference) into a ts object
    with the same timing information as the reference ts object.

    Args:
        data: Input data to convert. Can be:
            - array-like: numeric data (requires ref or start+sampling_freq/deltat)
            - ts: returns copy
            - dict: with keys 'data', 'start', 'sampling_freq' (or 'deltat')
        ref (ts, optional): Reference ts object to copy timing from.
            If provided, uses ref.start and ref.sampling_freq.
        start (float, optional): Start time (used if ref not provided).
        sampling_freq (float, optional): Sampling frequency in Hz.
        deltat (float, optional): Time interval (1/sampling_freq).

    Returns:
        ts: Time series object with same length as input data.

    Raises:
        ValueError: If data length doesn't match ref length (when ref provided).

    Examples:
        >>> # Use reference ts timing
        >>> ref = ts([1, 2, 3], start=10, sampling_freq=4)
        >>> data = [4, 5, 6]  # Same length as ref
        >>> x = tsref(data, ref=ref)
        >>> # x.start = 10, x.sampling_freq = 4

        >>> # Explicit timing
        >>> x = tsref([1, 2, 3], start=0, sampling_freq=1)

        >>> # From dict
        >>> x = tsref({'data': [1, 2, 3], 'start': 0, 'sampling_freq': 1})
    """
    # Case 1: Already a ts object
    if isinstance(data, ts):
        out = ts(data.data.copy(), start=data.start, sampling_freq=data.sampling_freq)
        inherit_ts_attrs(data, out)
        return out

    # Case 2: Dictionary
    if isinstance(data, dict):
        if "data" not in data:
            raise ValueError("Dictionary must contain 'data' key")

        data_vals = data["data"]
        start_val = data.get("start", start)
        freq_val = data.get("sampling_freq", sampling_freq)
        dt_val = data.get("deltat", deltat)

        if start_val is None:
            raise ValueError("Must provide 'start' either in dict or as argument")

        if freq_val is not None:
            return ts(data_vals, start=start_val, sampling_freq=freq_val)
        elif dt_val is not None:
            return ts(data_vals, start=start_val, deltat=dt_val)
        else:
            raise ValueError(
                "Must provide either 'sampling_freq' or 'deltat' in dict or as argument"
            )

    # Case 3: Array-like with reference
    try:
        data_arr = np.asarray(data, dtype=np.float64)

        # Use reference ts timing
        if ref is not None:
            if not isinstance(ref, ts):
                raise TypeError("ref must be a ts object")

            if len(data_arr) != len(ref.data):
                raise ValueError(
                    f"Data length ({len(data_arr)}) must match reference length ({len(ref.data)})"
                )

            out = ts(data_arr, start=ref.start, sampling_freq=ref.sampling_freq)
            # Don't inherit attributes - just timing
            return out

        # Use explicit timing parameters
        if start is None:
            raise ValueError(
                "Must provide either 'ref' or 'start' for array-like input"
            )

        if sampling_freq is not None:
            return ts(data_arr, start=start, sampling_freq=sampling_freq)
        elif deltat is not None:
            return ts(data_arr, start=start, deltat=deltat)
        else:
            raise ValueError("Must provide either 'sampling_freq' or 'deltat'")

    except (ValueError, TypeError) as e:
        if "Data length" in str(e) or "ref must be" in str(e):
            raise  # Re-raise our custom errors
        raise TypeError(
            f"Cannot convert {type(data).__name__} to ts object. "
            f"Supported types: ts, array-like, dict"
        ) from e


def interpolate_ts(
    times,
    data,
    sampling_freq=None,
    deltat=None,
    t_start=None,
    t_end=None,
    kind="linear",
    fill_value=0.0,
):
    """
    Interpolate irregularly sampled data to a regularly sampled ts object.

    Takes array-like data with corresponding irregular time stamps and
    creates a regularly sampled ts object via interpolation.

    This function is useful for converting numerical simulation output or
    other irregularly sampled gravitational wave strain data into the
    standard evenly-sampled ts format required for signal processing.

    Args:
        times (array-like): Irregular time stamps for the data points.
        data (array-like): Data values corresponding to each time stamp.
        sampling_freq (float, optional): Target sampling frequency in Hz.
            Either sampling_freq or deltat must be provided.
        deltat (float, optional): Target time interval (1/sampling_freq).
            Either sampling_freq or deltat must be provided.
        t_start (float, optional): Start time for the regular grid.
            Defaults to min(times).
        t_end (float, optional): End time for the regular grid.
            Defaults to max(times).
        kind (str, optional): Interpolation method. Options:
            - 'linear': Linear interpolation (default, recommended for GW data)
            - 'nearest': Nearest-neighbor interpolation
            - 'zero', 'slinear', 'quadratic', 'cubic': Spline interpolation
            See scipy.interpolate.interp1d for full details.
        fill_value (float or 'extrapolate', optional): Value to use for points
            outside the interpolation range. Defaults to 0.0.
            Use 'extrapolate' to extrapolate beyond the data range.

    Returns:
        ts: Regularly sampled time series object.

    Raises:
        ValueError: If neither sampling_freq nor deltat is provided, or if
            times and data have different lengths.
        ImportError: If scipy is not available.

    Examples:
        >>> # Interpolate irregular gravitational wave strain data
        >>> irreg_times = np.array([0.0, 0.15, 0.27, 0.5, 0.8, 1.0])
        >>> irreg_data = np.array([0.0, 1.5, 0.8, -0.5, 0.3, 0.0])
        >>>
        >>> # Interpolate to 4 Hz sampling
        >>> regular_ts = interpolate_ts(irreg_times, irreg_data, sampling_freq=4.0)
        >>> # Now regular_ts.sampling_freq = 4.0 Hz, regular grid from 0.0 to 1.0
        >>>
        >>> # Specify custom time range
        >>> regular_ts = interpolate_ts(irreg_times, irreg_data, sampling_freq=10.0,
        ...                             t_start=0.0, t_end=0.5)

    Notes:
        - Uses scipy.interpolate.interp1d for robust 1D interpolation
        - Linear interpolation ('linear') is recommended for gravitational wave
          strain data to avoid introducing spurious features
        - For Nyquist considerations, ensure target sampling_freq is at least
          2x the highest frequency component in your data
        - Points outside the original time range are filled with fill_value (default: 0.0)

    References:
        - GW data analysis: https://cplberry.com/2020/02/09/gw-data-guides/
        - SciPy interpolation: https://docs.scipy.org/doc/scipy/reference/interpolate.html
    """
    try:
        from scipy.interpolate import interp1d
    except ImportError:
        raise ImportError(
            "scipy is required for interpolation. Install with: pip install scipy"
        )

    # Validate sampling frequency input
    if sampling_freq is None and deltat is None:
        raise ValueError("Must provide either 'sampling_freq' or 'deltat'")

    if sampling_freq is None:
        sampling_freq = 1.0 / deltat

    # Convert inputs to numpy arrays
    times = np.asarray(times, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    # Validate input lengths
    if len(times) != len(data):
        raise ValueError(
            f"times and data must have same length. "
            f"Got times: {len(times)}, data: {len(data)}"
        )

    # Sort by time (required for interpolation)
    sort_idx = np.argsort(times)
    times_sorted = times[sort_idx]
    data_sorted = data[sort_idx]

    # Determine output time range
    if t_start is None:
        t_start = times_sorted[0]
    if t_end is None:
        t_end = times_sorted[-1]

    # Create regular time grid
    # Add small epsilon to ensure t_end is included
    t_regular = np.arange(t_start, t_end + 0.5 / sampling_freq, 1.0 / sampling_freq)

    # Create interpolator
    interpolator = interp1d(
        times_sorted,
        data_sorted,
        kind=kind,
        bounds_error=False,
        fill_value=fill_value,
        assume_sorted=True,
    )

    # Interpolate to regular grid
    data_regular = interpolator(t_regular)

    # Create ts object
    return ts(data_regular, start=t_start, sampling_freq=sampling_freq)
