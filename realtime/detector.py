from enum import IntEnum
from dataclasses import dataclass
import numpy as np
import scipy.signal as signal

ACTIVE_THRESHOLD_DB = 15   # dB above median noise floor (matches scanner.PSD_PEAK_THRESHOLD_DB)
CLOSE_HYSTERESIS = 3       # consecutive silent windows before a call is closed


class ChannelState(IntEnum):
    IDLE = 0
    ACTIVE = 1
    TRACKING = 2
    CLOSING = 3


@dataclass
class ChannelRecord:
    fo_hz: float
    state: ChannelState
    last_active_window: int
    missed_windows: int


class Detector:
    """Per-window energy detection with a frequency-indexed channel state table.
    Strategy C: every ACTIVE/TRACKING channel is dispatched on every window
    (voice frames are accumulated in time order by the aggregator).
    The dispatched IQ slice is the full wideband window — DDC/decimation happen
    in the worker, not here."""

    def __init__(self, sample_rate: float, channel_grid_hz: float = 12500.0,
                 threshold_db: float = ACTIVE_THRESHOLD_DB,
                 close_hysteresis: int = CLOSE_HYSTERESIS):
        self.sample_rate = sample_rate
        self.channel_grid_hz = channel_grid_hz
        self.threshold_db = threshold_db
        self.close_hysteresis = close_hysteresis
        self._channels: dict[float, ChannelRecord] = {}
        self._just_closed: list[float] = []

    def _quantize_freq(self, f_hz: float) -> float:
        return round(f_hz / self.channel_grid_hz) * self.channel_grid_hz

    def _detect_active_freqs(self, window_iq: np.ndarray) -> set[float]:
        f, psd = signal.welch(window_iq, fs=self.sample_rate,
                              nperseg=min(4096, len(window_iq)),
                              return_onesided=False)
        f = np.fft.fftshift(f)
        psd = np.fft.fftshift(psd)
        psd_db = 10 * np.log10(psd + 1e-12)
        nf = np.median(psd_db)
        peaks, _ = signal.find_peaks(psd_db, height=nf + self.threshold_db, distance=20)
        return {self._quantize_freq(float(f[p])) for p in peaks}

    def process_window(self, window_iq: np.ndarray, window_id: int
                       ) -> list[tuple[np.ndarray, float, int]]:
        self._just_closed = []
        active_freqs = self._detect_active_freqs(window_iq)

        # Update existing channels and open new ones
        for fo in active_freqs:
            rec = self._channels.get(fo)
            if rec is None:
                self._channels[fo] = ChannelRecord(
                    fo_hz=fo, state=ChannelState.ACTIVE,
                    last_active_window=window_id, missed_windows=0)
            else:
                rec.state = ChannelState.TRACKING
                rec.last_active_window = window_id
                rec.missed_windows = 0

        # Age out channels not seen this window
        for fo, rec in list(self._channels.items()):
            if fo not in active_freqs:
                rec.missed_windows += 1
                if rec.missed_windows >= self.close_hysteresis:
                    rec.state = ChannelState.CLOSING
                    self._just_closed.append(fo)
                    del self._channels[fo]

        # Strategy C: dispatch every channel still in the table (ACTIVE/TRACKING).
        # Channels aged out above have already been removed from self._channels,
        # so iterating here naturally excludes CLOSING channels.
        return [(window_iq, fo, window_id) for fo in sorted(self._channels)]

    def closed_channels(self) -> list[float]:
        return list(self._just_closed)
