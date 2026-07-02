from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RadioConfig:
    target_sample_rate_hz: float = 48_000.0
    sample_rate_tolerance_hz: float = 1.0
    wideband_sample_rate_hz: float = 2_500_000.0
    wideband_resample_up: int = 12
    wideband_resample_down: int = 625
    narrowband_max_sample_rate_hz: float = 200_000.0
    psd_peak_threshold_db: float = 15.0
    psd_nperseg: int = 4096
    psd_peak_min_distance_bins: int = 20


@dataclass(frozen=True)
class RealtimeConfig:
    active_threshold_db: float = 15.0
    channel_grid_hz: float = 12_500.0
    close_hysteresis_windows: int = 3
    call_timeout_windows: int = 5
    fo_bucket_hz: float = 5_000.0


DEFAULT_RADIO_CONFIG = RadioConfig()
DEFAULT_REALTIME_CONFIG = RealtimeConfig()
