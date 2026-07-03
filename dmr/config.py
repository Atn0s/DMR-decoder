from __future__ import annotations

from dataclasses import dataclass

from dmr.constants import (
    DEV_NOMINAL,
    Fs_dec,
    Fs_wide,
    NCC_THRESHOLD_DATA,
    NCC_THRESHOLD_VOICE,
    SPS,
)


@dataclass(frozen=True)
class DMRConfig:
    target_sample_rate_hz: float = Fs_dec
    wideband_sample_rate_hz: float = Fs_wide
    symbol_rate_hz: float = 4_800.0
    samples_per_symbol: int = SPS
    frontend_cutoff_hz: float = 9_500.0
    frontend_taps: int = 151
    frontend_min_samples: int = 512
    frontend_psd_nperseg: int = 4096
    nominal_deviation_hz: float = DEV_NOMINAL
    sync_threshold_voice: float = NCC_THRESHOLD_VOICE
    sync_threshold_data: float = NCC_THRESHOLD_DATA
    sync_peak_distance_samples: int = 800
    voice_burst_stride_samples: int = 2_880
    voice_burst_count: int = 6
    burst_dedup_window_samples: int = 50
    dedup_frequency_bucket_hz: float = 5_000.0


DEFAULT_DMR_CONFIG = DMRConfig()
