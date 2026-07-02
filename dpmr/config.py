from __future__ import annotations

from dataclasses import dataclass

from dpmr.constants import DPMR_DEV_NOMINAL, DPMR_FRONTEND_CUTOFF, FS_DEC, SPS, SYMBOL_RATE


@dataclass(frozen=True)
class DPMRConfig:
    target_sample_rate_hz: float = float(FS_DEC)
    symbol_rate_hz: float = float(SYMBOL_RATE)
    samples_per_symbol: int = int(SPS)
    frontend_cutoff_hz: float = DPMR_FRONTEND_CUTOFF
    frontend_taps: int = 151
    frontend_min_samples: int = 512
    frontend_psd_nperseg: int = 4096
    nominal_deviation_hz: float = DPMR_DEV_NOMINAL
    sync_threshold: float = 0.82
    sps_search_min: float = 19.0
    sps_search_max: float = 21.0
    sps_search_steps: int = 21
    frame_symbols: int = 384
    dedup_frame_bucket_samples: int = 3_840
    stable_color_min_repeats: int = 2


DEFAULT_DPMR_CONFIG = DPMRConfig()
