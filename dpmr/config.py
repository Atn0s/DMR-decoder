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
    sync_max_symbol_errors: int = 0
    sync_min_distance_samples: int = 1_200
    sync_dedup_window_symbols: int = 3
    sync_error_phase_min: float = -12.0
    sync_error_phase_max: float = 12.0
    sync_error_phase_steps: int = 13
    sps_search_min: float = 20.0
    sps_search_max: float = 20.0
    sps_search_steps: int = 1
    phase_search_min: float = -12.0
    phase_search_max: float = 12.0
    phase_search_steps: int = 25
    sample_windows: tuple[int, ...] = (0,)
    decision_ambiguous_threshold: float = 0.35
    frame_symbols: int = 384
    header_sync_candidate_limit: int = 50
    header_symbol_candidate_limit: int = 16
    voice_sync_candidate_limit: int = 100
    voice_symbol_candidate_limit: int = 8
    dedup_frame_bucket_samples: int = 3_840
    stable_color_min_repeats: int = 2


DEFAULT_DPMR_CONFIG = DPMRConfig()
