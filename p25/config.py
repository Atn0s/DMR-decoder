from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class P25Config:
    target_sample_rate_hz: float = 48_000.0
    symbol_rate_hz: float = 4_800.0
    samples_per_symbol: int = 10
    frontend_cutoff_hz: float = 9_500.0
    frontend_taps: int = 151
    nominal_deviation_hz: float = 1_944.0
    sync_threshold: float = 0.62
    ldu_symbols: int = 864


DEFAULT_P25_CONFIG = P25Config()
