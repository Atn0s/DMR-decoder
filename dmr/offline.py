from __future__ import annotations

from common.io import detect_sample_rate, read_rawiq
from dmr.engine import (
    BURST_STRIDE,
    _decode_dmr_loop,
    _lock_voice_phase,
    _recover_stepped_burst,
    decode,
)


def scan_file(path: str, freq_list: list[float] | None = None) -> list[dict]:
    """Compatibility wrapper for old DMR-only offline callers."""
    from radio.pipeline import scan_iq

    iq = read_rawiq(path)
    fs = detect_sample_rate(path)
    return scan_iq(iq, fs, freq_list=freq_list, protocol_names={"dmr"})
