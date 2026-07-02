from __future__ import annotations

import numpy as np

from dmr.config import DEFAULT_DMR_CONFIG, DMRConfig
from dmr.constants import SPS, SYNC_TEMPLATES
from dmr.decoder import LateEntryCollector, decode_burst
from dmr.dsp import _interp, adaptive_slice_bits, find_sync_positions, recover_burst


BURST_STRIDE = 2880


def _lock_voice_phase(y: np.ndarray, anchor: int, polarity: float, sync_type: str) -> float:
    ref = SYNC_TEMPLATES[sync_type]
    levels = np.array([-3, -1, 1, 3])
    best = (1e18, 0.0)
    for ph in np.linspace(-8, 8, 65):
        start = anchor - (54 + 12) * SPS + ph
        pos = start + np.arange(132) * SPS
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue
        seg = polarity * _interp(y, pos)
        sy = seg[54:78]
        a, b = np.linalg.lstsq(np.vstack([sy, np.ones(24)]).T, ref, rcond=None)[0]
        segc = a * seg + b
        near = levels[np.argmin(np.abs(segc[:, None] - levels[None, :]), axis=1)]
        resid = np.mean((segc - near) ** 2)
        if resid < best[0]:
            best = (resid, ph)
    return best[1]


def _recover_stepped_burst(
    y: np.ndarray,
    anchor: int,
    j: int,
    ph: float,
    polarity: float,
    burst_stride_samples: int = BURST_STRIDE,
):
    start = anchor + burst_stride_samples * j - (54 + 12) * SPS + ph
    pos = start + np.arange(132) * SPS
    if pos[0] < 0 or pos[-1] >= len(y) - 1:
        return None
    seg = polarity * _interp(y, pos)
    return adaptive_slice_bits(seg)


def _decode_dmr_loop(y: np.ndarray, config: DMRConfig | None = None) -> list[dict]:
    config = config or DEFAULT_DMR_CONFIG
    positions = find_sync_positions(
        y,
        voice_threshold=config.sync_threshold_voice,
        data_threshold=config.sync_threshold_data,
        peak_distance_samples=config.sync_peak_distance_samples,
    )
    results = []
    seen_bursts: set[tuple] = set()

    for center, polarity, sync_type in positions:
        dedup_key = (round(center / 50), sync_type)
        if dedup_key in seen_bursts:
            continue
        seen_bursts.add(dedup_key)

        if "VOICE" in sync_type:
            ph = _lock_voice_phase(y, center, polarity, sync_type)
            collector = LateEntryCollector()
            for j in range(6):
                ba = _recover_stepped_burst(
                    y,
                    center,
                    j,
                    ph,
                    polarity,
                    burst_stride_samples=config.voice_burst_stride_samples,
                )
                if ba is None:
                    break
                pdu = collector.feed(ba, sync_type)
                if pdu is not None:
                    results.append(dict(pdu))
                    break
        else:
            symbols = recover_burst(y, center, polarity, sync_type)
            if symbols is None:
                continue
            pdu = decode_burst(symbols, sync_type)
            if pdu is not None:
                results.append(dict(pdu))

    return results


def decode(y: np.ndarray, config: DMRConfig | None = None) -> list[dict]:
    pdus = _decode_dmr_loop(y, config)
    for pdu in pdus:
        pdu.setdefault("protocol", "DMR")
    return pdus
