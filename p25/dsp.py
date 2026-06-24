from __future__ import annotations

import numpy as np
from bitarray import bitarray

from p25.constants import FRAME_SYNC_SYMBOLS, SYMBOL_TO_DIBIT
from p25.sync import P25SyncCandidate


def _interp(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    i = np.floor(pos).astype(int)
    fr = pos - i
    i = np.clip(i, 0, len(arr) - 2)
    return arr[i] * (1 - fr) + arr[i + 1] * fr


def recover_symbols_from_fs(
    y: np.ndarray,
    candidate: P25SyncCandidate,
    symbol_count: int,
    sps: int = 10,
    phase_search: np.ndarray | None = None,
) -> np.ndarray | None:
    """Recover symbols forward from P25 FS start.

    P25 uses a frame-start sync anchor. This function intentionally samples
    from `candidate.fs_start` forward; it does not apply DMR's center-sync
    burst offset.
    """
    if symbol_count < len(FRAME_SYNC_SYMBOLS):
        return None

    if phase_search is None:
        phase_search = np.linspace(-4, 4, 33)

    levels = np.array([-3, -1, 1, 3])
    best: tuple[float, np.ndarray | None] = (1e18, None)
    for phase in phase_search:
        pos = candidate.fs_start + phase + np.arange(symbol_count) * sps
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue

        seg = candidate.polarity * _interp(y, pos)
        fs_seg = seg[:len(FRAME_SYNC_SYMBOLS)]
        a, b = np.linalg.lstsq(
            np.vstack([fs_seg, np.ones(len(fs_seg))]).T,
            FRAME_SYNC_SYMBOLS,
            rcond=None,
        )[0]
        calibrated = a * seg + b
        nearest = levels[
            np.argmin(np.abs(calibrated[:, None] - levels[None, :]), axis=1)
        ]
        resid = float(np.mean((calibrated[:len(FRAME_SYNC_SYMBOLS)] - FRAME_SYNC_SYMBOLS) ** 2))
        resid += 0.05 * float(np.mean((calibrated - nearest) ** 2))
        if resid < best[0]:
            best = (resid, calibrated)
    return best[1]


def slice_symbols_to_bits(symbols: np.ndarray) -> bitarray:
    levels = np.array([-3, -1, 1, 3])
    nearest = levels[np.argmin(np.abs(symbols[:, None] - levels[None, :]), axis=1)]
    bits = bitarray()
    bits.extend("".join(SYMBOL_TO_DIBIT[int(v)] for v in nearest))
    return bits
