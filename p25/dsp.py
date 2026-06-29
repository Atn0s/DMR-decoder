from __future__ import annotations

import numpy as np
from bitarray import bitarray

from p25.constants import (
    ES_HEXBIT_POSITIONS,
    FRAME_SYNC_SYMBOLS,
    HDU_DATA_HEXBIT_POSITIONS,
    HDU_GOLAY_PARITY_POSITIONS,
    LC_HEXBIT_POSITIONS,
    LDU_SYMBOLS,
    FS_BITS,
    NID_AIR_SYMBOLS,
    NID_BITS,
    NID_STATUS_SYMBOL_OFFSET,
    SYMBOL_TO_DIBIT,
)
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


def extract_nid_bits(fs_nid_bits: bitarray) -> bitarray:
    """Return the logical 64-bit P25 NID, skipping the in-header status dibit."""
    required = FS_BITS + NID_AIR_SYMBOLS * 2
    if len(fs_nid_bits) < required:
        raise ValueError("P25 FS+NID bits are shorter than the 57-symbol header")

    start = FS_BITS
    status = start + NID_STATUS_SYMBOL_OFFSET * 2
    out = bitarray(endian="big")
    out.extend(fs_nid_bits[start:status])
    out.extend(fs_nid_bits[status + 2:start + NID_AIR_SYMBOLS * 2])
    if len(out) != NID_BITS:
        raise ValueError("P25 logical NID extraction did not produce 64 bits")
    return out


def recover_full_frame(
    y: np.ndarray,
    candidate: P25SyncCandidate,
    sps: int = 10,
) -> np.ndarray | None:
    return recover_symbols_from_fs(y, candidate, symbol_count=LDU_SYMBOLS, sps=sps)


def deinterleave_lc(frame_bits: bitarray) -> list[bitarray]:
    if len(frame_bits) <= max(LC_HEXBIT_POSITIONS):
        raise ValueError("P25 LDU frame bits are shorter than LC layout")
    picked = bitarray(endian="big")
    picked.extend(frame_bits[pos] for pos in LC_HEXBIT_POSITIONS)
    return [picked[i * 10:(i + 1) * 10] for i in range(24)]


def deinterleave_es(frame_bits: bitarray) -> list[bitarray]:
    if len(frame_bits) <= max(ES_HEXBIT_POSITIONS):
        raise ValueError("P25 LDU frame bits are shorter than ES layout")
    picked = bitarray(endian="big")
    picked.extend(frame_bits[pos] for pos in ES_HEXBIT_POSITIONS)
    return [picked[i * 10:(i + 1) * 10] for i in range(24)]


def deinterleave_hdu(frame_bits: bitarray) -> list[tuple[bitarray, bitarray]]:
    max_pos = max(max(HDU_DATA_HEXBIT_POSITIONS), max(HDU_GOLAY_PARITY_POSITIONS))
    if len(frame_bits) <= max_pos:
        raise ValueError("P25 HDU frame bits are shorter than HDU layout")
    data = bitarray(endian="big")
    data.extend(frame_bits[pos] for pos in HDU_DATA_HEXBIT_POSITIONS)
    parity = bitarray(endian="big")
    parity.extend(frame_bits[pos] for pos in HDU_GOLAY_PARITY_POSITIONS)
    return [
        (
            data[i * 6:(i + 1) * 6],
            parity[i * 12:(i + 1) * 12],
        )
        for i in range(36)
    ]
