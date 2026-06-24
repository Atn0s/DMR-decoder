from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

from p25.constants import FRAME_SYNC_SYMBOLS


@dataclass(frozen=True)
class P25SyncCandidate:
    fs_start: int
    polarity: float
    ncc: float


def find_frame_sync(
    y: np.ndarray,
    sps: int = 10,
    threshold: float = 0.62,
    min_distance_symbols: int = 120,
) -> list[P25SyncCandidate]:
    """Find P25 frame sync and return frame-start anchors.

    Unlike DMR, P25 FS is at the start of a data unit. The returned sample is
    the FS start, not the FS center.
    """
    ref = np.repeat(FRAME_SYNC_SYMBOLS, sps)
    if len(y) < len(ref):
        return []

    c = signal.correlate(y, ref, mode="same")
    e = np.convolve(y ** 2, np.ones(len(ref)), mode="same")
    e = np.where(e <= 0, 1e-9, e)
    ncc = c / np.sqrt(e * np.sum(ref ** 2))
    distance = max(1, min_distance_symbols * sps)

    hits: list[P25SyncCandidate] = []
    pos_peaks, pos_props = signal.find_peaks(ncc, height=threshold, distance=distance)
    neg_peaks, neg_props = signal.find_peaks(-ncc, height=threshold, distance=distance)

    half = len(ref) // 2
    for peak, height in zip(pos_peaks, pos_props["peak_heights"]):
        hits.append(P25SyncCandidate(int(peak - half), 1.0, float(height)))
    for peak, height in zip(neg_peaks, neg_props["peak_heights"]):
        hits.append(P25SyncCandidate(int(peak - half), -1.0, float(height)))

    hits = [h for h in hits if 0 <= h.fs_start and h.fs_start + len(ref) <= len(y)]
    hits.sort(key=lambda h: h.fs_start)
    return hits
