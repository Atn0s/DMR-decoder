import numpy as np

from p25.constants import FRAME_SYNC_SYMBOLS
from p25.sync import P25SyncCandidate, find_frame_sync


def test_find_frame_sync_returns_start_anchor():
    sps = 10
    fs_start = 300
    y = np.random.default_rng(123).normal(0.0, 0.03, 900)
    y[fs_start:fs_start + len(FRAME_SYNC_SYMBOLS) * sps] += np.repeat(
        FRAME_SYNC_SYMBOLS,
        sps,
    )

    hits = find_frame_sync(y, sps=sps, threshold=0.85)

    assert hits
    assert isinstance(hits[0], P25SyncCandidate)
    assert abs(hits[0].fs_start - fs_start) <= 1
    assert hits[0].polarity == 1.0
    assert hits[0].ncc >= 0.85


def test_find_frame_sync_detects_inverted_polarity():
    sps = 10
    fs_start = 200
    y = np.zeros(700)
    y[fs_start:fs_start + len(FRAME_SYNC_SYMBOLS) * sps] = -np.repeat(
        FRAME_SYNC_SYMBOLS,
        sps,
    )

    hits = find_frame_sync(y, sps=sps, threshold=0.85)

    assert hits
    assert abs(hits[0].fs_start - fs_start) <= 1
    assert hits[0].polarity == -1.0


def test_find_frame_sync_returns_empty_for_short_signal():
    y = np.zeros(100)
    assert find_frame_sync(y, sps=10) == []


def test_find_frame_sync_rejects_truncated_tail_frame_sync():
    sps = 10
    fs_start = 200
    y = np.zeros(360)
    ref = np.repeat(FRAME_SYNC_SYMBOLS, sps)
    y[fs_start:] = ref[:len(y) - fs_start]

    assert find_frame_sync(y, sps=sps, threshold=0.75) == []
