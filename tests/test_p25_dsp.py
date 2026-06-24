import numpy as np

from p25.constants import FRAME_SYNC_SYMBOLS
from p25.dsp import recover_symbols_from_fs, slice_symbols_to_bits
from p25.sync import P25SyncCandidate


def test_recover_symbols_from_fs_uses_start_anchor():
    sps = 10
    fs_start = 80
    payload = np.array([1, 3, -1, -3, 1, -1], dtype=float)
    symbols = np.concatenate([FRAME_SYNC_SYMBOLS, payload])
    y = np.zeros(600)
    y[fs_start:fs_start + len(symbols) * sps] = np.repeat(symbols * 1.7 + 0.4, sps)
    candidate = P25SyncCandidate(fs_start=fs_start, polarity=1.0, ncc=0.99)

    recovered = recover_symbols_from_fs(y, candidate, symbol_count=len(symbols), sps=sps)

    assert recovered is not None
    assert np.array_equal(np.round(recovered[-len(payload):]).astype(int), payload.astype(int))


def test_recover_symbols_from_fs_handles_inverted_signal():
    sps = 10
    fs_start = 80
    payload = np.array([3, 1, -1, -3], dtype=float)
    symbols = np.concatenate([FRAME_SYNC_SYMBOLS, payload])
    y = np.zeros(600)
    y[fs_start:fs_start + len(symbols) * sps] = -np.repeat(symbols, sps)
    candidate = P25SyncCandidate(fs_start=fs_start, polarity=-1.0, ncc=1.0)

    recovered = recover_symbols_from_fs(y, candidate, symbol_count=len(symbols), sps=sps)

    assert recovered is not None
    assert np.array_equal(np.round(recovered[-len(payload):]).astype(int), payload.astype(int))


def test_recover_symbols_from_fs_returns_none_for_zero_symbols():
    candidate = P25SyncCandidate(fs_start=80, polarity=1.0, ncc=0.99)

    recovered = recover_symbols_from_fs(
        np.zeros(600),
        candidate,
        symbol_count=0,
        sps=10,
    )

    assert recovered is None


def test_recover_symbols_from_fs_returns_none_for_shorter_than_frame_sync():
    candidate = P25SyncCandidate(fs_start=80, polarity=1.0, ncc=0.99)

    recovered = recover_symbols_from_fs(
        np.zeros(600),
        candidate,
        symbol_count=len(FRAME_SYNC_SYMBOLS) - 1,
        sps=10,
    )

    assert recovered is None


def test_slice_symbols_to_bits_uses_p25_dibit_mapping():
    symbols = np.array([1, 3, -1, -3], dtype=float)
    assert slice_symbols_to_bits(symbols).to01() == "00011011"
