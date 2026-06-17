# tests/test_detector.py
import numpy as np
import pytest
from realtime.detector import (
    Detector, ChannelState, ChannelRecord,
    ACTIVE_THRESHOLD_DB, CLOSE_HYSTERESIS,
)


def _tone(fo_hz, fs, n, amp=1.0):
    t = np.arange(n) / fs
    return (amp * np.exp(1j * 2 * np.pi * fo_hz * t)).astype(np.complex64)


def _noise(n, amp=0.01):
    return (amp * (np.random.randn(n) + 1j * np.random.randn(n))).astype(np.complex64)


def test_quantize_freq_to_grid():
    det = Detector(sample_rate=2.5e6, channel_grid_hz=12500.0)
    assert det._quantize_freq(151000.0) == 150000.0
    assert det._quantize_freq(-299000.0) == -300000.0


def test_idle_to_active_on_energy():
    np.random.seed(0)
    det = Detector(sample_rate=2.5e6)
    win = _tone(150000.0, 2.5e6, 8192, amp=2.0) + _noise(8192)
    dispatched = det.process_window(win, window_id=0)
    fos = [d[1] for d in dispatched]
    assert 150000.0 in fos


def test_strategy_c_dispatches_every_active_window():
    np.random.seed(1)
    det = Detector(sample_rate=2.5e6)
    win = _tone(150000.0, 2.5e6, 8192, amp=2.0) + _noise(8192)
    d0 = det.process_window(win, 0)
    d1 = det.process_window(win, 1)
    d2 = det.process_window(win, 2)
    # Strategy C: same active channel dispatched on every window
    assert any(d[1] == 150000.0 for d in d0)
    assert any(d[1] == 150000.0 for d in d1)
    assert any(d[1] == 150000.0 for d in d2)


def test_silence_closes_after_hysteresis():
    np.random.seed(2)
    det = Detector(sample_rate=2.5e6)
    active = _tone(150000.0, 2.5e6, 8192, amp=2.0) + _noise(8192)
    silent = _noise(8192)
    det.process_window(active, 0)            # ACTIVE
    for w in range(1, CLOSE_HYSTERESIS):     # missed but within hysteresis
        det.process_window(silent, w)
        assert 150000.0 not in det.closed_channels()
    det.process_window(silent, CLOSE_HYSTERESIS)  # exceeds hysteresis
    assert 150000.0 in det.closed_channels()


def test_silent_spectrum_dispatches_nothing():
    np.random.seed(3)
    det = Detector(sample_rate=2.5e6)
    win = _noise(8192)
    assert det.process_window(win, 0) == []
