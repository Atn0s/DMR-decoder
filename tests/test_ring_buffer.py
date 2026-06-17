import numpy as np
import pytest
from realtime.ring_buffer import RingBuffer


def test_write_then_read_window():
    rb = RingBuffer(capacity_samples=1000)
    data = np.arange(500, dtype=np.complex64)
    dropped = rb.write(data)
    assert dropped == 0
    win = rb.read_window(window_samples=300, step_samples=200)
    assert win is not None
    assert len(win) == 300
    np.testing.assert_array_equal(win, data[:300])


def test_read_window_overlap_preserved():
    rb = RingBuffer(capacity_samples=1000)
    rb.write(np.arange(500, dtype=np.complex64))
    win1 = rb.read_window(window_samples=300, step_samples=200)
    win2 = rb.read_window(window_samples=300, step_samples=200)
    # step=200 so win2 starts at sample 200; overlap is samples [200,300)
    np.testing.assert_array_equal(win1[200:300], win2[0:100])


def test_read_window_insufficient_returns_none():
    rb = RingBuffer(capacity_samples=1000)
    rb.write(np.arange(100, dtype=np.complex64))
    assert rb.read_window(window_samples=300, step_samples=200) is None


def test_overflow_counts_dropped_samples():
    rb = RingBuffer(capacity_samples=100)
    dropped = rb.write(np.arange(150, dtype=np.complex64))
    assert dropped == 50
    assert rb.overflow_count == 50


def test_overflow_keeps_newest_data():
    rb = RingBuffer(capacity_samples=100)
    rb.write(np.arange(150, dtype=np.complex64))
    # Oldest 50 dropped; buffer holds samples 50..149
    win = rb.read_window(window_samples=100, step_samples=100)
    assert win is not None
    np.testing.assert_array_equal(win, np.arange(50, 150, dtype=np.complex64))


def test_available_tracks_unread():
    rb = RingBuffer(capacity_samples=1000)
    rb.write(np.arange(500, dtype=np.complex64))
    assert rb.available() == 500
    rb.read_window(window_samples=300, step_samples=200)
    assert rb.available() == 300
