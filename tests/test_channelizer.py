import numpy as np
import pytest
from realtime.channelizer import PolyphaseChannelizer


def _tone(f_hz, fs, n):
    t = np.arange(n) / fs
    return np.exp(1j * 2 * np.pi * f_hz * t).astype(np.complex64)


def test_subband_centers_ascending_grid():
    fs = 8000.0
    ch = PolyphaseChannelizer(fs, num_subbands=8, taps_per_phase=8, oversample=1)
    centers = ch.subband_centers()
    assert centers.shape == (8,)
    # spacing = fs/N = 1000 Hz, ascending, lowest = -fs/2
    np.testing.assert_allclose(np.diff(centers), 1000.0)
    assert centers.min() == pytest.approx(-4000.0)


def test_tone_lands_in_expected_subband():
    fs = 8000.0
    N = 8
    ch = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=12, oversample=1)
    # tone at +1000 Hz = center of one subband
    x = _tone(1000.0, fs, 8192)
    sub = ch.process(x)                       # (N, n_out)
    energies = np.mean(np.abs(sub) ** 2, axis=1)
    centers = ch.subband_centers()
    winner = int(np.argmax(energies))
    assert centers[winner] == pytest.approx(1000.0)
    # winner subband holds far more energy than the median subband
    assert energies[winner] > 10 * np.median(energies)


def test_subband_rate():
    ch = PolyphaseChannelizer(8000.0, num_subbands=8, taps_per_phase=8, oversample=1)
    assert ch.subband_rate == pytest.approx(1000.0)
