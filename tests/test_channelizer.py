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


def test_oversample_straddling_tone_in_two_subbands():
    fs = 8000.0
    N = 8
    ch = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=12, oversample=2)
    centers = ch.subband_centers()
    # place tone exactly on the boundary between two adjacent subbands
    boundary = (centers[3] + centers[4]) / 2.0
    x = _tone(boundary, fs, 16384)
    sub = ch.process(x)
    energies = np.mean(np.abs(sub) ** 2, axis=1)
    order = np.argsort(energies)[::-1]
    top_two = sorted(order[:2])
    # boundary tone shows up in BOTH adjacent subbands (not split/lost)
    assert top_two == [3, 4]
    assert energies[order[1]] > 0.25 * energies[order[0]]


def test_oversample_subband_rate_doubled():
    ch = PolyphaseChannelizer(8000.0, num_subbands=8, taps_per_phase=8, oversample=2)
    assert ch.subband_rate == pytest.approx(2000.0)


def _rand_iq(n, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


@pytest.mark.parametrize("oversample", [1, 2])
def test_streaming_matches_single_shot(oversample):
    fs = 8000.0
    N = 8
    x = _rand_iq(8192, seed=oversample)
    whole = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=10, oversample=oversample)
    out_whole = whole.process(x)
    streamed = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=10, oversample=oversample)
    parts = [streamed.process(x[:3000]), streamed.process(x[3000:5000]),
             streamed.process(x[5000:])]
    out_stream = np.concatenate(parts, axis=1)
    # carried state => split processing equals single-shot exactly (within float tol)
    assert out_stream.shape == out_whole.shape
    np.testing.assert_allclose(out_stream, out_whole, atol=1e-5)


def test_grid_coverage_every_subband_reachable():
    fs = 8000.0
    N = 8
    ch = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=12, oversample=1)
    centers = ch.subband_centers()
    hit = set()
    for c in centers:
        sub = ch.process(_tone(float(c), fs, 8192))
        hit.add(int(np.argmax(np.mean(np.abs(sub) ** 2, axis=1))))
        ch.reset()
    # every subband index is the winner for its own center tone -- no coverage hole
    assert hit == set(range(N))
