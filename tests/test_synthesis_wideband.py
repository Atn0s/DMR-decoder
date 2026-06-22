# tests/test_synthesis_wideband.py
import os
import numpy as np
import pytest


def test_wideband_grid_creates_file(tmp_path):
    from utils.synthesis import synthesize_wideband_grid
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "wb_grid.rawiq")
    placements = [(-1_800_000.0, "dmr_1_78125.rawiq"),
                  (+1_800_000.0, "dmr_2_78125.rawiq")]
    result = synthesize_wideband_grid(placements, out, fs_out=5e6, dur_sec=1.0,
                                      data_dir="data")
    assert os.path.exists(result)
    # ~1s of 5MHz complex int16 = 1 * 5e6 * 2 int16
    assert os.path.getsize(result) > 5e6 * 2 * 2 * 0.8


def test_wideband_grid_places_energy_at_offsets(tmp_path):
    from utils.synthesis import synthesize_wideband_grid
    from core.dsp import read_rawiq
    import scipy.signal as signal
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "wb_grid.rawiq")
    synthesize_wideband_grid([(-1_800_000.0, "dmr_1_78125.rawiq")],
                             out, fs_out=5e6, dur_sec=1.0, data_dir="data")
    iq = read_rawiq(out).astype(np.complex64)
    f, psd = signal.welch(iq, fs=5e6, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f); psd = np.fft.fftshift(psd)
    peak_f = f[int(np.argmax(psd))]
    assert abs(peak_f - (-1_800_000.0)) < 100_000.0
