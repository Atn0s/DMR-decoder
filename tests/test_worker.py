import os
import numpy as np
import pytest
from realtime.worker import decode_window, _decimation_factors
from core.burst_type import Fs_dec
from core.dsp import read_rawiq


def test_decimation_factors_2_5mhz():
    up, down = _decimation_factors(2.5e6)
    # 2.5e6 * up/down should be ~48000; gcd reduction yields ~77Hz residual, < 100Hz
    assert abs(2.5e6 * up / down - Fs_dec) < 100.0


def test_decimation_factors_960khz():
    up, down = _decimation_factors(960000.0)
    assert abs(960000.0 * up / down - Fs_dec) < 50.0


def test_decode_window_returns_list_on_noise():
    np.random.seed(0)
    win = (np.random.randn(100000) + 1j * np.random.randn(100000)).astype(np.complex64)
    result = decode_window(win, fo_hz=0.0, window_id=0, source_sample_rate=2.5e6)
    assert isinstance(result, list)


def test_decode_window_tags_fo_and_window_id():
    # Use real wideband file if present; expect DMR1 at -300kHz
    path = "data/synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(path):
        pytest.skip("wideband test file not present")
    iq = read_rawiq(path).astype(np.complex64)
    # use full file (10s at 2.5MHz); 1s slice has too few bursts for sync detection
    win = iq
    result = decode_window(win, fo_hz=-300000.0, window_id=7, source_sample_rate=2.5e6)
    for pdu in result:
        assert pdu["_fo_hz"] == -300000.0
        assert pdu["_window_id"] == 7
    # should decode at least one PDU from the known DMR signal
    assert any(p["type"] in ("LC_HEADER", "LATE_ENTRY", "TERMINATOR", "CSBK")
               for p in result)
