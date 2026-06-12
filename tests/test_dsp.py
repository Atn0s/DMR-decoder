import numpy as np
import os
from core.dsp import _interp, adaptive_slice_bits, read_rawiq, frontend, find_sync_positions


def test_interp_known():
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(_interp(arr, np.array([0.5, 1.5])), [0.5, 1.5])


def test_adaptive_slice_bits_pure_plus3():
    from bitarray import bitarray
    seg = np.full(132, 3.0)
    ba = adaptive_slice_bits(seg)
    assert len(ba) == 264
    assert ba[0:2] == bitarray('01')


def test_adaptive_slice_bits_all_levels():
    from bitarray import bitarray
    seg = np.array([-3.0, -1.0, 1.0, 3.0])
    ba = adaptive_slice_bits(seg)
    assert ba[0:2] == bitarray('11')   # -3
    assert ba[2:4] == bitarray('10')   # -1
    assert ba[4:6] == bitarray('00')   # +1
    assert ba[6:8] == bitarray('01')   # +3


def test_read_rawiq_shape():
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        return
    iq = read_rawiq(path)
    assert iq.dtype == complex
    assert len(iq) > 0


def test_frontend_output_length():
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        return
    import scipy.signal as sig
    iq_raw = read_rawiq(path)
    iq_dec = sig.resample_poly(iq_raw, 384, 625)
    y = frontend(iq_dec, fo=0.0, fs=48000.0)
    assert len(y) == len(iq_dec) - 1


def test_find_sync_positions_returns_list():
    y = np.zeros(50000)
    result = find_sync_positions(y)
    assert isinstance(result, list)


def test_find_sync_positions_on_real_data():
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        return
    import scipy.signal as sig
    iq = read_rawiq(path)
    iq_dec = sig.resample_poly(iq, 384, 625)
    y = frontend(iq_dec, fo=0.0)
    positions = find_sync_positions(y)
    assert len(positions) > 0
    for center, polarity, sync_type in positions:
        assert isinstance(center, int)
        assert polarity in (1.0, -1.0)
        assert sync_type in ("MS_VOICE", "BS_VOICE", "DATA_MS", "DATA_BS")
