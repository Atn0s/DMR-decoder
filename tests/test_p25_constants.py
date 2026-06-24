import numpy as np

from p25.constants import (
    DUID_NAMES,
    FRAME_SYNC_BITS,
    FRAME_SYNC_HEX,
    FRAME_SYNC_SYMBOLS,
    dibits_to_symbols,
    symbols_to_dibits,
)


def test_frame_sync_constant_shape():
    assert FRAME_SYNC_HEX == "5575F5FF77FF"
    assert len(FRAME_SYNC_BITS) == 48
    assert FRAME_SYNC_SYMBOLS.shape == (24,)
    assert set(FRAME_SYNC_SYMBOLS.tolist()) <= {-3, -1, 1, 3}


def test_dibit_symbol_round_trip():
    bits = "00011011"
    symbols = dibits_to_symbols(bits)
    assert np.array_equal(symbols, np.array([1, 3, -1, -3]))
    assert symbols_to_dibits(symbols) == bits


def test_duid_names_include_metadata_units():
    assert DUID_NAMES[0x0] == "HDU"
    assert DUID_NAMES[0x5] == "LDU1"
    assert DUID_NAMES[0x7] == "TSBK"
    assert DUID_NAMES[0xA] == "LDU2"
    assert DUID_NAMES[0xF] == "TDULC"
