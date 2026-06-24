import numpy as np

from p25.constants import FRAME_SYNC_SYMBOLS, dibits_to_symbols
from p25.decoder import decode


def nid_symbols(nac: int, duid: int) -> np.ndarray:
    bits = f"{nac:012b}{duid:04b}" + "0" * 48
    return dibits_to_symbols(bits)


def test_decode_emits_p25_nid_pdu_from_synthetic_y():
    sps = 10
    fs_start = 120
    symbols = np.concatenate([FRAME_SYNC_SYMBOLS, nid_symbols(0x293, 0x7)])
    y = np.random.default_rng(456).normal(0.0, 0.02, 900)
    y[fs_start:fs_start + len(symbols) * sps] += np.repeat(symbols, sps)

    pdus = decode(y, sps=sps, sync_threshold=0.85)

    assert len(pdus) == 1
    pdu = pdus[0]
    assert pdu["protocol"] == "P25"
    assert pdu["type"] == "P25_NID"
    assert pdu["src"] == 0
    assert pdu["dst"] == 0
    assert pdu["flco"] == "TSBK"
    assert pdu["fid"] == ""
    assert pdu["extra"]["nac"] == 0x293
    assert pdu["extra"]["duid"] == 0x7
    assert pdu["extra"]["duid_name"] == "TSBK"
    assert "raw_bits" in pdu
    assert isinstance(pdu["raw_bits"], bytes)


def test_decode_returns_empty_when_no_frame_sync():
    y = np.zeros(1000)
    assert decode(y, sps=10, sync_threshold=0.85) == []
