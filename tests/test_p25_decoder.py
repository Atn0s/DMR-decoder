import numpy as np
from bitarray import bitarray

from p25.constants import FRAME_SYNC_SYMBOLS, dibits_to_symbols
from p25.decoder import decode
from p25.fec import bch_63_16_encode


def nid_symbols(nac: int, duid: int) -> np.ndarray:
    info = bitarray(f"{nac:012b}{duid:04b}")
    logical = bch_63_16_encode(info)
    air = logical[:22] + bitarray("00") + logical[22:]
    return dibits_to_symbols(air.to01())


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
    assert pdu["extra"]["pdu_type"] == "P25_TSBK"
    assert pdu["extra"]["frame_category"] == "trunking_control"
    assert pdu["extra"]["is_voice"] is False
    assert pdu["extra"]["is_control"] is True
    assert pdu["extra"]["is_terminator"] is False
    assert pdu["extra"]["has_link_control"] is False
    assert "raw_bits" in pdu
    assert isinstance(pdu["raw_bits"], bytes)


def test_decode_returns_empty_when_no_frame_sync():
    y = np.zeros(1000)
    assert decode(y, sps=10, sync_threshold=0.85) == []
