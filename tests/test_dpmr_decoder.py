import os

import numpy as np
import pytest

from core.dsp import read_rawiq
from dpmr.cch import CCHRecord, air_interface_id_to_str, crc7, descramble
from dpmr.decoder import decode, filter_stable_pdus
from dpmr.dsp import find_dpmr_sync, frontend_dpmr
from dpmr.constants import DIBIT_TO_LEVEL, DPMR_FRAME_SYMBOLS, FS1_SYMBOLS, FS2_SYMBOLS, SPS
from dpmr.session import DPMRSessionAssembler


def _symbols_to_levels(symbols: np.ndarray) -> np.ndarray:
    return np.array([DIBIT_TO_LEVEL[int(symbol)] for symbol in symbols], dtype=float)


def _cch_record(frame_number: int, id_half: int, *, crc_ok: bool = True) -> CCHRecord:
    return CCHRecord(
        frame_number=frame_number,
        id_half=id_half,
        communication_mode=0,
        version=0,
        comms_format=0,
        emergency_priority=0,
        reserved=0,
        slow_data=0,
        crc_value=0,
        crc_computed=0,
        crc_ok=crc_ok,
        hamming_ok=True,
        hamming_blocks_ok=(True,) * 6,
        corrected_bits=0,
        bits=(0,) * 48,
    )


def _int_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _hamming_12_8_encode(data: list[int]) -> list[int]:
    rows = (
        (1, 0, 1, 0, 1, 1, 0, 0),
        (1, 1, 0, 1, 0, 1, 1, 0),
        (1, 1, 1, 0, 1, 0, 1, 1),
        (0, 1, 0, 1, 1, 0, 0, 1),
    )
    parity = [sum(bit * row[idx] for idx, bit in enumerate(data)) & 1 for row in rows]
    return data + parity


def _encode_cch(frame_number: int, id_half: int) -> list[int]:
    data: list[int] = []
    data.extend(_int_bits(frame_number, 2))
    data.extend(_int_bits(id_half, 12))
    data.extend(_int_bits(0, 3))
    data.extend(_int_bits(0, 2))
    data.extend(_int_bits(0, 2))
    data.extend([0, 0])
    data.extend(_int_bits(0, 18))
    data.extend(_int_bits(crc7(data), 7))

    deinterleaved: list[int] = []
    for idx in range(6):
        deinterleaved.extend(_hamming_12_8_encode(data[idx * 8:(idx + 1) * 8]))

    scrambled = [0] * 72
    for col in range(6):
        for row in range(12):
            scrambled[row * 6 + col] = deinterleaved[col * 12 + row]
    return descramble(scrambled)


def _bits_to_symbols(bits: list[int]) -> np.ndarray:
    return np.array([(bits[i] << 1) | bits[i + 1] for i in range(0, len(bits), 2)], dtype=float)


def test_decode_dpmr_sample_extracts_voice_metadata():
    path = "data/dpmr_1_48000.rawiq"
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")

    y = frontend_dpmr(read_rawiq(path))
    results = decode(y)

    assert results
    assert any(pdu["protocol"] == "dPMR" for pdu in results)
    assert any(pdu["extra"]["color_code"] == 2 for pdu in results)
    assert all(pdu["extra"]["polarity_inverted"] is True for pdu in results)
    assert all(pdu["extra"]["quality"]["crc_ok_count"] == 2 for pdu in results)
    assert any(pdu["src"] == "3939*5*" for pdu in results)
    assert any(pdu["dst"] == "3939*5*" for pdu in results)
    assert all("quality" in pdu["extra"] for pdu in results)


def test_decode_synthetic_dpmr_header_frame_extracts_cch_pair():
    cch0 = _bits_to_symbols(_encode_cch(0, 0x123))
    cch1 = _bits_to_symbols(_encode_cch(1, 0x456))
    payload = np.zeros(DPMR_FRAME_SYMBOLS - len(FS1_SYMBOLS), dtype=float)
    payload[:len(cch0)] = cch0
    payload[len(cch0):len(cch0) + len(cch1)] = cch1
    symbols = np.concatenate([FS1_SYMBOLS, payload])
    y = np.concatenate([np.zeros(200), np.repeat(_symbols_to_levels(symbols), SPS), np.zeros(200)])

    results = decode(y, sync_threshold=0.80)
    headers = [pdu for pdu in results if pdu["type"] == "DPMR_HEADER"]

    assert headers
    header = headers[0]
    assert header["dst"] == air_interface_id_to_str(0x123456)
    assert header["extra"]["sync_type"] == "FS1"
    assert header["extra"]["quality"]["crc_ok_count"] >= 2
    assert {0, 1}.issubset(set(header["extra"]["frame_numbers"]))


def test_find_dpmr_sync_classifies_header_and_voice_syncs():
    y = np.concatenate(
        [
            np.zeros(160),
            np.repeat(_symbols_to_levels(FS1_SYMBOLS), SPS),
            np.zeros(520),
            np.repeat(_symbols_to_levels(FS2_SYMBOLS), SPS),
            np.zeros(160),
        ]
    )

    syncs = find_dpmr_sync(y, threshold=0.80)

    assert [sync.sync_type for sync in syncs] == ["FS1", "FS2"]


def test_session_assembler_accumulates_id_halves_and_prefers_crc():
    session = DPMRSessionAssembler()
    stale = _cch_record(2, 0x001, crc_ok=False)
    trusted = _cch_record(2, 0x123, crc_ok=True)
    tail = _cch_record(3, 0x456, crc_ok=True)

    session.feed(stale, None)
    session.feed(trusted, None)
    src, dst, part = session.feed(tail, None)

    assert part == "src"
    assert src == air_interface_id_to_str(0x123456)
    assert dst == ""


def test_filter_stable_pdus_keeps_repeated_color_code():
    pdus = [
        {"protocol": "dPMR", "extra": {"color_code": 2}},
        {"protocol": "dPMR", "extra": {"color_code": 5}},
        {"protocol": "dPMR", "extra": {"color_code": 2}},
        {"protocol": "DMR", "extra": {}},
    ]

    filtered = filter_stable_pdus(pdus)

    assert [p.get("protocol") for p in filtered] == ["dPMR", "dPMR", "DMR"]
    assert all(
        p.get("protocol") != "dPMR" or p["extra"]["stable_color_code"] == 2
        for p in filtered
    )


def test_filter_stable_pdus_prefers_high_quality_when_available():
    pdus = [
        {"protocol": "dPMR", "extra": {"color_code": 2, "quality": {"front_end_confidence": "low"}}},
        {"protocol": "dPMR", "extra": {"color_code": 2, "quality": {"front_end_confidence": "high"}}},
        {"protocol": "dPMR", "extra": {"color_code": 5, "quality": {"front_end_confidence": "high"}}},
    ]

    filtered = filter_stable_pdus(pdus)

    assert len(filtered) == 1
    assert filtered[0]["extra"]["color_code"] == 2
    assert filtered[0]["extra"]["quality"]["front_end_confidence"] == "high"


def test_filter_stable_pdus_keeps_repeated_color_over_single_high_quality():
    pdus = [
        {"protocol": "dPMR", "extra": {"color_code": 2, "quality": {"front_end_confidence": "diagnostic"}}},
        {"protocol": "dPMR", "extra": {"color_code": 2, "quality": {"front_end_confidence": "diagnostic"}}},
        {"protocol": "dPMR", "extra": {"color_code": 4, "quality": {"front_end_confidence": "high"}}},
    ]

    filtered = filter_stable_pdus(pdus)

    assert len(filtered) == 2
    assert {pdu["extra"]["color_code"] for pdu in filtered} == {2}


def test_dpmr_sample_exposes_only_high_confidence_ids():
    path = "data/dpmr_1_48000.rawiq"
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")

    results = decode(frontend_dpmr(read_rawiq(path)))
    exposed = [pdu for pdu in results if pdu["src"] or pdu["dst"]]

    assert exposed
    assert all(
        pdu["extra"]["quality"]["front_end_confidence"] == "high"
        for pdu in exposed
    )
