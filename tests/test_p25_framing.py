from bitarray import bitarray

from p25.framing import frame_info_from_nid
from p25.nid import decode_nid


def _nid(nac: int, duid: int):
    bits = bitarray(endian="big")
    bits.extend(f"{nac:012b}{duid:04b}")
    bits.extend("0" * 48)
    return decode_nid(bits)


def test_frame_info_classifies_ldu1_as_voice_with_link_control():
    info = frame_info_from_nid(_nid(0x293, 0x5))

    assert info.nac == 0x293
    assert info.duid == 0x5
    assert info.duid_name == "LDU1"
    assert info.pdu_type == "P25_LDU1"
    assert info.category == "voice"
    assert info.is_voice is True
    assert info.is_control is False
    assert info.is_terminator is False
    assert info.has_link_control is True


def test_frame_info_classifies_tsbk_as_control():
    info = frame_info_from_nid(_nid(0x293, 0x7))

    assert info.duid_name == "TSBK"
    assert info.pdu_type == "P25_TSBK"
    assert info.category == "trunking_control"
    assert info.is_voice is False
    assert info.is_control is True
    assert info.is_terminator is False
    assert info.has_link_control is False


def test_frame_info_handles_unknown_duid_without_crashing():
    info = frame_info_from_nid(_nid(0x123, 0x2))

    assert info.duid_name == "UNKNOWN_0x2"
    assert info.pdu_type == "P25_UNKNOWN_0x2"
    assert info.category == "unknown"
    assert info.is_voice is False
    assert info.is_control is False
    assert info.is_terminator is False
    assert info.has_link_control is False
