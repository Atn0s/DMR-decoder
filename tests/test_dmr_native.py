from bitarray import bitarray
from bitarray.util import int2ba

from dmr.constants import VLC_RS_MASK
from dmr.fec import rs_12_9_4_check, rs_12_9_4_generate
from dmr.layer2 import LCSS, parse_csbk, parse_embedded_signalling, parse_full_link_control


def _bits_from_bytes(data: bytes) -> bitarray:
    bits = bitarray(endian="big")
    bits.frombytes(data)
    return bits


def test_rs_12_9_4_generate_and_check_with_vlc_mask():
    payload = bytes.fromhex("000000000001000001")
    protected = rs_12_9_4_generate(payload, VLC_RS_MASK)

    assert rs_12_9_4_check(protected, VLC_RS_MASK)

    damaged = bytearray(protected)
    damaged[0] ^= 0x01
    assert not rs_12_9_4_check(bytes(damaged), VLC_RS_MASK)


def test_parse_full_link_control_group_voice_fields():
    lc = _bits_from_bytes(bytes.fromhex("000000000001000002000000"))

    parsed = parse_full_link_control(lc)

    assert parsed.flco_name == "GroupVoiceChannelUser"
    assert parsed.fid_name == "StandardizedFID"
    assert parsed.group_address == 1
    assert parsed.source_address == 2


def test_parse_csbk_unit_to_unit_request_addresses():
    bits = bitarray("0" * 96, endian="big")
    bits[0] = 1
    bits[2:8] = int2ba(0x04, length=6, endian="big")
    bits[32:56] = int2ba(1234, length=24, endian="big")
    bits[56:80] = int2ba(5678, length=24, endian="big")

    parsed = parse_csbk(bits)

    assert parsed.last_block is True
    assert parsed.csbko_name == "UnitToUnitVoiceServiceRequest"
    assert parsed.target_address == 1234
    assert parsed.source_address == 5678


def test_parse_embedded_signalling_lcss_bits():
    bits = bitarray("0" * 16, endian="big")
    bits[5:7] = int2ba(LCSS.FirstFragmentLC.value, length=2, endian="big")

    parsed = parse_embedded_signalling(bits)

    assert parsed.link_control_start_stop == LCSS.FirstFragmentLC
    assert parsed.emb_parity_ok is False

