from bitarray import bitarray
from bitarray.util import int2ba

from dmr.constants import VLC_RS_MASK
from dmr.fec import rs_12_9_4_check, rs_12_9_4_generate
from dmr.layer2 import (
    LCSS,
    parse_csbk,
    parse_embedded_signalling,
    parse_full_link_control,
    parse_service_options,
)
from dmr.plugin import format_pdu
from dmr.session import DMRSessionAssembler


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
    lc = _bits_from_bytes(bytes.fromhex("00009b000001000002000000"))

    parsed = parse_full_link_control(lc)

    assert parsed.flco_name == "GroupVoiceChannelUser"
    assert parsed.fid_name == "StandardizedFID"
    assert parsed.service_options_value == 0x9B
    assert parsed.service_options.emergency is True
    assert parsed.service_options.privacy is False
    assert parsed.service_options.broadcast is True
    assert parsed.service_options.open_voice_call_mode is False
    assert parsed.service_options.priority == 3
    assert parsed.group_address == 1
    assert parsed.source_address == 2
    assert parsed.to_extra()["call_type"] == "group"


def test_parse_service_options_bits():
    svc = parse_service_options(bitarray("11010110", endian="big"))

    assert svc.value == 0xD6
    assert svc.emergency is True
    assert svc.privacy is True
    assert svc.reserved == 1
    assert svc.broadcast is False
    assert svc.open_voice_call_mode is True
    assert svc.priority == 2


def test_parse_csbk_unit_to_unit_request_addresses():
    bits = bitarray("0" * 96, endian="big")
    bits[0] = 1
    bits[2:8] = int2ba(0x04, length=6, endian="big")
    bits[16:24] = int2ba(0x81, length=8, endian="big")
    bits[32:56] = int2ba(1234, length=24, endian="big")
    bits[56:80] = int2ba(5678, length=24, endian="big")

    parsed = parse_csbk(bits)

    assert parsed.last_block is True
    assert parsed.csbko_name == "UnitToUnitVoiceServiceRequest"
    assert parsed.service_options_value == 0x81
    assert parsed.service_options.emergency is True
    assert parsed.target_address == 1234
    assert parsed.source_address == 5678


def test_parse_csbk_preamble_fields():
    bits = bitarray("0" * 96, endian="big")
    bits[0] = 1
    bits[2:8] = int2ba(0x3D, length=6, endian="big")
    bits[16] = 0
    bits[17] = 1
    bits[24:32] = int2ba(7, length=8, endian="big")
    bits[32:56] = int2ba(100, length=24, endian="big")
    bits[56:80] = int2ba(200, length=24, endian="big")

    parsed = parse_csbk(bits)

    assert parsed.csbko_name == "PreambleCSBK"
    assert parsed.csbk_content_follows_preambles is True
    assert parsed.target_address_is_individual is False
    assert parsed.blocks_to_follow == 7
    assert parsed.target_address == 100
    assert parsed.source_address == 200


def test_parse_embedded_signalling_lcss_bits():
    bits = bitarray("0" * 16, endian="big")
    bits[5:7] = int2ba(LCSS.FirstFragmentLC.value, length=2, endian="big")

    parsed = parse_embedded_signalling(bits)

    assert parsed.link_control_start_stop == LCSS.FirstFragmentLC
    assert parsed.emb_parity_ok is False


def test_dmr_session_emits_call_summary_on_finalize():
    session = DMRSessionAssembler()
    session.feed({
        "type": "LC_HEADER",
        "src": 10,
        "dst": 20,
        "flco": "GroupVoiceChannelUser",
        "fid": "StandardizedFID",
        "extra": {
            "fs_start": 1000,
            "color_code": 3,
            "flc": {"call_type": "group"},
        },
    })
    session.feed({
        "type": "LATE_ENTRY",
        "src": 10,
        "dst": 20,
        "flco": "GroupVoiceChannelUser",
        "fid": "StandardizedFID",
        "extra": {
            "fs_start": 49000,
            "color_code": 3,
            "flc": {"call_type": "group"},
        },
    })

    call = session.finalize()

    assert call["type"] == "DMR_CALL"
    assert call["src"] == 10
    assert call["dst"] == 20
    assert call["extra"]["duration_s"] == 1.0
    assert call["extra"]["late_entry_count"] == 1
    assert call["extra"]["closed_by"] == "end_of_scan"


def test_dmr_formatter_includes_full_decode_details():
    line = format_pdu({
        "protocol": "DMR",
        "type": "LC_HEADER",
        "src": 1,
        "dst": 2,
        "flco": "GroupVoiceChannelUser",
        "fid": "StandardizedFID",
        "extra": {
            "color_code": 0,
            "data_type": 1,
            "data_type_name": "VOICE_LC_HEADER",
            "fs_start": 123,
            "fec": {"golay_ok": True, "bptc_196_96_ok": True, "rs_12_9_4_ok": True},
            "flc": {
                "call_type": "group",
                "flco_value": 0,
                "fid_value": 0,
                "service_options_value": 0,
                "service_options": {
                    "emergency": False,
                    "privacy": False,
                    "broadcast": False,
                    "open_voice_call_mode": False,
                    "priority": 0,
                },
            },
        },
    })

    assert "TGID=2" in line
    assert "DT=1:VOICE_LC_HEADER" in line
    assert "FLCO=0x00(GroupVoiceChannelUser)" in line
    assert "FEC=[GOLAY=OK,BPTC=OK,RS=OK]" in line
