from __future__ import annotations

from dataclasses import dataclass

from p25.nid import P25NID


@dataclass(frozen=True)
class P25FrameInfo:
    nac: int
    duid: int
    duid_name: str
    pdu_type: str
    category: str
    is_voice: bool
    is_control: bool
    is_terminator: bool
    has_link_control: bool


_FRAME_DEFS = {
    0x0: ("P25_HDU", "header", False, False, False, False),
    0x3: ("P25_TDU", "terminator", False, False, True, False),
    0x5: ("P25_LDU1", "voice", True, False, False, True),
    0x7: ("P25_TSBK", "trunking_control", False, True, False, False),
    0xA: ("P25_LDU2", "voice", True, False, False, False),
    0xC: ("P25_PDU", "packet_data", False, True, False, False),
    0xF: ("P25_TDULC", "terminator", False, False, True, True),
}


def frame_info_from_nid(nid: P25NID) -> P25FrameInfo:
    pdu_type, category, is_voice, is_control, is_terminator, has_lc = (
        _FRAME_DEFS.get(
            nid.duid,
            (f"P25_UNKNOWN_0x{nid.duid:X}", "unknown", False, False, False, False),
        )
    )
    return P25FrameInfo(
        nac=nid.nac,
        duid=nid.duid,
        duid_name=nid.duid_name,
        pdu_type=pdu_type,
        category=category,
        is_voice=is_voice,
        is_control=is_control,
        is_terminator=is_terminator,
        has_link_control=has_lc,
    )
