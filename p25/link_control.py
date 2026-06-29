from __future__ import annotations

from dataclasses import dataclass

from bitarray import bitarray
from bitarray.util import ba2int

LCO_GROUP_VOICE = 0x00
LCO_UNIT_TO_UNIT = 0x03


@dataclass(frozen=True)
class LinkControl:
    lco: int
    mfid: int
    svc: int
    lc_info: int
    octet2: int
    octet3: int
    emergency: bool | None
    reserved: int
    reserved_bits: int
    src: int
    dst: int
    tgid: int
    is_group: bool
    call_type: str
    raw: bitarray


def parse_link_control(lc72: bitarray) -> LinkControl | None:
    if len(lc72) != 72:
        return None
    lco = ba2int(lc72[0:8])
    mfid = ba2int(lc72[8:16])
    octet2 = ba2int(lc72[16:24])
    octet3 = ba2int(lc72[24:32])
    lc_info = ba2int(lc72[16:32])
    svc = octet2
    is_group = lco == LCO_GROUP_VOICE
    if is_group:
        emergency = bool(lc72[16])
        reserved = ba2int(lc72[17:32])
        reserved_bits = 15
        tgid = ba2int(lc72[32:48])
        src = ba2int(lc72[48:72])
        dst = tgid
        call_type = "group"
    elif lco == LCO_UNIT_TO_UNIT:
        emergency = None
        reserved = octet2
        reserved_bits = 8
        tgid = 0
        dst = ba2int(lc72[24:48])
        src = ba2int(lc72[48:72])
        call_type = "unit_to_unit"
    else:
        emergency = None
        reserved = lc_info
        reserved_bits = 16
        tgid = 0
        src = 0
        dst = 0
        call_type = f"unknown_0x{lco:02X}"
    return LinkControl(
        lco=lco,
        mfid=mfid,
        svc=svc,
        lc_info=lc_info,
        octet2=octet2,
        octet3=octet3,
        emergency=emergency,
        reserved=reserved,
        reserved_bits=reserved_bits,
        src=src,
        dst=dst,
        tgid=tgid,
        is_group=is_group,
        call_type=call_type,
        raw=lc72.copy(),
    )
