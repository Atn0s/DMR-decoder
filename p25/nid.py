from __future__ import annotations

from dataclasses import dataclass

from bitarray import bitarray
from bitarray.util import ba2int

from p25.constants import DUID_NAMES, NID_BITS
from p25.fec import bch_63_16_decode


@dataclass(frozen=True)
class P25NID:
    nac: int
    duid: int
    duid_name: str
    valid_bch: bool | None
    corrected: bool
    raw_bits: bitarray


def decode_nid(bits: bitarray) -> P25NID:
    if len(bits) != NID_BITS:
        raise ValueError("P25 NID must be exactly 64 bits")
    raw_info = bits[0:16]
    synthetic_uncoded = raw_info.any() and not bits[16:64].any()
    info, corrected = (None, False) if synthetic_uncoded else bch_63_16_decode(bits)
    valid_bch = info is not None
    if info is None:
        info = raw_info
        corrected = False
    nac = ba2int(info[0:12])
    duid = ba2int(info[12:16])
    return P25NID(
        nac=nac,
        duid=duid,
        duid_name=DUID_NAMES.get(duid, f"UNKNOWN_0x{duid:X}"),
        valid_bch=valid_bch,
        corrected=corrected,
        raw_bits=bits.copy(),
    )
