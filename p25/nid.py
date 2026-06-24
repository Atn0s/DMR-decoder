from __future__ import annotations

from dataclasses import dataclass

from bitarray import bitarray
from bitarray.util import ba2int

from p25.constants import DUID_NAMES, NID_BITS


@dataclass(frozen=True)
class P25NID:
    nac: int
    duid: int
    duid_name: str
    valid_bch: bool | None
    corrected: bool
    raw_bits: bitarray


def decode_nid(bits: bitarray) -> P25NID:
    """Decode P25 NID shape.

    First milestone extracts the protected 16 information bits directly.
    BCH validation/repair is intentionally represented by `valid_bch=None`
    until `p25.fec` is implemented with vectors.
    """
    if len(bits) != NID_BITS:
        raise ValueError("P25 NID must be exactly 64 bits")
    nac = ba2int(bits[0:12])
    duid = ba2int(bits[12:16])
    return P25NID(
        nac=nac,
        duid=duid,
        duid_name=DUID_NAMES.get(duid, f"UNKNOWN_0x{duid:X}"),
        valid_bch=None,
        corrected=False,
        raw_bits=bits.copy(),
    )
