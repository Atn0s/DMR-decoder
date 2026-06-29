from __future__ import annotations

from dataclasses import dataclass

from bitarray import bitarray
from bitarray.util import ba2int

from p25.dsp import deinterleave_es
from p25.fec import hamming_10_6_3_decode, rs_24_16_9_decode


@dataclass(frozen=True)
class EncryptionSync:
    mi: int
    algid: int
    kid: int
    rs_ok: bool
    hamming_corrected: int
    raw: bitarray


def decode_ldu2_es(frame_bits: bitarray) -> EncryptionSync | None:
    hexbits = []
    corrected = 0
    for group in deinterleave_es(frame_bits):
        data, fixed = hamming_10_6_3_decode(group)
        hexbits.append(ba2int(data))
        corrected += int(fixed)

    decoded, ok = rs_24_16_9_decode(hexbits)
    if not ok or decoded is None:
        return None

    es_bits = bitarray(endian="big")
    for sym in reversed(decoded):
        es_bits.extend(f"{sym:06b}")

    return EncryptionSync(
        mi=ba2int(es_bits[0:72]),
        algid=ba2int(es_bits[72:80]),
        kid=ba2int(es_bits[80:96]),
        rs_ok=True,
        hamming_corrected=corrected,
        raw=es_bits,
    )
