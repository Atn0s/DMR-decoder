from __future__ import annotations

from bitarray import bitarray
from bitarray.util import ba2int

from p25.dsp import deinterleave_lc
from p25.fec import hamming_10_6_3_decode, rs_24_12_13_decode
from p25.link_control import LinkControl, parse_link_control


def decode_ldu1_lc(frame_bits: bitarray) -> LinkControl | None:
    hexbits = []
    for group in deinterleave_lc(frame_bits):
        data, _ = hamming_10_6_3_decode(group)
        hexbits.append(ba2int(data))
    decoded, ok = rs_24_12_13_decode(hexbits)
    if not ok or decoded is None:
        return None
    lc72 = bitarray(endian="big")
    for sym in reversed(decoded):
        lc72.extend(f"{sym:06b}")
    return parse_link_control(lc72)
