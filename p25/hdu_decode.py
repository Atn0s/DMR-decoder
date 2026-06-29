from __future__ import annotations

from dataclasses import dataclass

from bitarray import bitarray
from bitarray.util import ba2int

from p25.dsp import deinterleave_hdu
from p25.fec import golay_24_6_decode, rs_36_20_17_decode


@dataclass(frozen=True)
class HeaderCodeWord:
    mi: int
    mfid: int
    algid: int
    kid: int
    tgid: int
    rs_ok: bool
    golay_corrected: int
    raw: bitarray


def decode_hdu_hcw(frame_bits: bitarray) -> HeaderCodeWord | None:
    symbols: list[int] = []
    corrected = 0
    for data6, parity12 in deinterleave_hdu(frame_bits):
        value, fixed = golay_24_6_decode(data6, parity12)
        symbols.append(value)
        corrected += int(fixed)

    decoded, ok = rs_36_20_17_decode(symbols)
    if not ok or decoded is None:
        return None

    hcw = bitarray(endian="big")
    for sym in reversed(decoded):
        hcw.extend(f"{sym:06b}")

    return HeaderCodeWord(
        mi=ba2int(hcw[0:72]),
        mfid=ba2int(hcw[72:80]),
        algid=ba2int(hcw[80:88]),
        kid=ba2int(hcw[88:104]),
        tgid=ba2int(hcw[104:120]),
        rs_ok=True,
        golay_corrected=corrected,
        raw=hcw,
    )
