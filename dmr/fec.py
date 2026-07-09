"""Native DMR forward-error-correction helpers.

The functions in this module cover the DMR metadata paths used by the project:
Slot Type Golay, BPTC(196,96), Reed-Solomon(12,9,4), embedded LC VBPTC, QR,
and the 5-bit checksum.  They intentionally expose small functions instead of
protocol objects so link-layer code can stay independent from third-party DMR
libraries.
"""
from __future__ import annotations

from functools import lru_cache

from bitarray import bitarray
import numpy as np


def _ba(bits: list[int] | tuple[int, ...]) -> bitarray:
    out = bitarray(endian="big")
    out.extend(int(b) for b in bits)
    return out


def _bits_to_int(bits: bitarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _derive_parity_check(generator: np.ndarray) -> np.ndarray:
    k = generator.shape[0]
    return np.concatenate(
        (generator[:, k:].T, np.identity(generator.shape[1] - k, dtype=np.uint8)),
        axis=1,
    )


def _syndrome(bits: bitarray, parity_check: np.ndarray) -> np.ndarray:
    word = np.fromiter((int(b) for b in bits), dtype=np.uint8, count=len(bits))
    return (word @ parity_check.T) & 1


def _check_codeword(bits: bitarray, parity_check: np.ndarray) -> bool:
    return not np.any(_syndrome(bits, parity_check))


def _correct_hamming(bits: bitarray, parity_check: np.ndarray) -> bitarray:
    corrected = bits.copy()
    syndrome = _syndrome(corrected, parity_check)
    if not np.any(syndrome):
        return corrected

    columns = parity_check.T
    for idx, column in enumerate(columns):
        if np.array_equal(column, syndrome):
            corrected.invert(idx)
            return corrected
    return bits


_GOLAY_20_8_7_GENERATOR = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    ],
    dtype=np.uint8,
)
_GOLAY_20_8_7_H = _derive_parity_check(_GOLAY_20_8_7_GENERATOR)


def golay_20_8_7_check(bits: bitarray) -> bool:
    """Return True when a 20-bit DMR Slot Type Golay word is valid."""
    if len(bits) != 20:
        raise ValueError(f"Golay(20,8,7) expects 20 bits, got {len(bits)}")
    return _check_codeword(bits, _GOLAY_20_8_7_H)


_QR_16_7_6_GENERATOR = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    ],
    dtype=np.uint8,
)
_QR_16_7_6_H = _derive_parity_check(_QR_16_7_6_GENERATOR)


def qr_16_7_6_check(bits: bitarray) -> bool:
    """Return True when a 16-bit Embedded Signalling QR word is valid."""
    if len(bits) != 16:
        raise ValueError(f"QR(16,7,6) expects 16 bits, got {len(bits)}")
    return _check_codeword(bits, _QR_16_7_6_H)


_HAMMING_13_9_3_GENERATOR = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    ],
    dtype=np.uint8,
)
_HAMMING_15_11_3_GENERATOR = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    ],
    dtype=np.uint8,
)
_HAMMING_16_11_4_GENERATOR = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    ],
    dtype=np.uint8,
)
_HAMMING_13_9_3_H = _derive_parity_check(_HAMMING_13_9_3_GENERATOR)
_HAMMING_15_11_3_H = _derive_parity_check(_HAMMING_15_11_3_GENERATOR)
_HAMMING_16_11_4_H = _derive_parity_check(_HAMMING_16_11_4_GENERATOR)


def hamming_13_9_3_correct(bits: bitarray) -> bitarray:
    if len(bits) != 13:
        raise ValueError(f"Hamming(13,9,3) expects 13 bits, got {len(bits)}")
    return _correct_hamming(bits, _HAMMING_13_9_3_H)


def hamming_15_11_3_correct(bits: bitarray) -> bitarray:
    if len(bits) != 15:
        raise ValueError(f"Hamming(15,11,3) expects 15 bits, got {len(bits)}")
    return _correct_hamming(bits, _HAMMING_15_11_3_H)


def hamming_16_11_4_correct(bits: bitarray) -> bitarray:
    if len(bits) != 16:
        raise ValueError(f"Hamming(16,11,4) expects 16 bits, got {len(bits)}")
    return _correct_hamming(bits, _HAMMING_16_11_4_H)


def _bptc_196_96_table(bits: bitarray) -> list[list[int]]:
    table = [[0 for _ in range(15)] for _ in range(13)]
    for pos in range(1, 196):
        row = (pos - 1) // 15
        col = (pos - 1) % 15
        table[row][col] = int(bits[(181 * pos) % 196])
    return table


def _bptc_196_96_data_from_table(table: list[list[int]]) -> bitarray:
    out = bitarray(endian="big")
    for row in range(9):
        start_col = 3 if row == 0 else 0
        for col in range(start_col, 11):
            out.append(table[row][col])
    return out


def bptc_196_96_decode(bits: bitarray, repair_if_necessary: bool = True) -> bitarray:
    """Decode the 196-bit DMR BPTC payload to its 96 information bits."""
    if len(bits) != 196:
        raise ValueError(f"BPTC(196,96) expects 196 bits, got {len(bits)}")

    table = _bptc_196_96_table(bits)
    if repair_if_necessary:
        for row in range(9):
            table[row] = [int(b) for b in hamming_15_11_3_correct(_ba(table[row]))]
        for col in range(15):
            column = _ba([table[row][col] for row in range(13)])
            corrected = hamming_13_9_3_correct(column)
            for row in range(13):
                table[row][col] = int(corrected[row])
    return _bptc_196_96_data_from_table(table)


def _vbptc_128_72_table(bits: bitarray) -> list[list[int]]:
    table = [[0 for _ in range(16)] for _ in range(8)]
    for pos in range(128):
        row = pos // 16
        col = pos % 16
        table[row][col] = int(bits[row + 8 * col])
    return table


def vbptc_128_72_decode(
    bits: bitarray,
    include_cs5: bool = True,
    repair_if_necessary: bool = False,
) -> bitarray:
    """Decode a 128-bit embedded LC VBPTC block to 72 or 77 bits.

    The default keeps legacy behavior by only deinterleaving.  Set
    ``repair_if_necessary`` to True to apply row Hamming correction before
    extracting the information and checksum bits.
    """
    if len(bits) != 128:
        raise ValueError(f"VBPTC(128,72) expects 128 bits, got {len(bits)}")

    table = _vbptc_128_72_table(bits)
    if repair_if_necessary:
        for row in range(7):
            table[row] = [int(b) for b in hamming_16_11_4_correct(_ba(table[row]))]

    out = bitarray(endian="big")
    for row in range(7):
        limit = 11 if row < 2 else 10
        for col in range(limit):
            out.append(table[row][col])

    if include_cs5:
        for row in range(2, 7):
            out.append(table[row][10])
    return out


@lru_cache(maxsize=1)
def _gf256_tables() -> tuple[list[int], list[int]]:
    exp = [0] * 512
    log = [0] * 256
    value = 1
    for i in range(255):
        exp[i] = value
        log[value] = i
        value <<= 1
        if value & 0x100:
            value ^= 0x11D
        value &= 0xFF
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


def _gf256_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    exp, log = _gf256_tables()
    return exp[log[a] + log[b]]


def _xor_bytes(data: bytes, mask: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, mask))


def rs_12_9_4_generate(data: bytes, mask: bytes = b"\x00\x00\x00") -> bytes:
    """Return 9 data bytes plus the DMR RS(12,9,4) 3-byte parity."""
    if len(data) != 9:
        raise ValueError(f"RS(12,9,4) generate expects 9 data bytes, got {len(data)}")
    if len(mask) != 3:
        raise ValueError(f"RS(12,9,4) mask expects 3 bytes, got {len(mask)}")

    parity = [0x00, 0x00, 0x00]
    for byte in data:
        feedback = byte ^ parity[2]
        parity[2] = parity[1] ^ _gf256_mul(0x0E, feedback)
        parity[1] = parity[0] ^ _gf256_mul(0x38, feedback)
        parity[0] = _gf256_mul(0x40, feedback)
    return data + _xor_bytes(bytes(reversed(parity)), mask)


def rs_12_9_4_check(data: bytes, mask: bytes) -> bool:
    """Return True when a 12-byte RS(12,9,4) protected block is valid."""
    if len(data) != 12:
        raise ValueError(f"RS(12,9,4) expects 12 bytes, got {len(data)}")
    return rs_12_9_4_generate(data[:9], mask) == data


def five_bit_checksum(data: bytes) -> int:
    """Calculate the DMR 5-bit checksum over the 72-bit embedded LC payload."""
    if len(data) < 9:
        data = (b"\x00" * (9 - len(data))) + data
    if len(data) != 9:
        raise ValueError(f"5-bit checksum expects 9 bytes, got {len(data)}")
    return sum(data[i] for i in reversed(range(9))) % 31


def five_bit_checksum_verify(data: bytes, checksum: int) -> bool:
    if not 0 <= checksum < 31:
        return False
    return five_bit_checksum(data) == checksum

