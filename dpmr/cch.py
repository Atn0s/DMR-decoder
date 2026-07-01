from __future__ import annotations

from dataclasses import dataclass

from dpmr.color_code import bits_to_int

H12_8_H = (
    (1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0),
    (1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0),
    (1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0),
    (0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1),
)

H12_8_CORR = {
    0b1110: 0,
    0b0111: 1,
    0b1010: 2,
    0b0101: 3,
    0b1011: 4,
    0b1100: 5,
    0b0110: 6,
    0b0011: 7,
    0b1000: 8,
    0b0100: 9,
    0b0010: 10,
    0b0001: 11,
}


@dataclass(frozen=True)
class CCHRecord:
    frame_number: int
    id_half: int
    communication_mode: int
    version: int
    comms_format: int
    emergency_priority: int
    reserved: int
    slow_data: int
    crc_value: int
    crc_computed: int
    crc_ok: bool
    hamming_ok: bool
    hamming_blocks_ok: tuple[bool, ...]
    corrected_bits: int
    bits: tuple[int, ...]


def descramble(bits: list[int], lfsr_value: int = 0x1FF) -> list[int]:
    s = [(lfsr_value >> i) & 1 for i in range(9)]
    out: list[int] = []
    for bit in bits:
        out.append((int(bit) ^ s[0]) & 1)
        temp = s[4] ^ s[0]
        s = [s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], temp]
    return out


def deinterleave_6x12(bits: list[int]) -> list[int]:
    if len(bits) != 72:
        raise ValueError("dPMR CCH interleaver expects 72 bits")
    matrix = [bits[i * 6:(i + 1) * 6] for i in range(12)]
    out: list[int] = []
    for col in range(6):
        for row in range(12):
            out.append(int(matrix[row][col]))
    return out


def crc7(bits: list[int]) -> int:
    reg = 0
    poly = 0x09
    for bit in bits:
        if ((reg >> 6) & 1) ^ (int(bit) & 1):
            reg = ((reg << 1) ^ poly) & 0x7F
        else:
            reg = (reg << 1) & 0x7F
    return reg


def hamming_12_8_decode(codeword: list[int]) -> tuple[list[int], bool, int]:
    if len(codeword) != 12:
        raise ValueError("Hamming(12,8) codeword must be 12 bits")
    bits = [int(bit) & 1 for bit in codeword]
    syndrome = 0
    for row_idx, row in enumerate(H12_8_H):
        total = sum(bits[col] * row[col] for col in range(12)) & 1
        syndrome |= total << (3 - row_idx)
    corrected = 0
    ok = True
    if syndrome:
        pos = H12_8_CORR.get(syndrome)
        if pos is None:
            ok = False
        else:
            bits[pos] ^= 1
            corrected = 1
    return bits[:8], ok, corrected


def decode_cch(raw_bits: list[int]) -> CCHRecord | None:
    if len(raw_bits) != 72:
        return None
    descrambled = descramble(raw_bits)
    deinterleaved = deinterleave_6x12(descrambled)
    data: list[int] = []
    block_ok: list[bool] = []
    corrected = 0
    for idx in range(6):
        decoded, ok, corr = hamming_12_8_decode(deinterleaved[idx * 12:(idx + 1) * 12])
        data.extend(decoded)
        block_ok.append(ok)
        corrected += corr
    got_crc = bits_to_int(data[41:48])
    computed_crc = crc7(data[:41])
    return CCHRecord(
        frame_number=bits_to_int(data[0:2]),
        id_half=bits_to_int(data[2:14]),
        communication_mode=bits_to_int(data[14:17]),
        version=bits_to_int(data[17:19]),
        comms_format=bits_to_int(data[19:21]),
        emergency_priority=data[21],
        reserved=data[22],
        slow_data=bits_to_int(data[23:41]),
        crc_value=got_crc,
        crc_computed=computed_crc,
        crc_ok=got_crc == computed_crc,
        hamming_ok=all(block_ok),
        hamming_blocks_ok=tuple(block_ok),
        corrected_bits=corrected,
        bits=tuple(data),
    )


def air_interface_id_to_str(ai_id: int) -> str:
    weights = (1464100, 146410, 14641, 1331, 121, 11, 1)
    value = int(ai_id)
    chars: list[str] = []
    for weight in weights:
        digit = value // weight
        value %= weight
        chars.append("*" if digit == 10 else str(digit))
    return "".join(chars)
