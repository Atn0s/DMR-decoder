from __future__ import annotations

COLOR_CODE_BY_CHANNEL_CODE = {
    0x575F77: 0,
    0x577577: 1,
    0x57DD75: 2,
    0x57F775: 3,
    0x55577D: 4,
    0x557D7D: 5,
    0x55D57F: 6,
    0x55FF7F: 7,
    0x5F555F: 8,
    0x5F7F5F: 9,
    0x5FD75D: 10,
    0x5FFD5D: 11,
    0x5D5D55: 12,
    0x5D7755: 13,
    0x5DDF57: 14,
    0x5DF557: 15,
    0x775DD7: 16,
    0x7777D7: 17,
    0x77DFD5: 18,
    0x77F5D5: 19,
    0x7555DD: 20,
    0x757FDD: 21,
    0x75D7DF: 22,
    0x75FDDF: 23,
    0x7F57FF: 24,
    0x7F7DFF: 25,
    0x7FD5FD: 26,
    0x7FFFFD: 27,
    0x7D5FF5: 28,
    0x7D75F5: 29,
    0x7DDDF7: 30,
    0x7DF7F7: 31,
    0xD755F7: 32,
    0xD77FF7: 33,
    0xD7D7F5: 34,
    0xD7FDF5: 35,
    0xD55DFD: 36,
    0xD577FD: 37,
    0xD5DFFF: 38,
    0xD5F5FF: 39,
    0xDF5FDF: 40,
    0xDF75DF: 41,
    0xDFDDDD: 42,
    0xDFF7DD: 43,
    0xDD57D5: 44,
    0xDD7DD5: 45,
    0xDDD5D7: 46,
    0xDDFFD7: 47,
    0xF75757: 48,
    0xF77D57: 49,
    0xF7D555: 50,
    0xF7FF55: 51,
    0xF55F5D: 52,
    0xF5755D: 53,
    0xF5DD5F: 54,
    0xF5F75F: 55,
    0xFF5D7F: 56,
    0xFF777F: 57,
    0xFFDF7D: 58,
    0xFFF57D: 59,
    0xFD5575: 60,
    0xFD7F75: 61,
    0xFDD777: 62,
    0xFDFD77: 63,
}


def bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


def get_color_code(channel_code_bits: list[int]) -> int:
    if len(channel_code_bits) != 24:
        raise ValueError("dPMR channel code must be 24 bits")
    channel_code = bits_to_int(channel_code_bits) | 0x555555
    return COLOR_CODE_BY_CHANNEL_CODE.get(channel_code, -1)
