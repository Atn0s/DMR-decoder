from __future__ import annotations

import numpy as np

FRAME_SYNC_HEX = "5575F5FF77FF"
FRAME_SYNC_BITS = "".join(f"{int(c, 16):04b}" for c in FRAME_SYNC_HEX)

DIBIT_TO_SYMBOL = {
    "00": 1,
    "01": 3,
    "10": -1,
    "11": -3,
}
SYMBOL_TO_DIBIT = {v: k for k, v in DIBIT_TO_SYMBOL.items()}

DUID_NAMES = {
    0x0: "HDU",
    0x3: "TDU",
    0x5: "LDU1",
    0x7: "TSBK",
    0xA: "LDU2",
    0xC: "PDU",
    0xF: "TDULC",
}

FS_BITS = 48
NID_BITS = 64
FS_SYMBOLS = FS_BITS // 2
NID_SYMBOLS = NID_BITS // 2
FS_NID_SYMBOLS = FS_SYMBOLS + NID_SYMBOLS


def dibits_to_symbols(bits: str) -> np.ndarray:
    if len(bits) % 2 != 0:
        raise ValueError("dibit string length must be even")
    return np.array(
        [DIBIT_TO_SYMBOL[bits[i:i + 2]] for i in range(0, len(bits), 2)],
        dtype=float,
    )


def symbols_to_dibits(symbols: np.ndarray) -> str:
    levels = np.array([-3, -1, 1, 3])
    nearest = levels[np.argmin(np.abs(symbols[:, None] - levels[None, :]), axis=1)]
    return "".join(SYMBOL_TO_DIBIT[int(v)] for v in nearest)


FRAME_SYNC_SYMBOLS = dibits_to_symbols(FRAME_SYNC_BITS)
