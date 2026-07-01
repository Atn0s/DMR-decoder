from __future__ import annotations

import numpy as np

FS_DEC = 48_000
SYMBOL_RATE = 2_400
SPS = FS_DEC // SYMBOL_RATE

DPMR_FRONTEND_CUTOFF = 3_500.0
DPMR_DEV_NOMINAL = 1944.0

DPMR_FRAME_SYNC_1 = "111333331133131131111313"
DPMR_FRAME_SYNC_2 = "113333131331"
DPMR_FRAME_SYNC_3 = "133131333311"
DPMR_FRAME_SYNC_4 = "333111113311313313333131"
INV_DPMR_FRAME_SYNC_1 = "333111113311313313333131"
INV_DPMR_FRAME_SYNC_2 = "331111313113"
INV_DPMR_FRAME_SYNC_3 = "311313111133"
INV_DPMR_FRAME_SYNC_4 = "111333331133131131111313"

FS1_SYMBOLS = np.array([int(ch) for ch in DPMR_FRAME_SYNC_1], dtype=float)
FS2_SYMBOLS = np.array([int(ch) for ch in DPMR_FRAME_SYNC_2], dtype=float)
FS3_SYMBOLS = np.array([int(ch) for ch in DPMR_FRAME_SYNC_3], dtype=float)
FS4_SYMBOLS = np.array([int(ch) for ch in DPMR_FRAME_SYNC_4], dtype=float)
INV_FS1_SYMBOLS = np.array([int(ch) for ch in INV_DPMR_FRAME_SYNC_1], dtype=float)
INV_FS2_SYMBOLS = np.array([int(ch) for ch in INV_DPMR_FRAME_SYNC_2], dtype=float)
INV_FS3_SYMBOLS = np.array([int(ch) for ch in INV_DPMR_FRAME_SYNC_3], dtype=float)
INV_FS4_SYMBOLS = np.array([int(ch) for ch in INV_DPMR_FRAME_SYNC_4], dtype=float)

FS2_SYMBOL_COUNT = len(FS2_SYMBOLS)
DPMR_FRAME_SYMBOLS = 384
CCH_SYMBOLS = 36
TCH_SYMBOLS = 144
CC_SYMBOLS = 12
HEADER_FS1_PAYLOAD_SYMBOLS = DPMR_FRAME_SYMBOLS - len(FS1_SYMBOLS)
VOICE_FS2_PAYLOAD_SYMBOLS = CCH_SYMBOLS + TCH_SYMBOLS + CC_SYMBOLS + CCH_SYMBOLS + TCH_SYMBOLS
VOICE_FS2_TOTAL_SYMBOLS = FS2_SYMBOL_COUNT + VOICE_FS2_PAYLOAD_SYMBOLS

DIBIT_TO_BITS = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1),
}

# dsd-fme digitize() mapping for non-inverted 4FSK:
# +1 -> 0, +3 -> 1, -1 -> 2, -3 -> 3.
DIBIT_TO_LEVEL = {
    0: 1,
    1: 3,
    2: -1,
    3: -3,
}
LEVEL_TO_DIBIT = {value: key for key, value in DIBIT_TO_LEVEL.items()}
