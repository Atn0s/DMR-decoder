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
NID_AIR_SYMBOLS = 33
NID_STATUS_SYMBOL_OFFSET = 11
NID_SYMBOLS = NID_AIR_SYMBOLS
FS_NID_SYMBOLS = FS_SYMBOLS + NID_SYMBOLS
LDU_SYMBOLS = 864
HDU_SYMBOLS = 864


def _ldu1_lc_positions() -> list[int]:
    """Return LDU1 LC Hamming bit positions in DSD-FME read order.

    Frame bit index space includes FS and NID. DSD-FME starts LDU payload
    processing after the 56-symbol FS+NID header, with the next status symbol
    14 dibits later (`status_count = 21`). Voice IMBE and LC dibits skip the
    periodic status symbol every 36 dibits.
    """
    sym = (FS_BITS + NID_BITS) // 2
    status_count = 21
    by_word: dict[tuple[str, int], list[int]] = {}

    def read_dibit(label: tuple[str, int] | None = None) -> None:
        nonlocal sym, status_count
        if status_count == 35:
            sym += 1
            status_count = 1
        else:
            status_count += 1
        sym += 1
        if label is not None:
            by_word.setdefault(label, []).extend([sym * 2, sym * 2 + 1])

    def imbe() -> None:
        for _ in range(72):
            read_dibit()

    def hexword(label: tuple[str, int]) -> None:
        for _ in range(5):
            read_dibit(label)

    imbe()
    imbe()
    for i in (11, 10, 9, 8):
        hexword(("d", i))
    imbe()
    for i in (7, 6, 5, 4):
        hexword(("d", i))
    imbe()
    for i in (3, 2, 1, 0):
        hexword(("d", i))
    imbe()
    for i in (11, 10, 9, 8):
        hexword(("p", i))
    imbe()
    for i in (7, 6, 5, 4):
        hexword(("p", i))
    imbe()
    for i in (3, 2, 1, 0):
        hexword(("p", i))

    positions: list[int] = []
    for typ in ("d", "p"):
        for i in range(12):
            positions.extend(by_word[(typ, i)])
    return positions


LC_HEXBIT_POSITIONS = _ldu1_lc_positions()


def _ldu2_es_positions() -> list[int]:
    """Return LDU2 encryption sync Hamming bit positions in DSD-FME read order."""
    sym = (FS_BITS + NID_BITS) // 2
    status_count = 21
    by_word: dict[tuple[str, int], list[int]] = {}

    def read_dibit(label: tuple[str, int] | None = None) -> None:
        nonlocal sym, status_count
        if status_count == 35:
            sym += 1
            status_count = 1
        else:
            status_count += 1
        sym += 1
        if label is not None:
            by_word.setdefault(label, []).extend([sym * 2, sym * 2 + 1])

    def imbe() -> None:
        for _ in range(72):
            read_dibit()

    def hexword(label: tuple[str, int]) -> None:
        for _ in range(5):
            read_dibit(label)

    imbe()
    imbe()
    for i in (15, 14, 13, 12):
        hexword(("d", i))
    imbe()
    for i in (11, 10, 9, 8):
        hexword(("d", i))
    imbe()
    for i in (7, 6, 5, 4):
        hexword(("d", i))
    imbe()
    for i in (3, 2, 1, 0):
        hexword(("d", i))
    imbe()
    for i in (7, 6, 5, 4):
        hexword(("p", i))
    imbe()
    for i in (3, 2, 1, 0):
        hexword(("p", i))

    positions: list[int] = []
    for typ, count in (("d", 16), ("p", 8)):
        for i in range(count):
            positions.extend(by_word[(typ, i)])
    return positions


ES_HEXBIT_POSITIONS = _ldu2_es_positions()


def _hdu_hexbit_positions() -> tuple[list[int], list[int]]:
    """Return HDU data/parity Golay bit positions in DSD-FME read order."""
    sym = (FS_BITS + NID_BITS) // 2
    status_count = 21
    symbol_words: dict[int, list[int]] = {}
    golay_words: dict[int, list[int]] = {}

    def read_dibit(label: tuple[str, int] | None = None) -> None:
        nonlocal sym, status_count
        if status_count == 35:
            sym += 1
            status_count = 1
        else:
            status_count += 1
        sym += 1
        if label is not None:
            target = golay_words if label[0] == "g" else symbol_words
            target.setdefault(label[1], []).extend([sym * 2, sym * 2 + 1])

    def read_bits(bit_count: int, label: tuple[str, int]) -> None:
        for _ in range(bit_count // 2):
            read_dibit(label)

    for i in range(19, -1, -1):
        read_bits(6, ("d", i))
        read_bits(12, ("g", i))
    for i in range(15, -1, -1):
        read_bits(6, ("r", i + 20))
        read_bits(12, ("g", i + 20))

    data_positions: list[int] = []
    parity_positions: list[int] = []
    for i in range(36):
        data_positions.extend(symbol_words[i])
    for i in range(36):
        parity_positions.extend(golay_words[i])
    return data_positions, parity_positions


HDU_DATA_HEXBIT_POSITIONS, HDU_GOLAY_PARITY_POSITIONS = _hdu_hexbit_positions()


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
