from __future__ import annotations

import numpy as np

from p25.constants import FS_NID_SYMBOLS, FS_SYMBOLS, NID_SYMBOLS
from p25.dsp import recover_symbols_from_fs, slice_symbols_to_bits
from p25.nid import decode_nid
from p25.sync import find_frame_sync


def decode(
    y: np.ndarray,
    sps: int = 10,
    sync_threshold: float = 0.62,
) -> list[dict]:
    results: list[dict] = []
    for candidate in find_frame_sync(y, sps=sps, threshold=sync_threshold):
        symbols = recover_symbols_from_fs(
            y,
            candidate,
            symbol_count=FS_NID_SYMBOLS,
            sps=sps,
        )
        if symbols is None:
            continue
        bits = slice_symbols_to_bits(symbols)
        nid_bits = bits[FS_SYMBOLS * 2:(FS_SYMBOLS + NID_SYMBOLS) * 2]
        try:
            nid = decode_nid(nid_bits)
        except ValueError:
            continue
        results.append(
            {
                "protocol": "P25",
                "type": "P25_NID",
                "src": 0,
                "dst": 0,
                "ts": 0,
                "flco": nid.duid_name,
                "fid": "",
                "extra": {
                    "nac": nid.nac,
                    "duid": nid.duid,
                    "duid_name": nid.duid_name,
                    "valid_bch": nid.valid_bch,
                    "corrected": nid.corrected,
                    "fs_start": candidate.fs_start,
                    "sync_ncc": candidate.ncc,
                },
                "raw_bits": bits.tobytes(),
            }
        )
    return results
