from __future__ import annotations

import numpy as np

import scanner
from dpmr.decoder import decode as decode_dpmr
from p25.decoder import decode as decode_p25


def _dmr_decode_loop(y: np.ndarray) -> list[dict]:
    return scanner._decode_dmr_loop(y)


def decode_dmr(y: np.ndarray) -> list[dict]:
    pdus = _dmr_decode_loop(y)
    for pdu in pdus:
        pdu.setdefault("protocol", "DMR")
    return pdus


def decode_all(y: np.ndarray) -> list[dict]:
    results: list[dict] = []
    results.extend(decode_dmr(y))
    results.extend(decode_p25(y))
    results.extend(decode_dpmr(y))
    return results
