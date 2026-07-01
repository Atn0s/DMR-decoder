from __future__ import annotations

import numpy as np

import dmr.offline as dmr_offline
from dpmr.decoder import decode as _decode_dpmr
from p25.decoder import decode as decode_p25


def _dmr_decode_loop(y: np.ndarray) -> list[dict]:
    return dmr_offline._decode_dmr_loop(y)


def decode_dmr(y: np.ndarray) -> list[dict]:
    pdus = _dmr_decode_loop(y)
    for pdu in pdus:
        pdu.setdefault("protocol", "DMR")
    return pdus


def decode_dpmr(y: np.ndarray) -> list[dict]:
    return _decode_dpmr(y)


def decode_all(y: np.ndarray, protocol_names: set[str] | None = None) -> list[dict]:
    protocol_names = protocol_names or {"DMR", "P25", "dPMR"}
    results: list[dict] = []
    if "DMR" in protocol_names:
        results.extend(decode_dmr(y))
    if "P25" in protocol_names:
        results.extend(decode_p25(y))
    if "dPMR" in protocol_names:
        results.extend(decode_dpmr(y))
    return results
