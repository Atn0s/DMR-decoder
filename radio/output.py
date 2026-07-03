from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping

from radio import registry
from radio.pdu import PDU, pdu_to_dict


PDUItem = Mapping[str, object] | PDU


def format_lines(pdus: Iterable[PDUItem]) -> list[str]:
    return [registry.format_pdu(pdu) for pdu in pdus]


def print_results(pdus: Iterable[PDUItem]) -> None:
    for line in format_lines(pdus):
        print(line)


def json_ready(pdus: Iterable[PDUItem]) -> list[dict]:
    return [pdu_to_dict(pdu, include_raw_bits=False) for pdu in pdus]


def write_json(pdus: Iterable[PDUItem], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_ready(pdus), f, indent=2, default=str)
