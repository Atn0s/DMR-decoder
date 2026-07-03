from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    aliases: tuple[str, ...]
    config: object
    frontend_key: str
    frontend: Callable[[np.ndarray, float, object], np.ndarray]
    decode: Callable[[np.ndarray, object], list[dict]]
    postprocess: Callable[[list[dict]], list[dict]]
    dedup_key: Callable[[dict], tuple]
    formatter: Callable[[dict, str], str]


def postprocess_identity(pdus: list[dict]) -> list[dict]:
    return pdus
