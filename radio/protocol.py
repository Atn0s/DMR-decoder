from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect

import numpy as np


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    aliases: tuple[str, ...]
    decode_name: str
    config: object
    frontend_key: str
    frontend: Callable[[np.ndarray, float, object], np.ndarray]
    postprocess: Callable[[list[dict]], list[dict]]
    dedup_key: Callable[[dict], tuple]
    formatter: Callable[[dict, str], str]


def postprocess_identity(pdus: list[dict]) -> list[dict]:
    return pdus


def call_decoder(decoder: Callable, y: np.ndarray, config: object) -> list[dict]:
    params = list(inspect.signature(decoder).parameters.values())
    accepts_config = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params)
    if not accepts_config:
        accepts_config = len(params) >= 2
    if accepts_config:
        return decoder(y, config)
    return decoder(y)
