from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dmr.offline as dmr_offline
from dpmr.decoder import decode as _decode_dpmr
from p25.decoder import decode as decode_p25


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    aliases: tuple[str, ...]
    decode_name: str


PROTOCOL_REGISTRY: tuple[ProtocolSpec, ...] = (
    ProtocolSpec("DMR", ("dmr",), "decode_dmr"),
    ProtocolSpec("P25", ("p25",), "decode_p25"),
    ProtocolSpec("dPMR", ("dpmr",), "decode_dpmr"),
)

SUPPORTED_PROTOCOLS = tuple(spec.name for spec in PROTOCOL_REGISTRY)
_PROTOCOL_BY_NAME = {spec.name: spec for spec in PROTOCOL_REGISTRY}
_PROTOCOL_ALIASES = {
    alias: spec.name
    for spec in PROTOCOL_REGISTRY
    for alias in (spec.name.lower(), *spec.aliases)
}


def normalize_protocol_names(
    protocol_names: list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    if protocol_names is None:
        return set(SUPPORTED_PROTOCOLS)
    normalized: set[str] = set()
    for name in protocol_names:
        key = name.lower()
        if key not in _PROTOCOL_ALIASES:
            raise ValueError(f"unsupported protocol: {name}")
        normalized.add(_PROTOCOL_ALIASES[key])
    return normalized


def _dmr_decode_loop(y: np.ndarray) -> list[dict]:
    return dmr_offline._decode_dmr_loop(y)


def decode_dmr(y: np.ndarray) -> list[dict]:
    pdus = _dmr_decode_loop(y)
    for pdu in pdus:
        pdu.setdefault("protocol", "DMR")
    return pdus


def decode_dpmr(y: np.ndarray) -> list[dict]:
    return _decode_dpmr(y)


def decode_all(
    y: np.ndarray,
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
    frontends: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    enabled = normalize_protocol_names(protocol_names)
    frontends = frontends or {}
    results: list[dict] = []
    for spec in PROTOCOL_REGISTRY:
        if spec.name not in enabled:
            continue
        decoder = globals()[spec.decode_name]
        results.extend(decoder(frontends.get(spec.name, y)))
    return results


def dedup_key(pdu: dict) -> tuple:
    proto = pdu.get("protocol", "DMR")
    try:
        proto = next(iter(normalize_protocol_names({proto})))
    except ValueError:
        proto = str(proto)

    if proto == "P25":
        extra = pdu.get("extra", {})
        frame_bucket = round(extra.get("fs_start", 0) / 8640)
        return ("P25", extra.get("nac"), pdu.get("type", ""), frame_bucket)
    if proto == "dPMR":
        extra = pdu.get("extra", {})
        frame_bucket = round(extra.get("fs_start", 0) / 3840)
        return (
            "dPMR",
            pdu.get("src", ""),
            pdu.get("dst", ""),
            extra.get("color_code"),
            frame_bucket,
        )

    fo_bucket = round(pdu.get("_fo_hz", 0) / 5000) * 5000
    return (
        proto,
        pdu.get("src", 0),
        pdu.get("dst", 0),
        pdu.get("type", ""),
        fo_bucket,
    )


def deduplicate_pdus(pdus: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for pdu in pdus:
        key = dedup_key(pdu)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pdu)
    return unique
