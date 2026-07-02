from __future__ import annotations

from collections.abc import Callable

import numpy as np

from common.config import DEFAULT_RADIO_CONFIG
from dmr import plugin as dmr_plugin
from dpmr import plugin as dpmr_plugin
from p25 import plugin as p25_plugin
from radio.protocol import ProtocolSpec, call_decoder


PROTOCOL_REGISTRY: tuple[ProtocolSpec, ...] = (
    dmr_plugin.SPEC,
    p25_plugin.SPEC,
    dpmr_plugin.SPEC,
)

SUPPORTED_PROTOCOLS = tuple(spec.name for spec in PROTOCOL_REGISTRY)
_PROTOCOL_BY_NAME = {spec.name: spec for spec in PROTOCOL_REGISTRY}
_PROTOCOL_ALIASES = {
    alias: spec.name
    for spec in PROTOCOL_REGISTRY
    for alias in (spec.name.lower(), *spec.aliases)
}


# Backward-compatible protocol helpers. New code should prefer ProtocolSpec.
_dmr_decode_loop = dmr_plugin._dmr_decode_loop
_decode_p25 = p25_plugin._decode_p25
_decode_dpmr = dpmr_plugin._decode_dpmr

decode_dmr = dmr_plugin.decode
decode_p25 = p25_plugin.decode
decode_dpmr = dpmr_plugin.decode

_dmr_dedup_key = dmr_plugin.dedup_key
_p25_dedup_key = p25_plugin.dedup_key
_dpmr_dedup_key = dpmr_plugin.dedup_key

format_dmr_pdu = dmr_plugin.format_pdu
format_p25_pdu = p25_plugin.format_pdu
format_dpmr_pdu = dpmr_plugin.format_pdu
_p25_detail = p25_plugin.p25_detail
_format_dpmr_cch = dpmr_plugin.format_cch


def _canonical_protocol_name(name: str) -> str:
    key = str(name).lower()
    if key not in _PROTOCOL_ALIASES:
        raise ValueError(f"unsupported protocol: {name}")
    return _PROTOCOL_ALIASES[key]


def spec_for_protocol(name: str) -> ProtocolSpec:
    return _PROTOCOL_BY_NAME[_canonical_protocol_name(name)]


def spec_for_pdu(pdu: dict) -> ProtocolSpec:
    return spec_for_protocol(pdu.get("protocol", "DMR"))


def normalize_protocol_names(
    protocol_names: list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    if protocol_names is None:
        return set(SUPPORTED_PROTOCOLS)
    normalized: set[str] = set()
    for name in protocol_names:
        normalized.add(_canonical_protocol_name(name))
    return normalized


def _call_decoder(decoder: Callable, y: np.ndarray, config: object) -> list[dict]:
    return call_decoder(decoder, y, config)


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
        results.extend(_call_decoder(decoder, frontends.get(spec.name, y), spec.config))
    return results


def decode_iq(
    iq_dec: np.ndarray,
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
    sample_rate: float = DEFAULT_RADIO_CONFIG.target_sample_rate_hz,
) -> list[dict]:
    enabled = normalize_protocol_names(protocol_names)
    frontends: dict[str, np.ndarray] = {}
    results: list[dict] = []
    for spec in PROTOCOL_REGISTRY:
        if spec.name not in enabled:
            continue
        if spec.frontend_key not in frontends:
            frontends[spec.frontend_key] = spec.frontend(iq_dec, sample_rate, spec.config)
        decoder = globals()[spec.decode_name]
        results.extend(_call_decoder(decoder, frontends[spec.frontend_key], spec.config))
    return results


def postprocess_pdus(
    pdus: list[dict],
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict]:
    enabled = normalize_protocol_names(protocol_names)
    processed = pdus
    for spec in PROTOCOL_REGISTRY:
        if spec.name in enabled:
            processed = spec.postprocess(processed)
    return processed


def dedup_key(pdu: dict) -> tuple:
    try:
        return spec_for_pdu(pdu).dedup_key(pdu)
    except ValueError:
        return _dmr_dedup_key(pdu)


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


def _fo_suffix(pdu: dict) -> str:
    return f" (fo={pdu['_fo_hz']/1e3:+.1f}kHz)" if "_fo_hz" in pdu else ""


def format_pdu(pdu: dict) -> str:
    try:
        formatter = spec_for_pdu(pdu).formatter
    except ValueError:
        formatter = format_dmr_pdu
    return formatter(pdu, _fo_suffix(pdu))
