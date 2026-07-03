from __future__ import annotations

import numpy as np

from common.config import DEFAULT_RADIO_CONFIG
from dmr import plugin as dmr_plugin
from dpmr import plugin as dpmr_plugin
from p25 import plugin as p25_plugin
from radio.pdu import pdu_get, pdu_to_dict
from radio.protocol import ProtocolSpec


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


def _canonical_protocol_name(name: str) -> str:
    key = str(name).lower()
    if key not in _PROTOCOL_ALIASES:
        raise ValueError(f"unsupported protocol: {name}")
    return _PROTOCOL_ALIASES[key]


def spec_for_protocol(name: str) -> ProtocolSpec:
    return _PROTOCOL_BY_NAME[_canonical_protocol_name(name)]


def spec_for_pdu(pdu: dict) -> ProtocolSpec:
    return spec_for_protocol(pdu_get(pdu, "protocol", "DMR"))


def normalize_protocol_names(
    protocol_names: list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    if protocol_names is None:
        return set(SUPPORTED_PROTOCOLS)
    return {_canonical_protocol_name(name) for name in protocol_names}


def decode_iq_enabled(
    iq_dec: np.ndarray,
    enabled_protocols: set[str],
    sample_rate: float = DEFAULT_RADIO_CONFIG.target_sample_rate_hz,
) -> list[dict]:
    frontends: dict[str, np.ndarray] = {}
    results: list[dict] = []
    for spec in PROTOCOL_REGISTRY:
        if spec.name not in enabled_protocols:
            continue
        if spec.frontend_key not in frontends:
            frontends[spec.frontend_key] = spec.frontend(iq_dec, sample_rate, spec.config)
        results.extend(spec.decode(frontends[spec.frontend_key], spec.config))
    return results


def decode_iq(
    iq_dec: np.ndarray,
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
    sample_rate: float = DEFAULT_RADIO_CONFIG.target_sample_rate_hz,
) -> list[dict]:
    """Convenience/testing wrapper; the offline pipeline uses decode_iq_enabled()."""
    return decode_iq_enabled(
        iq_dec,
        normalize_protocol_names(protocol_names),
        sample_rate,
    )


def postprocess_pdus_enabled(pdus: list[dict], enabled_protocols: set[str]) -> list[dict]:
    processed = pdus
    for spec in PROTOCOL_REGISTRY:
        if spec.name in enabled_protocols:
            processed = spec.postprocess(processed)
    return processed


def postprocess_pdus(
    pdus: list[dict],
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict]:
    return postprocess_pdus_enabled(pdus, normalize_protocol_names(protocol_names))


def dedup_key(pdu: dict) -> tuple:
    try:
        return spec_for_pdu(pdu).dedup_key(pdu_to_dict(pdu))
    except ValueError:
        return dmr_plugin.dedup_key(pdu_to_dict(pdu))


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
    fo_hz = pdu_get(pdu, "_fo_hz")
    return f" (fo={fo_hz/1e3:+.1f}kHz)" if fo_hz is not None else ""


def format_pdu(pdu: dict) -> str:
    try:
        formatter = spec_for_pdu(pdu).formatter
    except ValueError:
        formatter = dmr_plugin.format_pdu
    return formatter(pdu_to_dict(pdu), _fo_suffix(pdu))
