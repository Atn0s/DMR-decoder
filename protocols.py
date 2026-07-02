from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from common.config import DEFAULT_RADIO_CONFIG
from dmr.config import DEFAULT_DMR_CONFIG
import dmr.offline as dmr_offline
from dmr.dsp import frontend as _frontend_c4fm
from dpmr.config import DEFAULT_DPMR_CONFIG
from dpmr.decoder import (
    decode as _decode_dpmr,
    filter_stable_pdus as _filter_stable_dpmr_pdus,
)
from dpmr.dsp import frontend_dpmr as _frontend_dpmr
from p25.config import DEFAULT_P25_CONFIG
from p25.decoder import decode as decode_p25


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    aliases: tuple[str, ...]
    decode_name: str
    config: object
    frontend_key: str
    frontend: Callable[[np.ndarray, float], np.ndarray]
    postprocess: Callable[[list[dict]], list[dict]]
    dedup_key: Callable[[dict], tuple]
    formatter: Callable[[dict, str], str]


def _frontend_dmr_p25(iq_dec: np.ndarray, sample_rate: float) -> np.ndarray:
    return _frontend_c4fm(iq_dec, fo=0.0, fs=sample_rate)


def _postprocess_identity(pdus: list[dict]) -> list[dict]:
    return pdus


def _dmr_dedup_key(pdu: dict) -> tuple:
    proto = pdu.get("protocol", "DMR")
    try:
        proto = _canonical_protocol_name(proto)
    except ValueError:
        proto = str(proto)
    fo_bucket = round(pdu.get("_fo_hz", 0) / 5000) * 5000
    return (
        proto,
        pdu.get("src", 0),
        pdu.get("dst", 0),
        pdu.get("type", ""),
        fo_bucket,
    )


def _p25_dedup_key(pdu: dict) -> tuple:
    extra = pdu.get("extra", {})
    frame_bucket = round(extra.get("fs_start", 0) / 8640)
    return ("P25", extra.get("nac"), pdu.get("type", ""), frame_bucket)


def _dpmr_dedup_key(pdu: dict) -> tuple:
    extra = pdu.get("extra", {})
    frame_bucket = round(extra.get("fs_start", 0) / 3840)
    return (
        "dPMR",
        pdu.get("src", ""),
        pdu.get("dst", ""),
        extra.get("color_code"),
        frame_bucket,
    )


def format_dmr_pdu(pdu: dict, fo_str: str = "") -> str:
    proto = pdu.get("protocol", "DMR")
    return (
        f"[{pdu['type']:<12}] PROTO={proto} SRC={pdu['src']} DST={pdu['dst']} "
        f"FLCO={pdu['flco']} FID={pdu.get('fid','')}{fo_str}"
    )


def format_dpmr_pdu(pdu: dict, fo_str: str = "") -> str:
    extra = pdu.get("extra", {})
    color_code = extra.get("color_code", -1)
    pol = "INV" if extra.get("polarity_inverted") else "NORM"
    sync_type = extra.get("sync_type", "")
    src = pdu.get("src") or ""
    dst = pdu.get("dst") or ""
    cc_text = f"{color_code:02d}" if isinstance(color_code, int) and color_code >= 0 else "--"
    cch_text = _format_dpmr_cch(extra.get("cch", []))
    return (
        f"[{pdu['type']:<12}] PROTO=dPMR SRC={src} DST={dst} "
        f"CC={cc_text} SYNC={sync_type} POL={pol}{cch_text}{fo_str}"
    )


def _format_dpmr_cch(cch_records: list[dict | None]) -> str:
    records = [record for record in cch_records if isinstance(record, dict)]
    if not records:
        return ""
    parts = []
    for record in records:
        parts.append(
            "FN={fn} IDH=0x{idh:03X} M={mode} V={version} F={fmt} "
            "E={emergency} RES={reserved} SLD=0x{slow:05X}".format(
                fn=record.get("frame_number", 0),
                idh=record.get("id_half", 0),
                mode=record.get("communication_mode", 0),
                version=record.get("version", 0),
                fmt=record.get("comms_format", 0),
                emergency=record.get("emergency_priority", 0),
                reserved=record.get("reserved", 0),
                slow=record.get("slow_data", 0),
            )
        )
    return " CCH=[" + "; ".join(parts) + "]"


def format_p25_pdu(pdu: dict, fo_str: str = "") -> str:
    extra = pdu.get("extra", {})
    prefix = f"[{pdu['type']:<12}] PROTO=P25"
    nac = f" NAC=0x{extra['nac']:03X}" if "nac" in extra else ""
    detail = _p25_detail(pdu)

    if pdu.get("type") == "P25_HDU":
        return f"{prefix} FRAME=HDU{nac}{detail}{fo_str}"

    if pdu.get("type") == "P25_LDU1":
        call_type = extra.get("call_type", "")
        if call_type == "group":
            party = f" SRC={pdu.get('src', 0)} TGID={extra.get('tgid', 0)}"
        elif call_type == "unit_to_unit":
            party = f" SRC={pdu.get('src', 0)} DEST={pdu.get('dst', 0)}"
        else:
            party = ""
        return f"{prefix} FRAME=LDU1{party}{nac}{detail}{fo_str}"

    if pdu.get("type") == "P25_LDU2":
        return f"{prefix} FRAME=LDU2{nac}{detail}{fo_str}"

    if pdu.get("type") == "P25_CALL":
        call = "GROUP" if pdu.get("flco") == "GROUP" else "UNIT"
        if call == "GROUP":
            party = f" SRC={pdu.get('src', 0)} TGID={pdu.get('dst', 0)}"
        else:
            party = f" SRC={pdu.get('src', 0)} DEST={pdu.get('dst', 0)}"
        duration = f" DUR={extra.get('duration_s')}s" if "duration_s" in extra else ""
        ldu_count = f" LDUS={extra.get('ldu_count')}" if "ldu_count" in extra else ""
        return f"{prefix} CALL={call}{party}{nac}{duration}{ldu_count}{fo_str}"

    frame = pdu.get("flco", extra.get("duid_name", ""))
    return f"{prefix} FRAME={frame}{nac}{detail}{fo_str}"


def _p25_detail(pdu: dict) -> str:
    extra = pdu.get("extra", {})
    base = (
        f" DUID=0x{extra['duid']:X} BCH={'OK' if extra.get('valid_bch') else 'FAIL'}"
        f" CORR={int(bool(extra.get('corrected')))}"
        if "duid" in extra
        else ""
    )
    if pdu.get("type") == "P25_HDU":
        return (
            f"{base} MI=0x{extra.get('mi', 0):018X}"
            f" MFID=0x{extra.get('hdu_mfid', 0):02X}"
            f" ALGID=0x{extra.get('algid', 0):02X}"
            f" KID=0x{extra.get('kid', 0):04X}"
            f" TGID={extra.get('hdu_tgid', 0)}"
        )
    if pdu.get("type") == "P25_LDU1":
        call_type = extra.get("call_type", "")
        if call_type == "group":
            lc_fields = (
                f" LCW16=0x{extra.get('lc_info', 0):04X}"
                f" EMERGENCY={int(bool(extra.get('lc_emergency')))}"
                f" RESERVED{extra.get('lc_reserved_bits', 0)}=0x{extra.get('lc_reserved', 0):04X}"
            )
        elif call_type == "unit_to_unit":
            lc_fields = (
                f" LCW16=0x{extra.get('lc_info', 0):04X}"
                f" RESERVED{extra.get('lc_reserved_bits', 0)}=0x{extra.get('lc_reserved', 0):02X}"
            )
        else:
            lc_fields = (
                f" LCW16=0x{extra.get('lc_info', 0):04X}"
                f" RESERVED{extra.get('lc_reserved_bits', 0)}=0x{extra.get('lc_reserved', 0):04X}"
            )
        return (
            f"{base} LCF=0x{extra.get('lco', 0):02X}"
            f" MFID=0x{extra.get('mfid', 0):02X}"
            f" CALL={call_type}"
            f"{lc_fields}"
        )
    if pdu.get("type") == "P25_LDU2":
        return (
            f"{base} ES_MI=0x{extra.get('es_mi', 0):018X}"
            f" ES_ALGID=0x{extra.get('es_algid', 0):02X}"
            f" ES_KID=0x{extra.get('es_kid', 0):04X}"
        )
    return base


PROTOCOL_REGISTRY: tuple[ProtocolSpec, ...] = (
    ProtocolSpec(
        "DMR",
        ("dmr",),
        "decode_dmr",
        DEFAULT_DMR_CONFIG,
        "c4fm_4fsk",
        _frontend_dmr_p25,
        _postprocess_identity,
        _dmr_dedup_key,
        format_dmr_pdu,
    ),
    ProtocolSpec(
        "P25",
        ("p25",),
        "decode_p25",
        DEFAULT_P25_CONFIG,
        "c4fm_4fsk",
        _frontend_dmr_p25,
        _postprocess_identity,
        _p25_dedup_key,
        format_p25_pdu,
    ),
    ProtocolSpec(
        "dPMR",
        ("dpmr",),
        "decode_dpmr",
        DEFAULT_DPMR_CONFIG,
        "dpmr_4fsk",
        _frontend_dpmr,
        _filter_stable_dpmr_pdus,
        _dpmr_dedup_key,
        format_dpmr_pdu,
    ),
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
            frontends[spec.frontend_key] = spec.frontend(iq_dec, sample_rate)
        decoder = globals()[spec.decode_name]
        results.extend(decoder(frontends[spec.frontend_key]))
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
