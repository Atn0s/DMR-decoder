from __future__ import annotations

import numpy as np

from dmr.config import DEFAULT_DMR_CONFIG
import dmr.decode_flow as dmr_decode_flow
from dmr.dsp import frontend as _frontend_c4fm
from radio.pdu import pdus_to_standard_dicts
from radio.protocol import ProtocolSpec, postprocess_identity


def frontend(iq_dec: np.ndarray, sample_rate: float, config: object) -> np.ndarray:
    return _frontend_c4fm(
        iq_dec,
        fo=0.0,
        fs=sample_rate,
        cutoff=config.frontend_cutoff_hz,
        ntaps=config.frontend_taps,
        dev_nominal=config.nominal_deviation_hz,
        min_samples=config.frontend_min_samples,
        psd_nperseg=config.frontend_psd_nperseg,
    )


def _canonical_dmr_protocol(proto: object) -> str:
    return "DMR" if str(proto).lower() == "dmr" else str(proto)


def dedup_key(pdu: dict) -> tuple:
    bucket_hz = DEFAULT_DMR_CONFIG.dedup_frequency_bucket_hz
    fo_bucket = round(pdu.get("_fo_hz", 0) / bucket_hz) * bucket_hz
    return (
        _canonical_dmr_protocol(pdu.get("protocol", "DMR")),
        pdu.get("src", 0),
        pdu.get("dst", 0),
        pdu.get("type", ""),
        fo_bucket,
    )


def format_pdu(pdu: dict, fo_str: str = "") -> str:
    proto = pdu.get("protocol", "DMR")
    ptype = pdu.get("type", "")
    extra = pdu.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}

    if ptype == "DMR_CALL":
        return _format_dmr_call(pdu, extra, fo_str)

    base = f"[{ptype:<12}] PROTO={proto}"
    party = _format_party(pdu, extra)
    cc = _format_color_code(extra)
    dt = _format_data_type(extra)
    fec = _format_fec(extra)
    flco = _format_flco(pdu, extra)
    details = _format_detail(ptype, extra)

    if not any((party, cc, dt, fec, details)) and not extra:
        return (
            f"[{pdu['type']:<12}] PROTO={proto} SRC={pdu['src']} DST={pdu['dst']} "
            f"FLCO={pdu['flco']} FID={pdu.get('fid','')}{fo_str}"
        )

    return f"{base}{party}{cc}{dt}{flco}{fec}{details}{fo_str}"


def _format_dmr_call(pdu: dict, extra: dict, fo_str: str) -> str:
    call_type = str(extra.get("call_type", "unknown")).upper()
    party = _format_party(pdu, {"flc": {"call_type": extra.get("call_type")}})
    cc = _format_color_code(extra)
    duration = f" DUR={extra.get('duration_s')}s" if "duration_s" in extra else ""
    counts = (
        f" SIG={extra.get('signalling_count', 0)}"
        f" LATE={extra.get('late_entry_count', 0)}"
        f" CSBK={extra.get('csbk_count', 0)}"
    )
    closed = f" CLOSED={extra.get('closed_by')}" if extra.get("closed_by") else ""
    return (
        f"[{pdu['type']:<12}] PROTO=DMR CALL={call_type}{party}{cc}"
        f" FLCO={pdu.get('flco', '')} FID={pdu.get('fid', '')}"
        f"{duration}{counts}{closed}{fo_str}"
    )


def _format_party(pdu: dict, extra: dict) -> str:
    flc = extra.get("flc", {})
    call_type = flc.get("call_type") if isinstance(flc, dict) else extra.get("call_type")
    src = pdu.get("src", 0)
    dst = pdu.get("dst", 0)
    if call_type == "group":
        return f" SRC={src} TGID={dst}"
    if call_type == "unit_to_unit":
        return f" SRC={src} DEST={dst}"
    if src or dst:
        return f" SRC={src} DST={dst}"
    return ""


def _format_color_code(extra: dict) -> str:
    color_code = extra.get("color_code")
    return f" CC={color_code}" if color_code is not None else ""


def _format_data_type(extra: dict) -> str:
    name = extra.get("data_type_name")
    value = extra.get("data_type")
    if name is None and value is None:
        return ""
    if value is None:
        return f" DT={name}"
    return f" DT={value}:{name}"


def _format_flco(pdu: dict, extra: dict) -> str:
    flc = extra.get("flc", {})
    csbk = extra.get("csbk", {})
    fid = pdu.get("fid", "")
    if isinstance(flc, dict) and flc:
        flco_value = flc.get("flco_value", 0)
        fid_value = flc.get("fid_value", 0)
        svc = _format_service_options(flc)
        return f" FLCO=0x{flco_value:02X}({pdu.get('flco', '')}) FID=0x{fid_value:02X}({fid}){svc}"
    if isinstance(csbk, dict) and csbk:
        csbko_value = csbk.get("csbko_value", 0)
        fid_value = csbk.get("fid_value", 0)
        svc = _format_service_options(csbk)
        return f" CSBKO=0x{csbko_value:02X}({pdu.get('flco', '')}) FID=0x{fid_value:02X}({fid}){svc}"
    return f" FLCO={pdu.get('flco', '')} FID={fid}"


def _format_service_options(fields: dict) -> str:
    svc = fields.get("service_options")
    if not isinstance(svc, dict):
        return ""
    flags = []
    if svc.get("emergency"):
        flags.append("EMERGENCY")
    if svc.get("privacy"):
        flags.append("PRIVACY")
    if svc.get("broadcast"):
        flags.append("BROADCAST")
    if svc.get("open_voice_call_mode"):
        flags.append("OVCM")
    flag_text = ",".join(flags) if flags else "-"
    return f" SVC=0x{fields.get('service_options_value', 0):02X}[{flag_text},PRI={svc.get('priority', 0)}]"


def _format_fec(extra: dict) -> str:
    fec = extra.get("fec", {})
    if not isinstance(fec, dict) or not fec:
        return ""
    parts = []
    if "golay_ok" in fec:
        parts.append(f"GOLAY={'OK' if fec.get('golay_ok') else 'FAIL'}")
    if "bptc_196_96_ok" in fec:
        parts.append(f"BPTC={'OK' if fec.get('bptc_196_96_ok') else 'FAIL'}")
    if "rs_12_9_4_ok" in fec:
        parts.append(f"RS={'OK' if fec.get('rs_12_9_4_ok') else 'FAIL'}")
    if "vbptc_128_72_ok" in fec:
        parts.append(f"VBPTC={'OK' if fec.get('vbptc_128_72_ok') else 'FAIL'}")
    if "cs5_ok" in fec:
        parts.append(f"CS5={'OK' if fec.get('cs5_ok') else 'FAIL'}")
    if "emb_qr_ok_count" in fec:
        parts.append(f"EMB_QR={fec.get('emb_qr_ok_count')}/{extra.get('fragment_count', 0)}")
    return " FEC=[" + ",".join(parts) + "]" if parts else ""


def _format_detail(ptype: str, extra: dict) -> str:
    if ptype == "CSBK":
        csbk = extra.get("csbk", {})
        if not isinstance(csbk, dict):
            return ""
        parts = [f"LB={int(bool(csbk.get('last_block')))}", f"PF={int(bool(csbk.get('protect_flag')))}"]
        for key, label in (
            ("blocks_to_follow", "BTF"),
            ("answer_response_name", "ANS"),
            ("reason_code", "REASON"),
            ("sync_age", "SYNC_AGE"),
            ("system_identity_code", "SYSID"),
            ("nrand_wait", "NRAND"),
            ("tscc_backoff", "BACKOFF"),
        ):
            if key in csbk and csbk.get(key) not in (None, ""):
                parts.append(f"{label}={csbk.get(key)}")
        return " " + " ".join(parts)

    if ptype == "LATE_ENTRY":
        return f" FRAGS={extra.get('fragment_count', 0)}"

    sample = extra.get("fs_start")
    return f" SAMPLE={sample}" if sample is not None else ""


def decode(y: np.ndarray, config: object | None = None) -> list[dict]:
    if config is None:
        config = DEFAULT_DMR_CONFIG
    pdus = pdus_to_standard_dicts(dmr_decode_flow.decode_dmr_flow(y, config))
    for pdu in pdus:
        pdu.setdefault("protocol", "DMR")
    return pdus


SPEC = ProtocolSpec(
    "DMR",
    ("dmr",),
    DEFAULT_DMR_CONFIG,
    "c4fm_4fsk",
    frontend,
    decode,
    postprocess_identity,
    dedup_key,
    format_pdu,
)
