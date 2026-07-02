from __future__ import annotations

import numpy as np

from dmr.dsp import frontend as _frontend_c4fm
from p25.config import DEFAULT_P25_CONFIG
from p25.decoder import decode as _decode_p25
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


def dedup_key(pdu: dict) -> tuple:
    extra = pdu.get("extra", {})
    frame_bucket = round(extra.get("fs_start", 0) / DEFAULT_P25_CONFIG.dedup_frame_bucket_samples)
    return ("P25", extra.get("nac"), pdu.get("type", ""), frame_bucket)


def format_pdu(pdu: dict, fo_str: str = "") -> str:
    extra = pdu.get("extra", {})
    prefix = f"[{pdu['type']:<12}] PROTO=P25"
    nac = f" NAC=0x{extra['nac']:03X}" if "nac" in extra else ""
    detail = p25_detail(pdu)

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


def p25_detail(pdu: dict) -> str:
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


def decode(y: np.ndarray, config: object | None = None) -> list[dict]:
    if config is None:
        config = DEFAULT_P25_CONFIG
    return _decode_p25(
        y,
        sps=config.samples_per_symbol,
        sync_threshold=config.sync_threshold,
    )


SPEC = ProtocolSpec(
    "P25",
    ("p25",),
    "decode_p25",
    DEFAULT_P25_CONFIG,
    "c4fm_4fsk",
    frontend,
    postprocess_identity,
    dedup_key,
    format_pdu,
)
