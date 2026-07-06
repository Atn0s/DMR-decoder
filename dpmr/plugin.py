from __future__ import annotations

import numpy as np

from dpmr.config import DEFAULT_DPMR_CONFIG
from dpmr.decode_flow import decode as _decode_dpmr
from dpmr.decode_flow import filter_stable_pdus
from dpmr.dsp import frontend_dpmr
from radio.pdu import pdus_to_standard_dicts
from radio.protocol import ProtocolSpec


def frontend(iq_dec: np.ndarray, sample_rate: float, config: object) -> np.ndarray:
    return frontend_dpmr(
        iq_dec,
        fs=sample_rate,
        cutoff=config.frontend_cutoff_hz,
        ntaps=config.frontend_taps,
        dev_nominal=config.nominal_deviation_hz,
        min_samples=config.frontend_min_samples,
        psd_nperseg=config.frontend_psd_nperseg,
    )


def dedup_key(pdu: dict) -> tuple:
    extra = pdu.get("extra", {})
    frame_bucket = round(extra.get("fs_start", 0) / DEFAULT_DPMR_CONFIG.dedup_frame_bucket_samples)
    return (
        "dPMR",
        pdu.get("src", ""),
        pdu.get("dst", ""),
        extra.get("color_code"),
        frame_bucket,
    )


def postprocess(pdus: list[dict]) -> list[dict]:
    return filter_stable_pdus(
        pdus,
        min_repeats=DEFAULT_DPMR_CONFIG.stable_color_min_repeats,
    )


def format_pdu(pdu: dict, fo_str: str = "") -> str:
    extra = pdu.get("extra", {})
    color_code = extra.get("color_code", -1)
    pol = "INV" if extra.get("polarity_inverted") else "NORM"
    sync_type = extra.get("sync_type", "")
    src = pdu.get("src") or ""
    dst = pdu.get("dst") or ""
    cc_text = f"{color_code:02d}" if isinstance(color_code, int) and color_code >= 0 else "--"
    cch_text = format_cch(extra.get("cch", []))
    return (
        f"[{pdu['type']:<12}] PROTO=dPMR SRC={src} DST={dst} "
        f"CC={cc_text} SYNC={sync_type} POL={pol}{cch_text}{fo_str}"
    )


def format_cch(cch_records: list[dict | None]) -> str:
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


def decode(y: np.ndarray, config: object | None = None) -> list[dict]:
    if config is None:
        config = DEFAULT_DPMR_CONFIG
    return pdus_to_standard_dicts(_decode_dpmr(y, config=config))


SPEC = ProtocolSpec(
    "dPMR",
    ("dpmr",),
    DEFAULT_DPMR_CONFIG,
    "dpmr_4fsk",
    frontend,
    decode,
    postprocess,
    dedup_key,
    format_pdu,
)
