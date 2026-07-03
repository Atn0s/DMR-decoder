from __future__ import annotations

import numpy as np

from dmr.config import DEFAULT_DMR_CONFIG
import dmr.engine as dmr_engine
from dmr.dsp import frontend as _frontend_c4fm
from radio.pdu import pdus_to_standard_dicts
from radio.protocol import ProtocolSpec, call_decoder, postprocess_identity


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
    return (
        f"[{pdu['type']:<12}] PROTO={proto} SRC={pdu['src']} DST={pdu['dst']} "
        f"FLCO={pdu['flco']} FID={pdu.get('fid','')}{fo_str}"
    )


def _dmr_decode_loop(y: np.ndarray, config: object | None = None) -> list[dict]:
    return dmr_engine._decode_dmr_loop(y, config)


def decode(y: np.ndarray, config: object | None = None) -> list[dict]:
    if config is None:
        config = DEFAULT_DMR_CONFIG
    pdus = pdus_to_standard_dicts(call_decoder(_dmr_decode_loop, y, config))
    for pdu in pdus:
        pdu.setdefault("protocol", "DMR")
    return pdus


SPEC = ProtocolSpec(
    "DMR",
    ("dmr",),
    "decode_dmr",
    DEFAULT_DMR_CONFIG,
    "c4fm_4fsk",
    frontend,
    postprocess_identity,
    dedup_key,
    format_pdu,
)
