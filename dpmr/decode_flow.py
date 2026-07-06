from __future__ import annotations

from dataclasses import replace

import numpy as np

from dpmr.cch import CCHRecord
from dpmr.config import DEFAULT_DPMR_CONFIG, DPMRConfig
from dpmr.constants import DPMR_FRAME_SYMBOLS, FS1_SYMBOLS
from dpmr.link_layer import (
    DPMRHeaderDecode,
    DPMRVoiceDecode,
    cch_extra,
    decode_header_payload,
    decode_voice_symbols,
)
from dpmr.dsp import (
    find_fs1_sync,
    find_fs2_sync,
    recover_frame_symbol_candidates,
    recover_voice_fs2_symbol_candidates,
)
from dpmr.session import DPMRSessionAssembler, cch_record_usable


def _quality_score(cch0: CCHRecord | None, cch1: CCHRecord | None, color_code: int) -> float:
    records = [rec for rec in (cch0, cch1) if rec is not None]
    score = 0.0
    score += 2.0 if color_code >= 0 else 0.0
    score += 5.0 * sum(1 for rec in records if rec.crc_ok)
    score += 2.0 * sum(1 for rec in records if rec.hamming_ok)
    frames = {rec.frame_number for rec in records}
    if {0, 1}.issubset(frames) or {2, 3}.issubset(frames):
        score += 3.0
    return score


def _candidate_score(decoded: DPMRVoiceDecode, resid: float) -> float:
    return _quality_score(decoded.cch0, decoded.cch1, decoded.color_code) - 0.2 * resid


def _header_score(decoded: DPMRHeaderDecode, resid: float) -> float:
    score = 0.0
    score += 8.0 * sum(1 for rec in decoded.cch_records if rec.crc_ok)
    score += 2.0 * sum(1 for rec in decoded.cch_records if rec.hamming_ok)
    score += 2.0 if decoded.color_codes else 0.0
    frames = {rec.frame_number for rec in decoded.cch_records if cch_record_usable(rec)}
    if {0, 1}.issubset(frames) or {2, 3}.issubset(frames):
        score += 3.0
    return score - 0.2 * resid


def _sync_error_phase_search(config: DPMRConfig) -> np.ndarray:
    return np.linspace(
        config.sync_error_phase_min,
        config.sync_error_phase_max,
        config.sync_error_phase_steps,
    )


def _symbol_phase_search(config: DPMRConfig) -> np.ndarray:
    return np.linspace(
        config.phase_search_min,
        config.phase_search_max,
        config.phase_search_steps,
    )


def _sps_search(config: DPMRConfig) -> np.ndarray:
    return np.linspace(
        config.sps_search_min,
        config.sps_search_max,
        config.sps_search_steps,
    )


def _candidate_timing(symbol_candidate) -> dict:
    return {
        "sps": symbol_candidate.sps,
        "phase": symbol_candidate.phase,
        "resid": symbol_candidate.resid,
        "sample_window": symbol_candidate.sample_window,
        "decision_error_p90": symbol_candidate.decision_error_p90,
        "ambiguous_symbols": symbol_candidate.ambiguous_symbols,
    }


def _decode_header_frame(y: np.ndarray, config: DPMRConfig) -> list[dict]:
    results: list[dict] = []
    seen: set[int] = set()
    sync_candidates = find_fs1_sync(
        y,
        threshold=config.sync_threshold,
        max_symbol_errors=config.sync_max_symbol_errors,
        min_distance_samples=config.sync_min_distance_samples,
        dedup_window_symbols=config.sync_dedup_window_symbols,
        sync_error_phase_search=_sync_error_phase_search(config),
    )
    if len(sync_candidates) > config.header_sync_candidate_limit:
        sync_candidates = sorted(
            sync_candidates,
            key=lambda item: item.ncc,
            reverse=True,
        )[:config.header_sync_candidate_limit]
        sync_candidates.sort(key=lambda item: item.fs_start)

    for candidate in sync_candidates:
        bucket = round(candidate.fs_start / 240)
        if bucket in seen:
            continue
        seen.add(bucket)

        symbol_candidates = recover_frame_symbol_candidates(
            y,
            candidate,
            total_symbols=DPMR_FRAME_SYMBOLS,
            phase_search=_symbol_phase_search(config),
            sps_search=_sps_search(config),
            sample_windows=config.sample_windows,
            limit=config.header_symbol_candidate_limit,
            decision_ambiguous_threshold=config.decision_ambiguous_threshold,
        )
        if not symbol_candidates:
            continue

        best = None
        for symbol_candidate in symbol_candidates:
            payload = symbol_candidate.symbols[len(FS1_SYMBOLS):]
            decoded = decode_header_payload(payload)
            if decoded is None:
                continue
            item = (
                _header_score(decoded, symbol_candidate.resid),
                -symbol_candidate.resid,
                symbol_candidate,
                decoded,
            )
            if best is None or item[:2] > best[:2]:
                best = item

        if best is None:
            continue

        _, _, symbol_candidate, decoded = best
        timing = _candidate_timing(symbol_candidate)
        results.append(
            {
                "protocol": "dPMR",
                "type": "DPMR_HEADER",
                "src": decoded.src,
                "dst": decoded.dst,
                "ts": 0,
                "flco": "HEADER",
                "fid": "",
                "extra": {
                    "color_code": decoded.color_code,
                    "sync_type": "FS1",
                    "polarity_inverted": candidate.polarity_inverted,
                    "sync_ncc": candidate.ncc,
                    "symbol_sps": timing["sps"],
                    "symbol_phase": timing["phase"],
                    "symbol_resid": timing["resid"],
                    "symbol_sample_window": timing["sample_window"],
                    "segment_timing": {"header": timing},
                    "fs_start": candidate.fs_start,
                    "superframe_part": decoded.superframe_part,
                    "quality": decoded.quality,
                    "cch": [cch_extra(record) for record in decoded.cch_records],
                    "cch_offsets": decoded.cch_offsets,
                    "color_code_candidates": decoded.color_codes,
                    "color_code_offsets": decoded.color_offsets,
                    "frame_numbers": [record.frame_number for record in decoded.cch_records],
                },
                "raw_bits": decoded.raw_bits,
            }
        )
    return results


def filter_stable_pdus(pdus: list[dict], min_repeats: int = 2) -> list[dict]:
    dpmr_pdus = [pdu for pdu in pdus if pdu.get("protocol") == "dPMR"]
    if len(dpmr_pdus) < min_repeats:
        return pdus

    color_counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for index, pdu in enumerate(dpmr_pdus):
        color_code = pdu.get("extra", {}).get("color_code", -1)
        if color_code < 0:
            continue
        color_counts[color_code] = color_counts.get(color_code, 0) + 1
        first_seen.setdefault(color_code, index)
    if not color_counts:
        return pdus

    quality_weight = {"high": 10, "medium": 6, "low": 1, "none": 0}
    color_quality_scores: dict[int, int] = {}
    for pdu in dpmr_pdus:
        color_code = pdu.get("extra", {}).get("color_code", -1)
        if color_code < 0:
            continue
        quality = pdu.get("extra", {}).get("quality", {})
        confidence = quality.get("front_end_confidence", quality.get("confidence", "none"))
        color_quality_scores[color_code] = (
            color_quality_scores.get(color_code, 0)
            + quality_weight.get(confidence, 0)
        )

    stable_color, repeats = max(
        color_counts.items(),
        key=lambda item: (
            item[1],
            -first_seen[item[0]],
            color_quality_scores.get(item[0], 0),
        ),
    )
    if repeats < min_repeats:
        return pdus

    stable_pdus = [
        pdu for pdu in dpmr_pdus
        if pdu.get("extra", {}).get("color_code") == stable_color
    ]
    has_high_or_medium = any(
        pdu.get("extra", {}).get("quality", {}).get(
            "front_end_confidence",
            pdu.get("extra", {}).get("quality", {}).get("confidence"),
        ) in ("high", "medium")
        for pdu in stable_pdus
    )

    filtered: list[dict] = []
    for pdu in pdus:
        if pdu.get("protocol") != "dPMR":
            filtered.append(pdu)
            continue
        extra = pdu.get("extra", {})
        if extra.get("color_code") == stable_color and (
            not has_high_or_medium
            or extra.get("quality", {}).get(
                "front_end_confidence",
                extra.get("quality", {}).get("confidence"),
            ) in ("high", "medium")
        ):
            extra["stable_color_code"] = stable_color
            extra["stable_color_repeats"] = repeats
            filtered.append(pdu)
    return filtered


def decode(
    y: np.ndarray,
    sync_threshold: float | None = None,
    config: DPMRConfig | None = None,
) -> list[dict]:
    config = config or DEFAULT_DPMR_CONFIG
    if sync_threshold is not None:
        config = replace(config, sync_threshold=sync_threshold)

    results: list[dict] = _decode_header_frame(y, config)
    session = DPMRSessionAssembler()
    seen: set[int] = set()
    sync_candidates = find_fs2_sync(
        y,
        threshold=config.sync_threshold,
        max_symbol_errors=config.sync_max_symbol_errors,
        min_distance_samples=config.sync_min_distance_samples,
        dedup_window_symbols=config.sync_dedup_window_symbols,
        sync_error_phase_search=_sync_error_phase_search(config),
    )
    if len(sync_candidates) > config.voice_sync_candidate_limit:
        sync_candidates = sorted(
            sync_candidates,
            key=lambda item: item.ncc,
            reverse=True,
        )[:config.voice_sync_candidate_limit]
        sync_candidates.sort(key=lambda item: item.fs_start)
    for candidate in sync_candidates:
        bucket = round(candidate.fs_start / 240)
        if bucket in seen:
            continue
        seen.add(bucket)

        symbol_candidates = recover_voice_fs2_symbol_candidates(
            y,
            candidate,
            phase_search=_symbol_phase_search(config),
            sps_search=_sps_search(config),
            sample_windows=config.sample_windows,
            limit=config.voice_symbol_candidate_limit,
            decision_ambiguous_threshold=config.decision_ambiguous_threshold,
        )
        if not symbol_candidates:
            continue
        best = None
        for symbol_candidate in symbol_candidates:
            decoded = decode_voice_symbols(symbol_candidate.symbols)
            if decoded is None:
                continue
            timing = _candidate_timing(symbol_candidate)
            item = (
                _candidate_score(decoded, symbol_candidate.resid),
                -symbol_candidate.resid,
                decoded,
                timing,
            )
            if best is None or item[:2] > best[:2]:
                best = item

        if best is None:
            continue
        _, _, decoded, timing = best
        quality = dict(decoded.quality)
        quality["timing_coherent"] = True
        quality["front_end_confidence"] = quality["confidence"]
        src, dst, superframe_part = session.feed(decoded.cch0, decoded.cch1)
        expose_ids = quality["confidence"] == "high" and superframe_part in ("src", "dst")
        src_out = src if expose_ids else ""
        dst_out = dst if expose_ids else ""
        results.append(
            {
                "protocol": "dPMR",
                "type": "DPMR_VOICE",
                "src": src_out,
                "dst": dst_out,
                "ts": 0,
                "flco": "VOICE",
                "fid": "",
                "extra": {
                    "color_code": decoded.color_code,
                    "sync_type": "FS2",
                    "polarity_inverted": candidate.polarity_inverted,
                    "sync_ncc": candidate.ncc,
                    "symbol_sps": timing["sps"],
                    "symbol_phase": timing["phase"],
                    "symbol_resid": timing["resid"],
                    "symbol_sample_window": timing["sample_window"],
                    "segment_timing": {
                        "cch0": timing,
                        "cc": timing,
                        "cch1": timing,
                    },
                    "fs_start": candidate.fs_start,
                    "superframe_part": superframe_part,
                    "quality": quality,
                    "cch": [cch_extra(decoded.cch0), cch_extra(decoded.cch1)],
                    "frame_numbers": [
                        rec.frame_number for rec in (decoded.cch0, decoded.cch1)
                        if rec is not None
                    ],
                },
                "raw_bits": decoded.raw_bits,
            }
        )
    return results
