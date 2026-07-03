from __future__ import annotations

from dataclasses import replace

import numpy as np

from dpmr.cch import CCHRecord, decode_cch
from dpmr.color_code import get_color_code
from dpmr.config import DEFAULT_DPMR_CONFIG, DPMRConfig
from dpmr.constants import CC_SYMBOLS, CCH_SYMBOLS, DPMR_FRAME_SYMBOLS, FS1_SYMBOLS
from dpmr.dsp import (
    find_fs1_sync,
    find_fs2_sync,
    recover_frame_symbol_candidates,
    recover_voice_fs2_symbol_candidates,
    split_voice_fs2,
    symbols_to_bits,
)
from dpmr.session import DPMRSessionAssembler, cch_record_usable


def _cch_extra(record: CCHRecord | None) -> dict | None:
    if record is None:
        return None
    return {
        "frame_number": record.frame_number,
        "id_half": record.id_half,
        "communication_mode": record.communication_mode,
        "version": record.version,
        "comms_format": record.comms_format,
        "emergency_priority": record.emergency_priority,
        "reserved": record.reserved,
        "slow_data": record.slow_data,
        "crc_ok": record.crc_ok,
        "crc_value": record.crc_value,
        "crc_computed": record.crc_computed,
        "hamming_ok": record.hamming_ok,
        "hamming_blocks_ok": record.hamming_blocks_ok,
        "corrected_bits": record.corrected_bits,
    }


def _quality_ok(cch0: CCHRecord | None, cch1: CCHRecord | None, color_code: int) -> bool:
    records = [rec for rec in (cch0, cch1) if rec is not None]
    if color_code >= 0 and any(rec.crc_ok for rec in records):
        return True
    return color_code >= 0 and any(rec.hamming_ok for rec in records)


def _records_quality(records: list[CCHRecord]) -> dict:
    usable = [rec for rec in records if cch_record_usable(rec)]
    crc_ok_count = sum(1 for rec in records if rec.crc_ok)
    hamming_ok_count = sum(1 for rec in records if rec.hamming_ok)
    frames = {rec.frame_number for rec in usable}
    valid_pair = {0, 1}.issubset(frames) or {2, 3}.issubset(frames)
    if crc_ok_count:
        confidence = "high"
    elif valid_pair:
        confidence = "medium"
    elif hamming_ok_count:
        confidence = "low"
    else:
        confidence = "none"
    return {
        "crc_ok_count": crc_ok_count,
        "hamming_ok_count": hamming_ok_count,
        "valid_frame_pair": valid_pair,
        "confidence": confidence,
        "front_end_confidence": confidence,
    }


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


def _decision_penalty(timing: dict) -> float:
    return 0.5 * float(timing.get("decision_error_p90", 0.0)) + 0.02 * float(timing.get("ambiguous_symbols", 0))


def _cch_score(record: CCHRecord | None, resid: float, timing: dict | None = None) -> float:
    if record is None:
        return -1e9
    score = 0.0
    score += 10.0 if record.crc_ok else 0.0
    score += 3.0 if record.hamming_ok else 0.0
    score += sum(1 for ok in record.hamming_blocks_ok if ok) / 6.0
    score -= 0.2 * resid
    if timing is not None:
        score -= _decision_penalty(timing)
    return score


def _cc_score(color_code: int, resid: float, timing: dict | None = None) -> float:
    score = 4.0 if color_code >= 0 else 0.0
    score -= 0.2 * resid
    if timing is not None:
        score -= _decision_penalty(timing)
    return score


def _quality_summary(cch0: CCHRecord | None, cch1: CCHRecord | None) -> dict:
    records = [rec for rec in (cch0, cch1) if rec is not None]
    frames = {rec.frame_number for rec in records if cch_record_usable(rec)}
    crc_ok_count = sum(1 for rec in records if rec.crc_ok)
    hamming_ok_count = sum(1 for rec in records if rec.hamming_ok)
    valid_pair = {0, 1}.issubset(frames) or {2, 3}.issubset(frames)
    if crc_ok_count:
        confidence = "high"
    elif valid_pair:
        confidence = "medium"
    elif hamming_ok_count:
        confidence = "low"
    else:
        confidence = "none"
    return {
        "crc_ok_count": crc_ok_count,
        "hamming_ok_count": hamming_ok_count,
        "valid_frame_pair": valid_pair,
        "confidence": confidence,
    }


def _timing_coherent(*timings: dict) -> bool:
    sps_values = [float(timing["sps"]) for timing in timings]
    phase_values = [float(timing["phase"]) for timing in timings]
    windows = {int(timing["sample_window"]) for timing in timings}
    return (
        max(sps_values) - min(sps_values) <= 0.3
        and max(phase_values) - min(phase_values) <= 3.0
        and len(windows) == 1
    )


def _front_end_confidence(quality: dict, coherent: bool) -> str:
    if not coherent:
        return "diagnostic"
    return quality["confidence"]


def _candidate_score(
    cch0: CCHRecord | None,
    cch1: CCHRecord | None,
    color_code: int,
    resid: float,
) -> float:
    return _quality_score(cch0, cch1, color_code) - 0.2 * resid


def _raw_bytes(bits: list[int]) -> bytes:
    return bytes(
        int("".join(str(bit) for bit in bits[i:i + 8]).ljust(8, "0"), 2)
        for i in range(0, len(bits), 8)
    )


def _assemble_ids_from_records(records: list[CCHRecord]) -> tuple[str, str, str]:
    session = DPMRSessionAssembler()
    src = ""
    dst = ""
    part = "unknown"
    for idx in range(0, len(records), 2):
        a = records[idx]
        b = records[idx + 1] if idx + 1 < len(records) else None
        src, dst, part = session.feed(a, b)
    return src, dst, part


def _header_score(records: list[CCHRecord], color_codes: list[int], resid: float) -> float:
    score = 0.0
    score += 8.0 * sum(1 for rec in records if rec.crc_ok)
    score += 2.0 * sum(1 for rec in records if rec.hamming_ok)
    score += 2.0 if color_codes else 0.0
    frames = {rec.frame_number for rec in records if cch_record_usable(rec)}
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
            cch_records: list[CCHRecord] = []
            cch_offsets: list[int] = []
            for offset in range(0, len(payload) - CCH_SYMBOLS + 1, CCH_SYMBOLS):
                record = decode_cch(symbols_to_bits(payload[offset:offset + CCH_SYMBOLS]))
                if record is not None and cch_record_usable(record):
                    cch_records.append(record)
                    cch_offsets.append(offset)

            color_codes: list[int] = []
            color_offsets: list[int] = []
            for offset in range(0, len(payload) - CC_SYMBOLS + 1, CC_SYMBOLS):
                color_code = get_color_code(symbols_to_bits(payload[offset:offset + CC_SYMBOLS]))
                if color_code >= 0:
                    color_codes.append(color_code)
                    color_offsets.append(offset)

            if not cch_records and not color_codes:
                continue

            item = (
                _header_score(cch_records, color_codes, symbol_candidate.resid),
                -symbol_candidate.resid,
                symbol_candidate,
                cch_records,
                cch_offsets,
                color_codes,
                color_offsets,
            )
            if best is None or item[:2] > best[:2]:
                best = item

        if best is None:
            continue

        _, _, symbol_candidate, cch_records, cch_offsets, color_codes, color_offsets = best
        payload = symbol_candidate.symbols[len(FS1_SYMBOLS):]
        payload_bits = symbols_to_bits(payload)
        quality = _records_quality(cch_records)
        src, dst, superframe_part = _assemble_ids_from_records([rec for rec in cch_records if rec.crc_ok])
        if quality["confidence"] != "high":
            src = ""
            dst = ""
        color_code = color_codes[0] if color_codes else -1
        timing = {
            "sps": symbol_candidate.sps,
            "phase": symbol_candidate.phase,
            "resid": symbol_candidate.resid,
            "sample_window": symbol_candidate.sample_window,
            "decision_error_p90": symbol_candidate.decision_error_p90,
            "ambiguous_symbols": symbol_candidate.ambiguous_symbols,
        }
        results.append(
            {
                "protocol": "dPMR",
                "type": "DPMR_HEADER",
                "src": src,
                "dst": dst,
                "ts": 0,
                "flco": "HEADER",
                "fid": "",
                "extra": {
                    "color_code": color_code,
                    "sync_type": "FS1",
                    "polarity_inverted": candidate.polarity_inverted,
                    "sync_ncc": candidate.ncc,
                    "symbol_sps": timing["sps"],
                    "symbol_phase": timing["phase"],
                    "symbol_resid": timing["resid"],
                    "symbol_sample_window": timing["sample_window"],
                    "segment_timing": {"header": timing},
                    "fs_start": candidate.fs_start,
                    "superframe_part": superframe_part,
                    "quality": quality,
                    "cch": [_cch_extra(record) for record in cch_records],
                    "cch_offsets": cch_offsets,
                    "color_code_candidates": color_codes,
                    "color_code_offsets": color_offsets,
                    "frame_numbers": [record.frame_number for record in cch_records],
                },
                "raw_bits": _raw_bytes(payload_bits),
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
            symbols = symbol_candidate.symbols
            resid = symbol_candidate.resid
            cch0_bits, cc_bits, cch1_bits = split_voice_fs2(symbols)
            cch0 = decode_cch(cch0_bits)
            cch1 = decode_cch(cch1_bits)
            color_code = get_color_code(cc_bits)

            timing = {
                "sps": symbol_candidate.sps,
                "phase": symbol_candidate.phase,
                "resid": resid,
                "sample_window": symbol_candidate.sample_window,
                "decision_error_p90": symbol_candidate.decision_error_p90,
                "ambiguous_symbols": symbol_candidate.ambiguous_symbols,
            }
            if not _quality_ok(cch0, cch1, color_code):
                continue
            item = (
                _candidate_score(cch0, cch1, color_code, resid),
                -resid,
                cch0_bits,
                cc_bits,
                cch1_bits,
                cch0,
                color_code,
                cch1,
                timing,
            )
            if best is None or item[:2] > best[:2]:
                best = item

        if best is None:
            continue
        _, _, cch0_bits, cc_bits, cch1_bits, cch0, color_code, cch1, timing = best
        quality = _quality_summary(cch0, cch1)
        quality["timing_coherent"] = True
        quality["front_end_confidence"] = quality["confidence"]
        src, dst, superframe_part = session.feed(cch0, cch1)
        expose_ids = quality["confidence"] == "high" and superframe_part in ("src", "dst")
        src_out = src if expose_ids else ""
        dst_out = dst if expose_ids else ""
        raw_bits = _raw_bytes(cc_bits)
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
                    "color_code": color_code,
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
                    "cch": [_cch_extra(cch0), _cch_extra(cch1)],
                    "frame_numbers": [
                        rec.frame_number for rec in (cch0, cch1) if rec is not None
                    ],
                },
                "raw_bits": raw_bits,
            }
        )
    return results
