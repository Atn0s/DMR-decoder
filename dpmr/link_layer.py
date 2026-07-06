from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpmr.cch import CCHRecord, decode_cch
from dpmr.color_code import get_color_code
from dpmr.constants import CC_SYMBOLS, CCH_SYMBOLS, DIBIT_TO_BITS, FS2_SYMBOLS, TCH_SYMBOLS
from dpmr.session import DPMRSessionAssembler, cch_record_usable


@dataclass(frozen=True)
class DPMRHeaderDecode:
    cch_records: list[CCHRecord]
    cch_offsets: list[int]
    color_codes: list[int]
    color_offsets: list[int]
    payload_bits: list[int]
    color_code: int
    quality: dict
    src: str
    dst: str
    superframe_part: str
    raw_bits: bytes


@dataclass(frozen=True)
class DPMRVoiceDecode:
    cch0_bits: list[int]
    cc_bits: list[int]
    cch1_bits: list[int]
    cch0: CCHRecord | None
    cch1: CCHRecord | None
    color_code: int
    quality: dict
    raw_bits: bytes


def symbols_to_bits(symbols: np.ndarray) -> list[int]:
    out: list[int] = []
    for sym in symbols:
        out.extend(DIBIT_TO_BITS[int(sym) & 3])
    return out


def split_voice_fs2(symbols: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    offset = len(FS2_SYMBOLS)
    cch0 = symbols_to_bits(symbols[offset:offset + CCH_SYMBOLS])
    offset += CCH_SYMBOLS + TCH_SYMBOLS
    cc = symbols_to_bits(symbols[offset:offset + CC_SYMBOLS])
    offset += CC_SYMBOLS
    cch1 = symbols_to_bits(symbols[offset:offset + CCH_SYMBOLS])
    return cch0, cc, cch1


def cch_extra(record: CCHRecord | None) -> dict | None:
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


def records_quality(records: list[CCHRecord]) -> dict:
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


def voice_quality(cch0: CCHRecord | None, cch1: CCHRecord | None) -> dict:
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


def voice_decode_usable(cch0: CCHRecord | None, cch1: CCHRecord | None, color_code: int) -> bool:
    records = [rec for rec in (cch0, cch1) if rec is not None]
    if color_code >= 0 and any(rec.crc_ok for rec in records):
        return True
    return color_code >= 0 and any(rec.hamming_ok for rec in records)


def raw_bytes(bits: list[int]) -> bytes:
    return bytes(
        int("".join(str(bit) for bit in bits[i:i + 8]).ljust(8, "0"), 2)
        for i in range(0, len(bits), 8)
    )


def assemble_ids_from_records(records: list[CCHRecord]) -> tuple[str, str, str]:
    session = DPMRSessionAssembler()
    src = ""
    dst = ""
    part = "unknown"
    for idx in range(0, len(records), 2):
        a = records[idx]
        b = records[idx + 1] if idx + 1 < len(records) else None
        src, dst, part = session.feed(a, b)
    return src, dst, part


def decode_header_payload(payload_symbols: np.ndarray) -> DPMRHeaderDecode | None:
    cch_records: list[CCHRecord] = []
    cch_offsets: list[int] = []
    for offset in range(0, len(payload_symbols) - CCH_SYMBOLS + 1, CCH_SYMBOLS):
        record = decode_cch(symbols_to_bits(payload_symbols[offset:offset + CCH_SYMBOLS]))
        if record is not None and cch_record_usable(record):
            cch_records.append(record)
            cch_offsets.append(offset)

    color_codes: list[int] = []
    color_offsets: list[int] = []
    for offset in range(0, len(payload_symbols) - CC_SYMBOLS + 1, CC_SYMBOLS):
        color_code = get_color_code(symbols_to_bits(payload_symbols[offset:offset + CC_SYMBOLS]))
        if color_code >= 0:
            color_codes.append(color_code)
            color_offsets.append(offset)

    if not cch_records and not color_codes:
        return None

    quality = records_quality(cch_records)
    src, dst, superframe_part = assemble_ids_from_records(
        [rec for rec in cch_records if rec.crc_ok]
    )
    if quality["confidence"] != "high":
        src = ""
        dst = ""
    payload_bits = symbols_to_bits(payload_symbols)
    return DPMRHeaderDecode(
        cch_records=cch_records,
        cch_offsets=cch_offsets,
        color_codes=color_codes,
        color_offsets=color_offsets,
        payload_bits=payload_bits,
        color_code=color_codes[0] if color_codes else -1,
        quality=quality,
        src=src,
        dst=dst,
        superframe_part=superframe_part,
        raw_bits=raw_bytes(payload_bits),
    )


def decode_voice_symbols(symbols: np.ndarray) -> DPMRVoiceDecode | None:
    cch0_bits, cc_bits, cch1_bits = split_voice_fs2(symbols)
    cch0 = decode_cch(cch0_bits)
    cch1 = decode_cch(cch1_bits)
    color_code = get_color_code(cc_bits)
    if not voice_decode_usable(cch0, cch1, color_code):
        return None
    return DPMRVoiceDecode(
        cch0_bits=cch0_bits,
        cc_bits=cc_bits,
        cch1_bits=cch1_bits,
        cch0=cch0,
        cch1=cch1,
        color_code=color_code,
        quality=voice_quality(cch0, cch1),
        raw_bits=raw_bytes(cc_bits),
    )
