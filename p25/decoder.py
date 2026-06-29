from __future__ import annotations

import numpy as np

from p25.constants import FS_NID_SYMBOLS
from p25.dsp import extract_nid_bits, recover_full_frame, recover_symbols_from_fs, slice_symbols_to_bits
from p25.framing import frame_info_from_nid
from p25.hdu_decode import HeaderCodeWord, decode_hdu_hcw
from p25.ldu2_decode import EncryptionSync, decode_ldu2_es
from p25.lc_decode import decode_ldu1_lc
from p25.link_control import LinkControl
from p25.nid import decode_nid
from p25.sync import find_frame_sync
from p25.session import P25SessionAssembler


def _stable_frame_indexes(frames: list[dict]) -> set[int]:
    valid = {i for i, f in enumerate(frames) if f["nid"].valid_bch is True}
    if valid:
        return valid

    counts: dict[int, int] = {}
    for f in frames:
        counts[f["nid"].nac] = counts.get(f["nid"].nac, 0) + 1
    if not counts:
        return set()
    nac, count = max(counts.items(), key=lambda item: item[1])
    if count < 5 or count < max(5, int(0.4 * len(frames))):
        return set()
    return {i for i, f in enumerate(frames) if f["nid"].nac == nac}


def _nid_pdu(
    nid,
    frame,
    candidate,
    bits,
    src: int,
    dst: int,
    lc: LinkControl | None,
    hcw: HeaderCodeWord | None,
    es: EncryptionSync | None,
) -> dict:
    extra = {
        "nac": nid.nac,
        "duid": nid.duid,
        "duid_name": nid.duid_name,
        "pdu_type": frame.pdu_type,
        "frame_category": frame.category,
        "is_voice": frame.is_voice,
        "is_control": frame.is_control,
        "is_terminator": frame.is_terminator,
        "has_link_control": frame.has_link_control,
        "valid_bch": nid.valid_bch,
        "corrected": nid.corrected,
        "fs_start": candidate.fs_start,
        "sync_ncc": candidate.ncc,
        "tgid": lc.tgid if lc is not None else (hcw.tgid if hcw is not None else 0),
        "rs_ok": lc is not None or hcw is not None or es is not None,
    }
    if lc is not None:
        extra.update(
            {
                "lco": lc.lco,
                "mfid": lc.mfid,
                "svc": lc.svc,
                "lc_info": lc.lc_info,
                "lc_octet2": lc.octet2,
                "lc_octet3": lc.octet3,
                "lc_emergency": lc.emergency,
                "lc_reserved": lc.reserved,
                "lc_reserved_bits": lc.reserved_bits,
                "is_group": lc.is_group,
                "call_type": lc.call_type,
            }
        )
    if hcw is not None:
        extra.update(
            {
                "mi": hcw.mi,
                "hdu_mfid": hcw.mfid,
                "algid": hcw.algid,
                "kid": hcw.kid,
                "hdu_tgid": hcw.tgid,
                "hdu_golay_corrected": hcw.golay_corrected,
            }
        )
    if es is not None:
        extra.update(
            {
                "es_mi": es.mi,
                "es_algid": es.algid,
                "es_kid": es.kid,
                "es_hamming_corrected": es.hamming_corrected,
            }
        )
    return {
        "protocol": "P25",
        "type": "P25_NID",
        "src": src,
        "dst": dst,
        "ts": 0,
        "flco": nid.duid_name,
        "fid": "",
        "extra": extra,
        "raw_bits": bits.tobytes(),
    }


def decode(
    y: np.ndarray,
    sps: int = 10,
    sync_threshold: float = 0.62,
) -> list[dict]:
    frames: list[dict] = []
    for candidate in find_frame_sync(y, sps=sps, threshold=sync_threshold):
        symbols = recover_symbols_from_fs(
            y,
            candidate,
            symbol_count=FS_NID_SYMBOLS,
            sps=sps,
        )
        if symbols is None:
            continue
        bits = slice_symbols_to_bits(symbols)
        try:
            nid_bits = extract_nid_bits(bits)
            nid = decode_nid(nid_bits)
        except ValueError:
            continue
        frame = frame_info_from_nid(nid)

        lc = None
        hcw = None
        es = None
        src = 0
        dst = 0
        if frame.duid in (0x0, 0x5, 0xA):
            full = recover_full_frame(y, candidate, sps=sps)
            if full is not None and frame.duid == 0x5:
                lc = decode_ldu1_lc(slice_symbols_to_bits(full))
                if lc is not None:
                    src = lc.src
                    dst = lc.dst
            elif full is not None and frame.duid == 0x0:
                hcw = decode_hdu_hcw(slice_symbols_to_bits(full))
                if hcw is not None:
                    dst = hcw.tgid
            elif full is not None and frame.duid == 0xA:
                es = decode_ldu2_es(slice_symbols_to_bits(full))

        frames.append(
            {
                "nid": nid,
                "frame": frame,
                "candidate": candidate,
                "bits": bits,
                "src": src,
                "dst": dst,
                "lc": lc,
                "hcw": hcw,
                "es": es,
            }
        )

    keep = _stable_frame_indexes(frames)
    results: list[dict] = []
    session = P25SessionAssembler()
    for i, rec in enumerate(frames):
        if i not in keep:
            continue
        nid = rec["nid"]
        frame = rec["frame"]
        candidate = rec["candidate"]
        lc = rec["lc"]
        hcw = rec["hcw"]
        es = rec["es"]
        pdu = _nid_pdu(nid, frame, candidate, rec["bits"], rec["src"], rec["dst"], lc, hcw, es)
        if frame.duid == 0x5 and lc is not None:
            pdu["type"] = "P25_LDU1"
        elif frame.duid == 0x0 and hcw is not None:
            pdu["type"] = "P25_HDU"
        elif frame.duid == 0xA and es is not None:
            pdu["type"] = "P25_LDU2"
        results.append(pdu)

        call = session.feed(frame, lc, candidate.fs_start, sps=sps)
        if call is not None:
            results.append(call)
    return results
