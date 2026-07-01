import re
import json
import os
import numpy as np
import scipy.signal as signal
from math import gcd
from dataclasses import dataclass, field

from core.burst_type import Fs_wide, Fs_dec, UP_FACTOR, DOWN_FACTOR, SPS, SYNC_TEMPLATES
from core.dsp import read_rawiq, frontend, find_sync_positions, recover_burst, adaptive_slice_bits, _interp
from core.decoder import decode_burst, LateEntryCollector
from dpmr.decoder import filter_stable_pdus
from dpmr.dsp import frontend_dpmr


@dataclass
class Session:
    src: int
    dst: int
    start_pdu: dict
    voice_raw: list = field(default_factory=list)
    terminator: dict | None = None
    late_entry_lc: dict | None = None
    duration_s: float | None = None


def detect_sample_rate(path: str) -> int | None:
    """Extract sample rate from filename, e.g. dmr_1_78125.rawiq -> 78125. Returns None if not found."""
    m = re.search(r'_(\d{4,7})\.rawiq', os.path.basename(path))
    return int(m.group(1)) if m else None


PSD_PEAK_THRESHOLD_DB = 15  # dB above median noise floor for signal candidate detection
# At 48kHz, same-slot consecutive voice bursts are separated by one full 60ms TDMA frame = 2880 samples
BURST_STRIDE = 2880


def _psd_blind_search(iq: np.ndarray, fs: float) -> list[float]:
    """Find signal candidates in wideband IQ via Welch PSD peak detection."""
    f, psd = signal.welch(iq, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-12)
    nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=nf + PSD_PEAK_THRESHOLD_DB, distance=20)
    return [float(f[p]) for p in peaks]


def _lock_voice_phase(y: np.ndarray, anchor: int, polarity: float, sync_type: str) -> float:
    """Lock sub-symbol phase using Burst A's known Voice Sync region (symbols [54,78))."""
    ref = SYNC_TEMPLATES[sync_type]
    levels = np.array([-3, -1, 1, 3])
    best = (1e18, 0.0)
    for ph in np.linspace(-8, 8, 65):
        start = anchor - (54 + 12) * SPS + ph
        pos = start + np.arange(132) * SPS
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue
        seg = polarity * _interp(y, pos)
        sy = seg[54:78]
        a, b = np.linalg.lstsq(np.vstack([sy, np.ones(24)]).T, ref, rcond=None)[0]
        segc = a * seg + b
        near = levels[np.argmin(np.abs(segc[:, None] - levels[None, :]), axis=1)]
        resid = np.mean((segc - near) ** 2)
        if resid < best[0]:
            best = (resid, ph)
    return best[1]


def _recover_stepped_burst(y: np.ndarray, anchor: int, j: int, ph: float, polarity: float):
    """Recover voice burst j hops from Burst A anchor using fixed BURST_STRIDE.
    Returns 264-bit bitarray or None if out of bounds."""
    start = anchor + BURST_STRIDE * j - (54 + 12) * SPS + ph
    pos = start + np.arange(132) * SPS
    if pos[0] < 0 or pos[-1] >= len(y) - 1:
        return None
    seg = polarity * _interp(y, pos)
    return adaptive_slice_bits(seg)


def _decode_dmr_loop(y: np.ndarray) -> list[dict]:
    """Existing DMR-only decode loop.

    This is the old _decode_loop body. Keep all DMR behavior unchanged.
    """
    positions = find_sync_positions(y)
    results = []
    seen_bursts: set[tuple] = set()

    for center, polarity, sync_type in positions:
        dedup_key = (round(center / 50), sync_type)
        if dedup_key in seen_bursts:
            continue
        seen_bursts.add(dedup_key)

        if "VOICE" in sync_type:
            # Burst A anchor: step through A(j=0) to F(j=5) at fixed stride
            ph = _lock_voice_phase(y, center, polarity, sync_type)
            collector = LateEntryCollector()
            for j in range(6):
                ba = _recover_stepped_burst(y, center, j, ph, polarity)
                if ba is None:
                    break
                pdu = collector.feed(ba, sync_type)
                if pdu is not None:
                    results.append(dict(pdu))
                    break
        else:
            symbols = recover_burst(y, center, polarity, sync_type)
            if symbols is None:
                continue
            pdu = decode_burst(symbols, sync_type)
            if pdu is not None:
                results.append(dict(pdu))

    return results


def _decode_loop(y: np.ndarray) -> list[dict]:
    import protocols

    return protocols.decode_all(y)


def _decode_protocol_frontends(y: np.ndarray, y_dpmr: np.ndarray | None = None) -> list[dict]:
    import protocols

    results: list[dict] = []
    results.extend(protocols.decode_dmr(y))
    results.extend(protocols.decode_p25(y))
    results.extend(protocols.decode_dpmr(y_dpmr if y_dpmr is not None else y))
    return results


def _resample_factors(source_sample_rate: float, target: float = Fs_dec) -> tuple[int, int]:
    up = int(round(target))
    down = int(round(source_sample_rate))
    g = gcd(up, down)
    return up // g, down // g


def _process_candidate(iq: np.ndarray, fo: float, fs_in: float) -> list[dict]:
    """DDC + resample + frontend, then decode; tags each PDU with _fo_hz."""
    t = np.arange(len(iq)) / fs_in
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * fo * t)
    if abs(fs_in - Fs_dec) < 1:
        iq_dec = iq_shifted
    elif abs(fs_in - Fs_wide) < 1:
        iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)
    else:
        up, down = _resample_factors(fs_in)
        iq_dec = signal.resample_poly(iq_shifted, up, down)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
    y_dpmr = frontend_dpmr(iq_dec, fs=Fs_dec)

    results = _decode_protocol_frontends(y, y_dpmr)
    for pdu in results:
        pdu["_fo_hz"] = fo
    return results


def _process_narrowband(iq: np.ndarray, fs_in: float | None = None) -> list[dict]:
    """Resample + frontend for a narrowband stream already at or near 48kHz, then decode."""
    fs = fs_in or Fs_dec
    if abs(fs - Fs_dec) < 1:
        iq_dec = iq
    else:
        up, down = _resample_factors(fs)
        iq_dec = signal.resample_poly(iq, up, down)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
    y_dpmr = frontend_dpmr(iq_dec, fs=Fs_dec)
    return _decode_protocol_frontends(y, y_dpmr)


def scan_file(path: str, freq_list: list[float] | None = None,
              output_json: str | None = None) -> list[dict]:
    """Scan an offline IQ file. Returns all decoded PDUs.

    For wideband files (fs > 200kHz): Welch PSD blind search for candidates.
    For narrowband files (fs <= 200kHz): direct processing.
    freq_list overrides blind search with explicit frequency offsets.
    """
    iq = read_rawiq(path)
    fs = detect_sample_rate(path)

    if freq_list is not None:
        fs_in = fs or Fs_wide
        all_pdus = []
        for fo in freq_list:
            all_pdus.extend(_process_candidate(iq, fo, fs_in))
    elif fs is None or fs > 200_000:
        fs_in = fs or Fs_wide
        fos = _psd_blind_search(iq, fs_in)
        all_pdus = []
        for fo in fos:
            all_pdus.extend(_process_candidate(iq, fo, fs_in))
    else:
        all_pdus = _process_narrowband(iq, fs)

    all_pdus = filter_stable_pdus(all_pdus)

    # Cross-candidate dedup: same burst seen at two very close frequency offsets
    # (within 5kHz) is dropped; PDUs from genuinely different candidates are kept.
    seen_pdus: set[tuple] = set()
    unique: list[dict] = []
    for pdu in all_pdus:
        if pdu.get("protocol") == "P25":
            extra = pdu.get("extra", {})
            frame_bucket = round(extra.get("fs_start", 0) / 8640)
            k = ("P25", extra.get("nac"), pdu["type"], frame_bucket)
        elif pdu.get("protocol") == "dPMR":
            extra = pdu.get("extra", {})
            frame_bucket = round(extra.get("fs_start", 0) / 3840)
            k = (
                "dPMR",
                pdu.get("src", ""),
                pdu.get("dst", ""),
                extra.get("color_code"),
                frame_bucket,
            )
        else:
            fo_bucket = round(pdu.get("_fo_hz", 0) / 5000) * 5000
            k = ("DMR", pdu["src"], pdu["dst"], pdu["type"], fo_bucket)
        if k not in seen_pdus:
            seen_pdus.add(k)
            unique.append(pdu)

    _print_results(unique)
    if output_json:
        _write_json(unique, output_json)
    return unique


def _print_results(pdus: list[dict]) -> None:
    for p in pdus:
        fo_str = f" (fo={p['_fo_hz']/1e3:+.1f}kHz)" if "_fo_hz" in p else ""
        proto = p.get("protocol", "DMR")
        if proto == "P25":
            print(_format_p25_result(p, fo_str))
            continue
        if proto == "dPMR":
            print(_format_dpmr_result(p, fo_str))
            continue
        extra = p.get("extra", {})
        print(
            f"[{p['type']:<12}] PROTO={proto} SRC={p['src']} DST={p['dst']} "
            f"FLCO={p['flco']} FID={p.get('fid','')}{fo_str}"
        )


def _format_dpmr_result(pdu: dict, fo_str: str = "") -> str:
    extra = pdu.get("extra", {})
    color_code = extra.get("color_code", -1)
    pol = "INV" if extra.get("polarity_inverted") else "NORM"
    quality = extra.get("quality", {})
    confidence = quality.get("front_end_confidence", quality.get("confidence", ""))
    crc = quality.get("crc_ok_count", 0)
    ham = quality.get("hamming_ok_count", 0)
    sync_type = extra.get("sync_type", "")
    timing = extra.get("segment_timing", {}).get("cc", {})
    if not timing:
        timing = extra.get("segment_timing", {}).get("header", {})
    e90 = timing.get("decision_error_p90")
    amb = timing.get("ambiguous_symbols")
    decision = (
        f" E90={e90:.2f} AMB={amb}"
        if isinstance(e90, (int, float)) and isinstance(amb, int)
        else ""
    )
    src = pdu.get("src") or ""
    dst = pdu.get("dst") or ""
    cc_text = f"{color_code:02d}" if isinstance(color_code, int) and color_code >= 0 else "--"
    return (
        f"[{pdu['type']:<12}] PROTO=dPMR SRC={src} DST={dst} "
        f"CC={cc_text} SYNC={sync_type} POL={pol} QUAL={confidence} CRC={crc} HAM={ham}"
        f"{decision}{fo_str}"
    )


def _format_p25_result(pdu: dict, fo_str: str = "") -> str:
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


def _write_json(pdus: list[dict], path: str) -> None:
    clean = [{k: v for k, v in p.items() if k != "raw_bits"} for p in pdus]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["data/dmr_1_78125.rawiq"]
    for t in targets:
        if not os.path.exists(t):
            print(f"File not found: {t}")
            continue
        print(f"\n=== {t} ===")
        scan_file(t)
