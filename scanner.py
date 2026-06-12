import re
import json
import os
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass, field

from core.burst_type import Fs_wide, Fs_dec, UP_FACTOR, DOWN_FACTOR
from core.dsp import read_rawiq, frontend, find_sync_positions, recover_burst, adaptive_slice_bits
from core.decoder import decode_burst, LateEntryCollector


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


def _psd_blind_search(iq: np.ndarray, fs: float) -> list[float]:
    """Find signal candidates in wideband IQ via Welch PSD peak detection."""
    f, psd = signal.welch(iq, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-12)
    nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=nf + 15, distance=20)
    return [float(f[p]) for p in peaks]


def _process_candidate(iq: np.ndarray, fo: float, fs_in: float) -> list[dict]:
    """Run full decode pipeline for one frequency offset candidate."""
    t = np.arange(len(iq)) / fs_in
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * fo * t)
    iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)

    positions = find_sync_positions(y)
    results = []
    collector = LateEntryCollector()
    seen_bursts: set[int] = set()

    for center, polarity, sync_type in positions:
        dedup_key = (round(center / 50), sync_type)
        if dedup_key in seen_bursts:
            continue
        seen_bursts.add(dedup_key)

        symbols = recover_burst(y, center, polarity, sync_type)
        if symbols is None:
            continue

        ba264 = adaptive_slice_bits(symbols)

        if "VOICE" in sync_type:
            pdu = collector.feed(ba264, sync_type)
        else:
            pdu = decode_burst(symbols, sync_type)

        if pdu is not None:
            pdu = dict(pdu)
            pdu["_fo_hz"] = fo
            results.append(pdu)

    return results


def _process_narrowband(iq: np.ndarray) -> list[dict]:
    """Process a narrowband IQ stream already at or near 48kHz."""
    iq_dec = signal.resample_poly(iq, 384, 625)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
    positions = find_sync_positions(y)
    results = []
    collector = LateEntryCollector()
    seen: set[int] = set()

    for center, polarity, sync_type in positions:
        key = (round(center / 50), sync_type)
        if key in seen:
            continue
        seen.add(key)
        symbols = recover_burst(y, center, polarity, sync_type)
        if symbols is None:
            continue
        ba264 = adaptive_slice_bits(symbols)
        if "VOICE" in sync_type:
            pdu = collector.feed(ba264, sync_type)
        else:
            pdu = decode_burst(symbols, sync_type)
        if pdu is not None:
            results.append(pdu)

    return results


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
        all_pdus = _process_narrowband(iq)

    # Cross-candidate dedup: keep first occurrence of each (src, dst, type)
    seen_pdus: set[tuple] = set()
    unique: list[dict] = []
    for pdu in all_pdus:
        k = (pdu["src"], pdu["dst"], pdu["type"])
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
        print(f"[{p['type']:<12}] SRC={p['src']} DST={p['dst']} FLCO={p['flco']}{fo_str}")


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
