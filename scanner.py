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


PSD_PEAK_THRESHOLD_DB = 15  # dB above median noise floor for signal candidate detection


def _psd_blind_search(iq: np.ndarray, fs: float) -> list[float]:
    """Find signal candidates in wideband IQ via Welch PSD peak detection."""
    f, psd = signal.welch(iq, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-12)
    nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=nf + PSD_PEAK_THRESHOLD_DB, distance=20)
    return [float(f[p]) for p in peaks]


def _decode_loop(y: np.ndarray) -> list[dict]:
    """Shared decode loop: sync detection, dedup, burst recovery, and PDU decoding."""
    positions = find_sync_positions(y)
    results = []
    collector = LateEntryCollector()
    seen_bursts: set[tuple] = set()

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
            results.append(dict(pdu))

    return results


def _process_candidate(iq: np.ndarray, fo: float, fs_in: float) -> list[dict]:
    """DDC + resample + frontend, then decode; tags each PDU with _fo_hz."""
    t = np.arange(len(iq)) / fs_in
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * fo * t)
    iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)

    results = _decode_loop(y)
    for pdu in results:
        pdu["_fo_hz"] = fo
    return results


def _process_narrowband(iq: np.ndarray) -> list[dict]:
    """Resample + frontend for a narrowband stream already at or near 48kHz, then decode."""
    iq_dec = signal.resample_poly(iq, 384, 625)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
    return _decode_loop(y)


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

    # Cross-candidate dedup: same burst seen at two very close frequency offsets
    # (within 5kHz) is dropped; PDUs from genuinely different candidates are kept.
    seen_pdus: set[tuple] = set()
    unique: list[dict] = []
    for pdu in all_pdus:
        fo_bucket = round(pdu.get("_fo_hz", 0) / 5000) * 5000
        k = (pdu["src"], pdu["dst"], pdu["type"], fo_bucket)
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
