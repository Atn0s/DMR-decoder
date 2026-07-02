import json
import os
import numpy as np
import scipy.signal as signal
from math import gcd

import protocols
from common.io import detect_sample_rate as _detect_sample_rate, read_rawiq
from dmr.constants import Fs_wide, Fs_dec, UP_FACTOR, DOWN_FACTOR
from dmr.dsp import frontend
from dmr.offline import _decode_dmr_loop as _dmr_decode_loop
from dpmr.decoder import filter_stable_pdus
from dpmr.dsp import frontend_dpmr


SUPPORTED_PROTOCOLS = protocols.SUPPORTED_PROTOCOLS


def detect_sample_rate(path: str) -> int | None:
    """Extract sample rate from filenames like dmr_1_78125.rawiq."""
    return _detect_sample_rate(path)


PSD_PEAK_THRESHOLD_DB = 15  # dB above median noise floor for signal candidate detection
# At 48kHz, same-slot consecutive voice bursts are separated by one full 60ms TDMA frame = 2880 samples
def _psd_blind_search(iq: np.ndarray, fs: float) -> list[float]:
    """Find signal candidates in wideband IQ via Welch PSD peak detection."""
    f, psd = signal.welch(iq, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-12)
    nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=nf + PSD_PEAK_THRESHOLD_DB, distance=20)
    return [float(f[p]) for p in peaks]


def _decode_dmr_loop(y: np.ndarray) -> list[dict]:
    """Backward-compatible wrapper for the relocated DMR decode loop."""
    return _dmr_decode_loop(y)


def _decode_loop(y: np.ndarray) -> list[dict]:
    return protocols.decode_all(y)


def _normalize_protocol_names(protocol_names: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    return protocols.normalize_protocol_names(protocol_names)


def _decode_protocol_frontends(
    y: np.ndarray,
    y_dpmr: np.ndarray | None = None,
    protocol_names: set[str] | None = None,
) -> list[dict]:
    names = protocol_names or set(SUPPORTED_PROTOCOLS)
    frontends = {"dPMR": y_dpmr if y_dpmr is not None else y}
    return protocols.decode_all(y, protocol_names=names, frontends=frontends)


def _resample_factors(source_sample_rate: float, target: float = Fs_dec) -> tuple[int, int]:
    up = int(round(target))
    down = int(round(source_sample_rate))
    g = gcd(up, down)
    return up // g, down // g


def _process_candidate(
    iq: np.ndarray,
    fo: float,
    fs_in: float,
    protocol_names: set[str] | None = None,
) -> list[dict]:
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

    results = _decode_protocol_frontends(y, y_dpmr, protocol_names)
    for pdu in results:
        pdu["_fo_hz"] = fo
    return results


def _process_narrowband(
    iq: np.ndarray,
    fs_in: float | None = None,
    protocol_names: set[str] | None = None,
) -> list[dict]:
    """Resample + frontend for a narrowband stream already at or near 48kHz, then decode."""
    fs = fs_in or Fs_dec
    if abs(fs - Fs_dec) < 1:
        iq_dec = iq
    else:
        up, down = _resample_factors(fs)
        iq_dec = signal.resample_poly(iq, up, down)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
    y_dpmr = frontend_dpmr(iq_dec, fs=Fs_dec)
    return _decode_protocol_frontends(y, y_dpmr, protocol_names)


def scan_file(path: str, freq_list: list[float] | None = None,
              output_json: str | None = None,
              protocol_names: list[str] | tuple[str, ...] | set[str] | None = None) -> list[dict]:
    """Scan an offline IQ file. Returns all decoded PDUs.

    For wideband files (fs > 200kHz): Welch PSD blind search for candidates.
    For narrowband files (fs <= 200kHz): direct processing.
    freq_list overrides blind search with explicit frequency offsets.
    """
    enabled_protocols = _normalize_protocol_names(protocol_names)
    iq = read_rawiq(path)
    fs = detect_sample_rate(path)

    if freq_list is not None:
        fs_in = fs or Fs_wide
        all_pdus = []
        for fo in freq_list:
            all_pdus.extend(_process_candidate(iq, fo, fs_in, enabled_protocols))
    elif fs is None or fs > 200_000:
        fs_in = fs or Fs_wide
        fos = _psd_blind_search(iq, fs_in)
        all_pdus = []
        for fo in fos:
            all_pdus.extend(_process_candidate(iq, fo, fs_in, enabled_protocols))
    else:
        all_pdus = _process_narrowband(iq, fs, enabled_protocols)

    all_pdus = filter_stable_pdus(all_pdus)

    # Cross-candidate dedup: protocol-specific keys live in protocols.py.
    unique = protocols.deduplicate_pdus(all_pdus)

    _print_results(unique)
    if output_json:
        _write_json(unique, output_json)
    return unique


def _print_results(pdus: list[dict]) -> None:
    for p in pdus:
        print(protocols.format_pdu(p))


def _format_dpmr_result(pdu: dict, fo_str: str = "") -> str:
    return protocols.format_dpmr_pdu(pdu, fo_str)


def _format_dpmr_cch(cch_records: list[dict | None]) -> str:
    return protocols._format_dpmr_cch(cch_records)


def _format_p25_result(pdu: dict, fo_str: str = "") -> str:
    return protocols.format_p25_pdu(pdu, fo_str)


def _p25_detail(pdu: dict) -> str:
    return protocols._p25_detail(pdu)


def _write_json(pdus: list[dict], path: str) -> None:
    clean = [{k: v for k, v in p.items() if k != "raw_bits"} for p in pdus]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Scan offline IQ files for DMR, P25, and dPMR metadata.")
    parser.add_argument("targets", nargs="*", default=["data/dmr_1_78125.rawiq"])
    parser.add_argument("--protocol", action="append", choices=["dmr", "p25", "dpmr"],
                        help="limit decoding to one protocol; repeat to enable several")
    parser.add_argument("--fo", type=float, action="append", default=None,
                        help="frequency offset in Hz; repeat for multiple candidates")
    parser.add_argument("--json", dest="output_json", default=None,
                        help="write decoded PDUs to JSON; only valid for one target")
    args = parser.parse_args(argv)

    if args.output_json and len(args.targets) != 1:
        parser.error("--json can only be used with one target")

    for t in args.targets:
        if not os.path.exists(t):
            print(f"File not found: {t}")
            continue
        print(f"\n=== {t} ===")
        scan_file(
            t,
            freq_list=args.fo,
            output_json=args.output_json,
            protocol_names=args.protocol,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
