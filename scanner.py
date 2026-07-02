import json
import os
import numpy as np

import protocols
from common.config import DEFAULT_RADIO_CONFIG
from common.io import detect_sample_rate as _detect_sample_rate, read_rawiq
from dmr.dsp import frontend  # Backward-compatible scanner.frontend export.
from dmr.offline import _decode_dmr_loop as _dmr_decode_loop
from radio import pipeline as radio_pipeline


SUPPORTED_PROTOCOLS = protocols.SUPPORTED_PROTOCOLS
RADIO_CONFIG = DEFAULT_RADIO_CONFIG

# Backward-compatible module-level names. New code should use RADIO_CONFIG.
Fs_dec = RADIO_CONFIG.target_sample_rate_hz
Fs_wide = RADIO_CONFIG.wideband_sample_rate_hz
UP_FACTOR = RADIO_CONFIG.wideband_resample_up
DOWN_FACTOR = RADIO_CONFIG.wideband_resample_down
PSD_PEAK_THRESHOLD_DB = RADIO_CONFIG.psd_peak_threshold_db


def detect_sample_rate(path: str) -> int | None:
    """Extract sample rate from filenames like dmr_1_78125.rawiq."""
    return _detect_sample_rate(path)


# At 48kHz, same-slot consecutive voice bursts are separated by one full 60ms TDMA frame = 2880 samples
def _psd_blind_search(iq: np.ndarray, fs: float) -> list[float]:
    """Find signal candidates in wideband IQ via Welch PSD peak detection."""
    return radio_pipeline.psd_blind_search(iq, fs, RADIO_CONFIG)


def _decode_dmr_loop(y: np.ndarray, config: object | None = None) -> list[dict]:
    """Backward-compatible wrapper for the relocated DMR decode loop."""
    return _dmr_decode_loop(y, config)


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
    return radio_pipeline.resample_factors(source_sample_rate, target)


def _process_candidate(
    iq: np.ndarray,
    fo: float,
    fs_in: float,
    protocol_names: set[str] | None = None,
) -> list[dict]:
    """DDC + resample + frontend, then decode; tags each PDU with _fo_hz."""
    return radio_pipeline.process_candidate(iq, fo, fs_in, protocol_names, RADIO_CONFIG)


def _process_narrowband(
    iq: np.ndarray,
    fs_in: float | None = None,
    protocol_names: set[str] | None = None,
) -> list[dict]:
    """Resample + frontend for a narrowband stream already at or near 48kHz, then decode."""
    return radio_pipeline.process_narrowband(iq, fs_in, protocol_names, RADIO_CONFIG)


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

    unique = radio_pipeline.scan_iq(
        iq,
        fs,
        freq_list=freq_list,
        protocol_names=enabled_protocols,
        radio_config=RADIO_CONFIG,
    )

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
