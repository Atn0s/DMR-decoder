import json
import os

import protocols
from common.config import DEFAULT_RADIO_CONFIG
from common.io import detect_sample_rate as _detect_sample_rate, read_rawiq
from radio import pipeline as radio_pipeline
from radio.pdu import pdu_to_dict


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


def scan_file(path: str, freq_list: list[float] | None = None,
              output_json: str | None = None,
              protocol_names: list[str] | tuple[str, ...] | set[str] | None = None) -> list[dict]:
    """Scan an offline IQ file. Returns all decoded PDUs.

    For wideband files (fs > 200kHz): Welch PSD blind search for candidates.
    For narrowband files (fs <= 200kHz): direct processing.
    freq_list overrides blind search with explicit frequency offsets.
    """
    enabled_protocols = protocols.normalize_protocol_names(protocol_names)
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


def _write_json(pdus: list[dict], path: str) -> None:
    clean = [pdu_to_dict(p, include_raw_bits=False) for p in pdus]
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
