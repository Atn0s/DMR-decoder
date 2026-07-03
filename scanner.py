import os

import protocols
from common.config import DEFAULT_RADIO_CONFIG
from common.io import detect_sample_rate as _detect_sample_rate, read_rawiq
from radio import output as radio_output
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


def scan_file(path: str, freq_list: list[float] | None = None,
              output_json: str | None = None,
              protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
              sample_rate: float | None = None,
              blind_search: bool = False,
              iq_dtype: str = "int16") -> list[dict]:
    """Scan an offline IQ file. Returns all decoded PDUs.

    By default the input IQ is treated as a centered baseband channel.
    Set blind_search=True for Welch PSD candidate search over a wider IQ span.
    freq_list overrides blind search with explicit frequency offsets.
    """
    iq = read_rawiq(path, dtype=iq_dtype)
    fs = sample_rate if sample_rate is not None else detect_sample_rate(path)
    if fs is None:
        raise ValueError(
            "sample rate is required; pass sample_rate/--fs or use a filename "
            "with sample rate metadata"
        )

    unique = radio_pipeline.scan_iq(
        iq,
        fs,
        freq_list=freq_list,
        blind_search=blind_search,
        protocol_names=protocol_names,
        radio_config=RADIO_CONFIG,
    )

    radio_output.print_results(unique)
    if output_json:
        radio_output.write_json(unique, output_json)
    return unique


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Scan offline IQ files for DMR, P25, and dPMR metadata.")
    parser.add_argument("targets", nargs="*", default=["data/dmr_1_78125.rawiq"])
    parser.add_argument("--protocol", action="append", choices=["dmr", "p25", "dpmr"],
                        help="limit decoding to one protocol; repeat to enable several")
    parser.add_argument("--fo", type=float, action="append", default=None,
                        help="frequency offset in Hz; repeat for multiple candidates")
    parser.add_argument("--fs", "--sample-rate", dest="sample_rate", type=float, default=None,
                        help="input IQ sample rate in Hz; overrides filename inference")
    parser.add_argument("--blind-search", action="store_true",
                        help="run Welch PSD candidate search instead of assuming centered baseband")
    parser.add_argument("--iq-dtype", default="int16",
                        choices=["int8", "int16", "int32", "float32", "float64"],
                        help="interleaved IQ scalar dtype")
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
            sample_rate=args.sample_rate,
            blind_search=args.blind_search,
            iq_dtype=args.iq_dtype,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
