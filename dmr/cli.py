from __future__ import annotations

import argparse
import json
import os

from dmr.offline import scan_file


def _format_dmr_result(pdu: dict) -> str:
    fo_str = f" (fo={pdu['_fo_hz']/1e3:+.1f}kHz)" if "_fo_hz" in pdu else ""
    proto = pdu.get("protocol", "DMR")
    return (
        f"[{pdu['type']:<12}] PROTO={proto} SRC={pdu['src']} DST={pdu['dst']} "
        f"FLCO={pdu['flco']} FID={pdu.get('fid','')}{fo_str}"
    )


def _write_json(pdus: list[dict], path: str) -> None:
    clean = [{k: v for k, v in p.items() if k != "raw_bits"} for p in pdus]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode DMR metadata from offline IQ files.")
    parser.add_argument("targets", nargs="*", default=["data/dmr_1_78125.rawiq"])
    parser.add_argument("--fo", type=float, action="append", default=None,
                        help="frequency offset in Hz; repeat for multiple candidates")
    parser.add_argument("--json", dest="output_json", default=None,
                        help="write decoded PDUs to JSON; only valid for one target")
    args = parser.parse_args(argv)

    if args.output_json and len(args.targets) != 1:
        parser.error("--json can only be used with one target")

    for target in args.targets:
        if not os.path.exists(target):
            print(f"File not found: {target}")
            continue
        print(f"\n=== {target} ===")
        pdus = scan_file(target, freq_list=args.fo)
        for pdu in pdus:
            print(_format_dmr_result(pdu))
        if args.output_json:
            _write_json(pdus, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

