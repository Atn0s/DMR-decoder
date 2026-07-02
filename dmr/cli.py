from __future__ import annotations

import argparse
import os

import scanner


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
        scanner.scan_file(
            target,
            freq_list=args.fo,
            output_json=args.output_json,
            protocol_names=["dmr"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
