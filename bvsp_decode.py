"""Parse + decode real USRP/BVSP wideband captures through the channelizer.

The DMR_signal/*.bvsp files are USRP captures: a 112-byte header followed by
interleaved little-endian int16 IQ at 61.44 Msps, center 431 MHz, 1 s each
(see DMR_signal/README.txt).  This script runs each capture through the real
WidebandScanner (PFB channelizer -> per-sub-band detect -> decode) and reports
the decoded calls at their absolute RF.

Usage (run from the project root; no -m needed):
  python bvsp_decode.py [N|all]
    N    : decode DMR_signal/N.bvsp (default 1)
    all  : decode files 1..5

With the project's dedicated interpreter:
  /home/lzkj/miniconda3/envs/DMR_demo/bin/python bvsp_decode.py all
"""
import os
import sys
import time

# This file lives at the repository root, so the root *is* its own directory.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner

# BVSP / USRP capture parameters (from DMR_signal/README.txt + header decode)
BVSP_HEADER_BYTES = 112
FS = 61.44e6
CENTER_HZ = 431e6

# Channelizer config: N=48, oversample=2 -> sub-band rate 2.56 MHz (close to the
# 2.5 MHz decode design point; 2.56 MHz -> 48 kHz resamples as a clean 3/160),
# sub-band spacing 1.28 MHz, owning region +/-640 kHz.
NUM_SUBBANDS = 48
OVERSAMPLE = 2
# Files are only 1 s long; use sub-second windows so there are several decode
# windows per capture (a DMR call needs ~720 ms of presence to lock).
WINDOW_SEC = 0.5
STEP_SEC = 0.25


def decode_file(path: str) -> list:
    src = FileWidebandSource(path, sample_rate=FS, center_hz=CENTER_HZ,
                             chunk_samples=int(FS), throttle=False,
                             header_bytes=BVSP_HEADER_BYTES)
    scanner = WidebandScanner(src, num_subbands=NUM_SUBBANDS, oversample=OVERSAMPLE,
                              window_sec=WINDOW_SEC, step_sec=STEP_SEC)

    t0 = time.time()
    calls = scanner.run(on_call=lambda c: print(
        f"    [CALL] RF={c.fo_hz/1e6:.4f}MHz SRC={c.src} DST={c.dst} "
        f"FLCO={c.flco} FID={c.fid} closed_by={c.closed_by} "
        f"windows={c.start_window}-{c.end_window}"))
    dt = time.time() - t0
    print(f"  -> {len(calls)} call(s) in {dt:.1f}s")
    return calls


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    if arg == "all":
        indices = range(1, 6)
    else:
        indices = [int(arg)]

    grand_total = []
    for i in indices:
        path = os.path.join(_ROOT, "DMR_signal", f"{i}.bvsp")
        if not os.path.exists(path):
            print(f"== {i}.bvsp: NOT FOUND, skipping ==")
            continue
        print(f"== Decoding {i}.bvsp "
              f"(fs={FS/1e6:g}MHz center={CENTER_HZ/1e6:g}MHz "
              f"N={NUM_SUBBANDS} os={OVERSAMPLE}) ==")
        calls = decode_file(path)
        for c in calls:
            grand_total.append((i, c))

    print("\n=== SUMMARY ===")
    if not grand_total:
        print("no calls decoded")
    voice = [(i, c) for (i, c) in grand_total if c.flco == "GroupVoiceChannelUser"]
    print(f"total calls: {len(grand_total)}  (voice: {len(voice)})")
    for (i, c) in grand_total:
        print(f"  file {i}: RF={c.fo_hz/1e6:.4f}MHz SRC={c.src} DST={c.dst} "
              f"FLCO={c.flco} FID={c.fid} closed_by={c.closed_by}")


if __name__ == "__main__":
    main()
