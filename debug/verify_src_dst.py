"""Independently verify SRC/DST parsing on a real BVSP capture.

Hooks the decode core at runtime (no permanent code change) and, for every
LC / terminator / late-entry PDU that decodes, prints side by side:
  - the raw FEC-protected LC bytes,
  - whether the Reed-Solomon (LC) / CS5 (late-entry) check passed,
  - SRC/DST/FLCO/FID extracted MANUALLY per ETSI TS 102 361-1 bit positions,
  - SRC/DST/FLCO/FID as reported by the okdmr library.

If the manual extraction matches the library AND the FEC check passes, the
addresses are genuinely carried in the signal (not a parsing artifact or a
coincidental "1").

Usage (run from the project root; no -m needed):
  python debug/verify_src_dst.py [N]       # default file 1
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import core.decoder as D
from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner

BVSP_HEADER_BYTES = 112
FS = 61.44e6
CENTER_HZ = 431e6

_seen = set()


def _manual_lc_fields(lc9: bytes):
    """Extract LC fields from the 9-byte payload per ETSI TS 102 361-1 Tbl 7.1."""
    flco = lc9[0] & 0x3F                       # byte0: PF(1)+reserved(1)+FLCO(6)
    fid = lc9[1]                               # byte1: Feature set ID
    svc = lc9[2]                               # byte2: Service Options
    dst = int.from_bytes(lc9[3:6], "big")      # bytes3-5: target/group address
    src = int.from_bytes(lc9[6:9], "big")      # bytes6-8: source address
    return flco, fid, svc, dst, src


def _hook_lc():
    orig = D._decode_lc_or_terminator

    def patched(ba264, info196, color_code, pdu_type):
        decoded = D.BPTC19696.deinterleave_data_bits(info196, repair_if_necessary=True)
        data12 = decoded[0:96].tobytes()       # 12 bytes: 9 LC + 3 RS parity
        rs_ok = D.ReedSolomon1294.check(data12, D.VLC_RS_MASK)
        res = orig(ba264, info196, color_code, pdu_type)
        if res is not None:
            mflco, mfid, msvc, mdst, msrc = _manual_lc_fields(data12[:9])
            key = (pdu_type, data12[:9])
            if key not in _seen:
                _seen.add(key)
                match = (msrc == res["src"] and mdst == res["dst"])
                print(f"\n  [{pdu_type}]  RS_FEC_ok={rs_ok}")
                print(f"    raw LC bytes : {data12[:9].hex()}  (+RS {data12[9:12].hex()})")
                print(f"    manual (ETSI): FLCO={mflco} FID={mfid} SVC={msvc} "
                      f"DST={mdst} SRC={msrc}")
                print(f"    okdmr library: FLCO={res['flco']} FID={res['fid']} "
                      f"DST={res['dst']} SRC={res['src']}")
                print(f"    --> SRC/DST match: {'YES' if match else 'NO !!!'}")
        return res

    D._decode_lc_or_terminator = patched


def _hook_late_entry():
    orig = D.LateEntryCollector._decode_assembled

    def patched(self, last_ba264):
        # reconstruct the 72-bit LC the same way _decode_assembled does, so we can
        # extract the address fields manually before the library parses them.
        b128 = self._frags[0] + self._frags[1] + self._frags[2] + self._frags[3]
        lc77 = D.VBPTC12873.deinterleave_data_bits(b128, include_cs5=True)
        lc9 = lc77[0:72].tobytes()
        res = orig(self, last_ba264)
        if res is not None:
            key = ("LATE_ENTRY", lc9)
            if key not in _seen:
                _seen.add(key)
                mflco, mfid, msvc, mdst, msrc = _manual_lc_fields(lc9)
                match = (msrc == res["src"] and mdst == res["dst"])
                print(f"\n  [LATE_ENTRY] (embedded LC, VBPTC+CS5 — FEC carrier "
                      f"INDEPENDENT from the LC header)")
                print(f"    raw LC bytes : {lc9.hex()}")
                print(f"    manual (ETSI): FLCO={mflco} FID={mfid} SVC={msvc} "
                      f"DST={mdst} SRC={msrc}")
                print(f"    okdmr library: FLCO={res['flco']} FID={res['fid']} "
                      f"DST={res['dst']} SRC={res['src']}")
                print(f"    --> SRC/DST match: {'YES' if match else 'NO !!!'}")
        return res

    D.LateEntryCollector._decode_assembled = patched


def main():
    idx = sys.argv[1] if len(sys.argv) > 1 else "1"
    path = os.path.join(_ROOT, "DMR_signal", f"{idx}.bvsp")
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")

    _hook_lc()
    _hook_late_entry()

    print(f"== Verifying SRC/DST parsing on {idx}.bvsp ==")
    print("   (manual ETSI bit extraction vs okdmr library, with FEC status)")
    src = FileWidebandSource(path, sample_rate=FS, center_hz=CENTER_HZ,
                             chunk_samples=int(FS), throttle=False,
                             header_bytes=BVSP_HEADER_BYTES)
    scanner = WidebandScanner(src, num_subbands=48, oversample=2,
                              window_sec=0.5, step_sec=0.25)
    calls = scanner.run()

    print("\n== Decoded calls ==")
    for c in calls:
        print(f"  RF={c.fo_hz/1e6:.4f}MHz SRC={c.src} DST={c.dst} "
              f"FLCO={c.flco} FID={c.fid}")
    print("\nIf every block above shows RS_FEC_ok=True and SRC/DST match: YES,")
    print("the addresses are FEC-validated and parsed exactly per ETSI — not noise.")


if __name__ == "__main__":
    main()
