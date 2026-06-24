"""Live interactive demo of the wideband DMR channelizer scan.

Opens a pop-up window (like dmr_pipeline_v2) and runs the scan *live*: the
channelizer splits the wideband capture into sub-bands, then the window steps
across the band sub-band by sub-band — you watch the scan cursor move, signals
light up when detected, and decoded calls pop onto the RF axis in real time.

It drives the real production components (PolyphaseChannelizer / Detector /
decode_window) exactly as WidebandScanner.run() does, so what you see on screen
is the real pipeline working, not a mock-up.

Examples (run from the project root; no -m needed)
--------
Real USRP/BVSP capture (default = DMR_signal/5.bvsp, two calls):
    python utils/wideband_live.py
    python utils/wideband_live.py --file DMR_signal/1.bvsp
Synthetic 2-signal scene (no capture needed):
    python utils/wideband_live.py --synth
Slower/faster playback:
    python utils/wideband_live.py --pause 0.6
Headless (no display) — render to GIF instead of a window:
    python utils/wideband_live.py --save out.gif
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.signal as signal

# ---- choose a backend BEFORE importing pyplot -----------------------------
import matplotlib
_SAVE_MODE = "--save" in sys.argv
if _SAVE_MODE:
    matplotlib.use("Agg")
# else: leave the default interactive backend (qtagg/tkagg) for the pop-up
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner
from realtime.worker import decode_window

plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Defaults for the real BVSP captures (DMR_signal/README.txt + header decode).
BVSP_HEADER_BYTES = 112
BVSP_FS = 61.44e6
BVSP_CENTER = 431e6


def _psd_db(iq, fs, nperseg=4096):
    nperseg = min(nperseg, len(iq))
    if nperseg < 16:
        return np.array([0.0]), np.array([-120.0])
    f, p = signal.welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)
    f = np.fft.fftshift(f)
    p = np.fft.fftshift(p)
    return f, 10 * np.log10(p + 1e-12)


def _make_synth_scene(data_dir, fs, dur):
    from utils.synthesis import synthesize_wideband_grid
    s1 = os.path.join(data_dir, "dmr_1_78125.rawiq")
    s2 = os.path.join(data_dir, "dmr_2_78125.rawiq")
    if not (os.path.exists(s1) and os.path.exists(s2)):
        sys.exit(f"synthetic source files not found in {data_dir!r}")
    out = os.path.join(_ROOT, "output", "wb_live_scene.rawiq")
    synthesize_wideband_grid([(-1_800_000.0, "dmr_1_78125.rawiq"),
                              (+1_800_000.0, "dmr_2_78125.rawiq")],
                             out, fs_out=fs, dur_sec=dur, data_dir=data_dir)
    return out, [(-1_800_000.0 + 435e6), (1_800_000.0 + 435e6)]


def main():
    ap = argparse.ArgumentParser(description="Live wideband channelizer scan demo")
    ap.add_argument("--file", default=None,
                    help="capture file (.bvsp or .rawiq). Default: DMR_signal/5.bvsp")
    ap.add_argument("--synth", action="store_true",
                    help="use a synthesized 2-signal scene instead of a capture")
    ap.add_argument("--fs", type=float, default=None, help="sample rate Hz")
    ap.add_argument("--center", type=float, default=None, help="band center Hz")
    ap.add_argument("--header", type=int, default=None, help="file header bytes to skip")
    ap.add_argument("--nsub", type=int, default=None,
                    help="number of sub-bands (default 48 for captures, 4 for --synth)")
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--window-sec", type=float, default=0.5)
    ap.add_argument("--step-sec", type=float, default=0.25)
    ap.add_argument("--pause", type=float, default=0.25,
                    help="seconds to pause per scan step (playback speed)")
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--save", default=None,
                    help="headless: write the animation to this GIF instead of a window")
    args = ap.parse_args()

    # ---- resolve the input + parameters -----------------------------------
    expected = None
    if args.synth:
        fs = args.fs or 5e6
        path, expected = _make_synth_scene(args.data_dir, fs, dur=10.0)
        center = args.center if args.center is not None else 435e6
        header = 0
        if args.nsub is None:
            args.nsub = 4
    else:
        path = args.file or os.path.join(_ROOT, "DMR_signal", "5.bvsp")
        if not os.path.exists(path):
            sys.exit(f"capture not found: {path}")
        is_bvsp = path.endswith(".bvsp")
        fs = args.fs or (BVSP_FS if is_bvsp else 5e6)
        center = args.center if args.center is not None else (BVSP_CENTER if is_bvsp else 0.0)
        header = args.header if args.header is not None else (BVSP_HEADER_BYTES if is_bvsp else 0)
        if args.nsub is None:
            args.nsub = 48

    print(f"[setup] file={os.path.basename(path)} fs={fs/1e6:g}MHz "
          f"center={center/1e6:g}MHz header={header}B nsub={args.nsub} os={args.oversample}")

    # ---- stage 1: read + channelize (the slow part) -----------------------
    src = FileWidebandSource(path, sample_rate=fs, center_hz=center,
                             chunk_samples=int(fs), throttle=False, header_bytes=header)
    scanner = WidebandScanner(src, num_subbands=args.nsub, oversample=args.oversample,
                              window_sec=args.window_sec, step_sec=args.step_sec)
    print("[stage 1] reading + channelizing wideband capture ...")
    t0 = time.time()
    wide = scanner._read_all()
    if len(wide) == 0:
        sys.exit("capture is empty")
    subbands = scanner.channelizer.process(wide)
    centers = scanner.centers
    active = scanner._active_subbands(subbands)
    halfwidth = scanner._owning_halfwidth_hz
    print(f"[stage 1] done in {time.time()-t0:.1f}s — "
          f"{scanner.channelizer.N} sub-bands, active: {active}")

    wf, wdb = _psd_db(wide, fs, nperseg=4096)
    rf_lo = (center - fs / 2) / 1e6
    rf_hi = (center + fs / 2) / 1e6

    n_out = subbands.shape[1]
    n_windows = max(0, (n_out - scanner.window_samples) // scanner.step_samples + 1)

    # ---- figure scaffold ---------------------------------------------------
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1.3, 1.3, 0.9]})
    fig.suptitle("DMR Wideband Channelizer — live scan demo", fontsize=13, fontweight="bold")

    lit_calls = []  # accumulated (rf_hz, src, dst, flco)

    def render(wid, i, sub_win, detections, has_signal):
        for ax in (ax_top, ax_mid, ax_bot):
            ax.clear()

        # top: full-band PSD + moving scan box (absolute RF on the x-axis)
        ax_top.plot((center + wf) / 1e6, wdb, lw=0.7, color="#1f77b4")
        for cc in centers:
            ax_top.axvline((center + cc) / 1e6, color="#dddddd", lw=0.5, ls="--")
        if expected is not None:
            for rf in expected:
                ax_top.axvline(rf / 1e6, color="#d62728", lw=1.0, ls="--", alpha=0.5)
        c = centers[i]
        lo, hi = (center + c - halfwidth) / 1e6, (center + c + halfwidth) / 1e6
        box_color = "#2ca02c" if has_signal else "#ff7f0e"
        ax_top.set_xlim(rf_lo, rf_hi)
        ymin, ymax = ax_top.get_ylim()
        ax_top.add_patch(Rectangle((lo, ymin), hi - lo, ymax - ymin,
                                   facecolor=box_color, alpha=0.20,
                                   edgecolor=box_color, lw=2, zorder=4))
        abs_c = (center + c) / 1e6
        status = "SIGNAL -> decoding" if has_signal else "scanning (quiet)"
        ax_top.set_title(f"Scanning sub-band #{i} @ {abs_c:.3f} MHz  "
                         f"(window {wid+1}/{n_windows})  --  {status}")
        ax_top.set_xlabel("Absolute RF (MHz)")
        ax_top.set_ylabel("PSD (dB)")
        ax_top.grid(True, alpha=0.3)

        # middle: current sub-band baseband PSD + owning region + detections
        sf, spdb = _psd_db(sub_win, scanner.subband_rate)
        ax_mid.plot(sf / 1e3, spdb, lw=0.7, color="#1f77b4")
        ax_mid.axvspan(-halfwidth / 1e3, halfwidth / 1e3, color="#2ca02c", alpha=0.10)
        for fo_rel, owned in detections:
            ax_mid.axvline(fo_rel / 1e3, color="#2ca02c" if owned else "#999999",
                           lw=1.4 if owned else 0.9, ls="-" if owned else ":",
                           alpha=0.9 if owned else 0.6)
        ax_mid.set_title(f"Sub-band #{i} baseband — green band = owning region "
                         f"(+/-{halfwidth/1e3:.0f} kHz); green = owned (decoded), "
                         f"gray = alias (skipped)", fontsize=9)
        ax_mid.set_xlabel("Sub-band baseband frequency (kHz)")
        ax_mid.set_ylabel("PSD (dB)")
        ax_mid.grid(True, alpha=0.3)

        # bottom: decoded calls so far
        ax_bot.set_xlim(rf_lo, rf_hi)
        ax_bot.set_ylim(0, 1)
        ax_bot.set_yticks([])
        if expected is not None:
            for rf in expected:
                ax_bot.axvline(rf / 1e6, color="#d62728", lw=1.0, ls="--", alpha=0.5)
        for (rf_hz, csrc, cdst, cflco, cfid) in lit_calls:
            rf = rf_hz / 1e6
            ax_bot.scatter([rf], [0.5], s=140, marker="v", color="#2ca02c",
                           edgecolor="black", zorder=5)
            ax_bot.annotate(f"{rf:.4f}M\nSRC={csrc} DST={cdst}\nFID={cfid}",
                            (rf, 0.5), textcoords="offset points", xytext=(0, -12),
                            ha="center", va="top", fontsize=7)
        ax_bot.set_title(f"Decoded calls: {len(lit_calls)}")
        ax_bot.set_xlabel("Absolute RF (MHz)")
        ax_bot.grid(True, axis="x", alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.98])

    # ---- stage 2: live scan loop (faithful to WidebandScanner.run) ---------
    print("[stage 2] live scan — watch the window")
    frames_for_gif = []

    def known(rf_hz, src, dst):
        k = (round(rf_hz / 5000.0), src, dst)
        return k in {(round(a[0] / 5000.0), a[1], a[2]) for a in lit_calls}

    if args.save:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=max(1, int(1 / max(args.pause, 0.05))))
        writer.setup(fig, args.save, dpi=100)

    if not args.save:
        plt.ion()
        plt.show(block=False)

    for wid in range(n_windows):
        start = wid * scanner.step_samples
        stop = start + scanner.window_samples
        for i in active:
            win = subbands[i, start:stop]
            tasks = scanner._detectors[i].process_window(win, wid)
            detections = []
            for (iq, fo_rel, w) in tasks:
                owned = abs(fo_rel) <= halfwidth
                detections.append((fo_rel, owned))
                if not owned:
                    continue
                pdus = decode_window(iq, fo_rel, w, scanner.subband_rate)
                rf = scanner.center_hz + float(scanner.centers[i]) + fo_rel
                for pdu in pdus:
                    pdu["_rf_hz"] = rf
                    scanner.aggregator.feed(pdu)
                if pdus and not known(rf, pdus[0].get("src"), pdus[0].get("dst")):
                    lit_calls.append((rf, pdus[0].get("src"), pdus[0].get("dst"),
                                      pdus[0].get("flco", ""), pdus[0].get("fid", "")))
                    print(f"    [CALL] RF={rf/1e6:.4f}MHz SRC={pdus[0].get('src')} "
                          f"DST={pdus[0].get('dst')} FLCO={pdus[0].get('flco','')} "
                          f"FID={pdus[0].get('fid','')}")
            closed_rf = [scanner.center_hz + float(scanner.centers[i]) + fo
                         for fo in scanner._detectors[i].closed_channels()]
            scanner.aggregator.expire(wid, closed_rf)

            has_signal = any(o for _, o in detections)
            render(wid, i, win, detections, has_signal)
            if args.save:
                writer.grab_frame()
            else:
                plt.pause(args.pause)

    print(f"\n[done] decoded {len(lit_calls)} call(s):")
    for (rf_hz, csrc, cdst, cflco, cfid) in lit_calls:
        print(f"    RF={rf_hz/1e6:.4f}MHz SRC={csrc} DST={cdst} FLCO={cflco} FID={cfid}")

    if args.save:
        writer.finish()
        print(f"[saved] {args.save}")
    else:
        ax_top.set_title(ax_top.get_title() + "   [SCAN COMPLETE]")
        plt.ioff()
        print("[window] close the window to exit.")
        plt.show(block=True)


if __name__ == "__main__":
    main()
