"""Wideband channelizer pipeline — animated (real-time) visualizer.

Animates the *time progression* of the scan: the WidebandScanner advances
window by window and, within each window, sweeps across the active sub-bands.
This script highlights which sub-band is being inspected right now, shows the
detector firing when a signal is present, and lights up decoded calls on the
absolute-RF axis as they are recovered.

Each animation frame = one (window, sub-band) inspection step, mirroring the
exact loop of WidebandScanner.run() so the animation is faithful to production
behavior (it reuses the real channelizer / detector / decode_window).

Layout (3 stacked axes):
  top    : full-band PSD with a moving highlight box = the sub-band being scanned
  middle : that sub-band's baseband PSD + owning region + detected peaks
  bottom : absolute-RF axis accumulating decoded calls over time

Output: an animated GIF (Pillow). Use --mp4 for an MP4 via ffmpeg.

Usage (run from the project root; no -m needed):
  python utils/wideband_anim.py \
      [--fs HZ] [--dur SEC] [--nsub N] [--oversample K] [--center HZ] \
      [--out PATH] [--mp4] [--fps N]
"""
import argparse
import os
import sys

import numpy as np
import scipy.signal as signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner
from realtime.worker import decode_window

plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _psd_db(iq, fs, nperseg=2048):
    nperseg = min(nperseg, len(iq))
    if nperseg < 16:
        return np.array([0.0]), np.array([-120.0])
    f, p = signal.welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)
    f = np.fft.fftshift(f)
    p = np.fft.fftshift(p)
    return f, 10 * np.log10(p + 1e-12)


def _build_scenario(args):
    from utils.synthesis import synthesize_wideband_grid

    src1 = os.path.join(args.data_dir, "dmr_1_78125.rawiq")
    src2 = os.path.join(args.data_dir, "dmr_2_78125.rawiq")
    if not (os.path.exists(src1) and os.path.exists(src2)):
        sys.exit(f"source narrowband files not found in {args.data_dir!r}")
    out = os.path.join(_ROOT, "output", "wb_anim_scene.rawiq")
    placements = [(-1_800_000.0, "dmr_1_78125.rawiq"),
                  (+1_800_000.0, "dmr_2_78125.rawiq")]
    synthesize_wideband_grid(placements, out, fs_out=args.fs, dur_sec=args.dur,
                             data_dir=args.data_dir)
    return out, [p[0] for p in placements]


def _capture_frames(scanner, subbands, active):
    """Replay WidebandScanner.run()'s loop, recording one frame per
    (window, active sub-band) step.  Logic mirrors wideband_scanner.run()."""
    frames = []
    n_out = subbands.shape[1]
    n_windows = max(0, (n_out - scanner.window_samples) // scanner.step_samples + 1)
    halfwidth = scanner._owning_halfwidth_hz
    accepted_calls = []  # (rf_hz, src, dst, flco) lit so far

    for wid in range(n_windows):
        start = wid * scanner.step_samples
        stop = start + scanner.window_samples
        for i in active:
            win = subbands[i, start:stop]
            tasks = scanner._detectors[i].process_window(win, wid)

            detections = []   # (fo_rel, owned_bool)
            new_calls = []
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
                if pdus:
                    new_calls.append((rf, pdus[0].get("src"), pdus[0].get("dst"),
                                      pdus[0].get("flco", "")))
            # advance detector close path (kept faithful; result unused for viz)
            closed_rf = [scanner.center_hz + float(scanner.centers[i]) + fo
                         for fo in scanner._detectors[i].closed_channels()]
            scanner.aggregator.expire(wid, closed_rf)

            # accumulate distinct lit calls by (rounded rf, src, dst)
            for nc in new_calls:
                key = (round(nc[0] / 5000.0), nc[1], nc[2])
                if key not in {(round(a[0] / 5000.0), a[1], a[2]) for a in accepted_calls}:
                    accepted_calls.append(nc)

            # snapshot
            sf, spdb = _psd_db(win, scanner.subband_rate)
            frames.append({
                "wid": wid, "n_windows": n_windows, "sub": i,
                "t_start": start / scanner.subband_rate,
                "sub_psd_f": sf, "sub_psd_db": spdb,
                "detections": detections,
                "has_signal": any(o for _, o in detections),
                "calls": list(accepted_calls),
            })
    return frames


def main():
    ap = argparse.ArgumentParser(description="Animated wideband channelizer pipeline")
    ap.add_argument("--file", default=None)
    ap.add_argument("--fs", type=float, default=5e6)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--nsub", type=int, default=4)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--center", type=float, default=435e6)
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--out", default=None, help="output path (default GIF in output/)")
    ap.add_argument("--mp4", action="store_true", help="write MP4 (ffmpeg) instead of GIF")
    ap.add_argument("--fps", type=int, default=2, help="frames per second (default 2)")
    args = ap.parse_args()

    expected = None
    if args.file:
        if not os.path.exists(args.file):
            sys.exit(f"input file not found: {args.file}")
        wb_path = args.file
    else:
        wb_path, expected = _build_scenario(args)

    src = FileWidebandSource(wb_path, sample_rate=args.fs, center_hz=args.center,
                             chunk_samples=int(args.fs), throttle=False)
    scanner = WidebandScanner(src, num_subbands=args.nsub, oversample=args.oversample,
                              window_sec=1.0, step_sec=0.9)
    wide = scanner._read_all()
    if len(wide) == 0:
        sys.exit("wideband capture is empty")
    subbands = scanner.channelizer.process(wide)
    centers = scanner.centers
    active = scanner._active_subbands(subbands)
    halfwidth = scanner._owning_halfwidth_hz

    # full-band PSD (static background for the top panel)
    wf, wdb = _psd_db(wide, args.fs, nperseg=4096)

    frames = _capture_frames(scanner, subbands, active)
    if not frames:
        sys.exit("no scan steps to animate (capture too short for one window)")

    # ----- figure scaffold -----------------------------------------------------
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1.3, 1.3, 0.9]})

    rf_lo = (args.center - args.fs / 2) / 1e6
    rf_hi = (args.center + args.fs / 2) / 1e6

    def draw(frame_idx):
        fr = frames[frame_idx]
        i = fr["sub"]
        for ax in (ax_top, ax_mid, ax_bot):
            ax.clear()

        # --- top: full-band PSD + moving scan highlight ---
        ax_top.plot(wf / 1e6, wdb, lw=0.7, color="#1f77b4")
        for j, c in enumerate(centers):
            ax_top.axvline(c / 1e6, color="#cccccc", lw=0.6, ls="--")
        if expected is not None:
            for fo in expected:
                ax_top.axvline(fo / 1e6, color="#d62728", lw=1.0, ls="--", alpha=0.5)
        # highlight box over the sub-band currently being scanned
        c = centers[i]
        lo, hi = (c - halfwidth) / 1e6, (c + halfwidth) / 1e6
        box_color = "#2ca02c" if fr["has_signal"] else "#ff7f0e"
        ymin, ymax = ax_top.get_ylim()
        ax_top.add_patch(Rectangle((lo, ymin), hi - lo, ymax - ymin,
                                   facecolor=box_color, alpha=0.20,
                                   edgecolor=box_color, lw=2, zorder=4))
        status = "SIGNAL -> decode" if fr["has_signal"] else "scanning (quiet)"
        ax_top.set_title(
            f"Scanning sub-band #{i}  (window {fr['wid'] + 1}/{fr['n_windows']}, "
            f"t={fr['t_start']:.2f}s)  --  {status}")
        ax_top.set_xlabel("Baseband frequency (MHz)")
        ax_top.set_ylabel("PSD (dB)")
        ax_top.grid(True, alpha=0.3)

        # --- middle: current sub-band baseband PSD + owning region + peaks ---
        ax_mid.plot(fr["sub_psd_f"] / 1e3, fr["sub_psd_db"], lw=0.7, color="#1f77b4")
        ax_mid.axvspan(-halfwidth / 1e3, halfwidth / 1e3, color="#2ca02c", alpha=0.10)
        for fo_rel, owned in fr["detections"]:
            ax_mid.axvline(fo_rel / 1e3, color="#2ca02c" if owned else "#999999",
                           lw=1.4 if owned else 0.9,
                           ls="-" if owned else ":",
                           alpha=0.9 if owned else 0.6)
        abs_center = (args.center + c) / 1e6
        ax_mid.set_title(
            f"Sub-band #{i} baseband (abs RF {abs_center:.3f} MHz)  --  "
            f"green band = owning region (+/-{halfwidth / 1e3:.0f} kHz); "
            f"green line = owned detection (decoded), gray = alias (skipped)",
            fontsize=9)
        ax_mid.set_xlabel("Sub-band baseband frequency (kHz)")
        ax_mid.set_ylabel("PSD (dB)")
        ax_mid.grid(True, alpha=0.3)

        # --- bottom: accumulated decoded calls on absolute RF ---
        ax_bot.set_xlim(rf_lo, rf_hi)
        ax_bot.set_ylim(0, 1)
        ax_bot.set_yticks([])
        if expected is not None:
            for fo in expected:
                rf = (args.center + fo) / 1e6
                ax_bot.axvline(rf, color="#d62728", lw=1.0, ls="--", alpha=0.5)
        for (rf_hz, csrc, cdst, cflco) in fr["calls"]:
            rf = rf_hz / 1e6
            ax_bot.scatter([rf], [0.5], s=130, marker="v", color="#2ca02c",
                           edgecolor="black", zorder=5)
            ax_bot.annotate(f"{rf:.3f}M\nSRC={csrc} DST={cdst}",
                            (rf, 0.5), textcoords="offset points", xytext=(0, -12),
                            ha="center", va="top", fontsize=7)
        ax_bot.set_title(f"Decoded calls so far: {len(fr['calls'])} "
                         f"(should converge to the injected red-dashed positions)")
        ax_bot.set_xlabel("Absolute RF (MHz)")
        ax_bot.grid(True, axis="x", alpha=0.3)

        fig.suptitle("DMR Wideband Channelizer -- live scan animation",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / args.fps)

    if args.out is None:
        ext = "mp4" if args.mp4 else "gif"
        args.out = os.path.join(_ROOT, "output", f"wideband_anim.{ext}")
    if args.mp4:
        anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    else:
        anim.save(args.out, writer=PillowWriter(fps=args.fps))

    print(f"saved {args.out}  ({len(frames)} frames @ {args.fps} fps)")
    print(f"  active sub-bands: {active}  decoded calls: {len(frames[-1]['calls'])}")
    for (rf_hz, csrc, cdst, cflco) in frames[-1]["calls"]:
        print(f"    RF={rf_hz/1e6:.4f}MHz SRC={csrc} DST={cdst} FLCO={cflco}")


if __name__ == "__main__":
    main()
