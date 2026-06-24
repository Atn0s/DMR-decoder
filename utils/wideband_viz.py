"""Wideband channelizer pipeline visualizer.

Walks the real WidebandScanner chain stage by stage and plots the intermediate
data at each step, so the wideband -> channelize -> sub-band -> detect -> decode
flow (and the principle behind it) is visible at a glance, while also acting as
an end-to-end sanity check.

It reuses the real production components (PolyphaseChannelizer, WidebandScanner,
Detector) rather than re-implementing any DSP, so the figure reflects exactly
what the pipeline does — not an idealized stand-in.

Stages plotted:
  1. Wideband input  : time-domain snippet + full-band PSD with sub-band grid
  2. Sub-band energy : per-sub-band power (dB) + energy gate threshold
  3. Active sub-bands : baseband PSD of each active sub-band + owning region
  4. Decoded calls   : decoded calls placed on the absolute RF axis

Usage (run from the project root; no -m needed):
  python utils/wideband_viz.py [--fs HZ] [--dur SEC] [--nsub N]
                               [--oversample K] [--center HZ] [--out PATH]

With the project's dedicated interpreter:
  /home/lzkj/miniconda3/envs/DMR_demo/bin/python utils/wideband_viz.py
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

# Repo root on path so `realtime` / `utils` import when run as a plain script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from realtime.channelizer import PolyphaseChannelizer
from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner

# A CJK-capable fallback keeps any stray non-ASCII glyphs from turning into
# tofu boxes; all titles/labels below are deliberately English-only.
plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _psd_db(iq, fs, nperseg=4096):
    """Two-sided Welch PSD in dB, fftshifted to ascending frequency."""
    nperseg = min(nperseg, len(iq))
    f, p = signal.welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)
    f = np.fft.fftshift(f)
    p = np.fft.fftshift(p)
    return f, 10 * np.log10(p + 1e-12)


def _build_scenario(args):
    """Synthesize a small wideband grid scene; return its file path.

    Two DMR signals placed +/-1.8 MHz from band center — far enough apart that
    no single 2.5 MHz sub-band reaches both, so channelization is required.
    """
    from utils.synthesis import synthesize_wideband_grid

    src1 = os.path.join(args.data_dir, "dmr_1_78125.rawiq")
    src2 = os.path.join(args.data_dir, "dmr_2_78125.rawiq")
    if not (os.path.exists(src1) and os.path.exists(src2)):
        sys.exit(
            f"source narrowband files not found in {args.data_dir!r} "
            "(need dmr_1_78125.rawiq and dmr_2_78125.rawiq)"
        )
    out = os.path.join(_ROOT, "output", "wb_viz_scene.rawiq")
    placements = [(-1_800_000.0, "dmr_1_78125.rawiq"),
                  (+1_800_000.0, "dmr_2_78125.rawiq")]
    synthesize_wideband_grid(placements, out, fs_out=args.fs, dur_sec=args.dur,
                             data_dir=args.data_dir)
    return out, [p[0] for p in placements]


def main():
    ap = argparse.ArgumentParser(description="Wideband channelizer pipeline visualizer")
    ap.add_argument("--file", default=None,
                    help="existing wideband .rawiq file (default: synthesize a scene)")
    ap.add_argument("--fs", type=float, default=5e6, help="sample rate Hz (default 5e6)")
    ap.add_argument("--dur", type=float, default=10.0,
                    help="synthesized duration sec (default 10)")
    ap.add_argument("--nsub", type=int, default=4, help="number of sub-bands (default 4)")
    ap.add_argument("--oversample", type=int, default=2, help="oversample factor (default 2)")
    ap.add_argument("--center", type=float, default=435e6,
                    help="absolute RF band center Hz (default 435e6)")
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data"),
                    help="directory holding source narrowband .rawiq files")
    ap.add_argument("--out", default=os.path.join(_ROOT, "output", "wideband_viz.png"),
                    help="output PNG path")
    args = ap.parse_args()

    # --- Stage 0: obtain a wideband capture -----------------------------------
    expected = None
    if args.file:
        if not os.path.exists(args.file):
            sys.exit(f"input file not found: {args.file}")
        wb_path = args.file
    else:
        wb_path, expected = _build_scenario(args)

    # --- Stage 1: read wideband + channelize (real components) ----------------
    src = FileWidebandSource(wb_path, sample_rate=args.fs, center_hz=args.center,
                             chunk_samples=int(args.fs), throttle=False)
    scanner = WidebandScanner(src, num_subbands=args.nsub,
                              oversample=args.oversample,
                              window_sec=1.0, step_sec=0.9)

    wide = scanner._read_all()
    if len(wide) == 0:
        sys.exit("wideband capture is empty")
    subbands = scanner.channelizer.process(wide)        # (N, n_out)
    centers = scanner.centers                            # ascending, Hz (baseband)
    active = scanner._active_subbands(subbands)

    # Per-sub-band power and the energy-gate threshold (mirrors _active_subbands).
    power = np.mean(np.abs(subbands) ** 2, axis=1) + 1e-12
    power_db = 10 * np.log10(power)
    gate_floor = np.percentile(power_db, 25)
    gate_line = gate_floor + scanner.energy_floor_db

    # --- Stage 4: decode (real end-to-end run) --------------------------------
    # Re-open the source: scanner._read_all() consumed and closed the first one.
    src2 = FileWidebandSource(wb_path, sample_rate=args.fs, center_hz=args.center,
                              chunk_samples=int(args.fs), throttle=False)
    scanner2 = WidebandScanner(src2, num_subbands=args.nsub,
                               oversample=args.oversample,
                               window_sec=1.0, step_sec=0.9)
    calls = scanner2.run()

    # ==========================================================================
    # Figure
    # ==========================================================================
    n_active = len(active)
    fig = plt.figure(figsize=(15, 4 + 2.6 * max(1, n_active)))
    gs = fig.add_gridspec(3 + max(1, n_active), 2,
                          height_ratios=[1.4, 1.4, 1.0] + [1.0] * max(1, n_active))

    # --- Panel 1a: wideband time-domain snippet -------------------------------
    ax_t = fig.add_subplot(gs[0, 0])
    snip = wide[: min(len(wide), 4000)]
    t_us = np.arange(len(snip)) / args.fs * 1e6
    ax_t.plot(t_us, snip.real, lw=0.6, color="#1f77b4")
    ax_t.set_title("Stage 1: Wideband input (time domain, real part)")
    ax_t.set_xlabel("Time (us)")
    ax_t.set_ylabel("Amplitude")
    ax_t.grid(True, alpha=0.3)

    # --- Panel 1b: wideband PSD with sub-band grid ----------------------------
    ax_f = fig.add_subplot(gs[0, 1])
    f, pdb = _psd_db(wide, args.fs)
    ax_f.plot(f / 1e6, pdb, lw=0.7, color="#1f77b4")
    half = args.fs / args.nsub / 2.0  # sub-band half spacing (fs/N/2)
    for i, c in enumerate(centers):
        is_active = i in active
        ax_f.axvline(c / 1e6, color="#2ca02c" if is_active else "#bbbbbb",
                     lw=1.4 if is_active else 0.8, ls="-" if is_active else "--",
                     alpha=0.8)
        # boundaries between sub-bands
        ax_f.axvline((c - half) / 1e6, color="#dddddd", lw=0.5, ls=":")
    if expected is not None:
        for fo in expected:
            ax_f.axvline(fo / 1e6, color="#d62728", lw=1.2, ls="--", alpha=0.7)
    ax_f.set_title("Stage 1: Wideband PSD + sub-band grid\n"
                   "(green=active sub-band center, red dashed=injected signal)")
    ax_f.set_xlabel("Baseband frequency (MHz)")
    ax_f.set_ylabel("PSD (dB)")
    ax_f.grid(True, alpha=0.3)

    # --- Panel 2: per-sub-band energy + gate ----------------------------------
    ax_e = fig.add_subplot(gs[1, :])
    idx = np.arange(scanner.channelizer.N)
    colors = ["#2ca02c" if i in active else "#cccccc" for i in idx]
    ax_e.bar(idx, power_db, color=colors, edgecolor="black", lw=0.5)
    ax_e.axhline(gate_line, color="#d62728", lw=1.4, ls="--",
                 label=f"gate = pct25 + {scanner.energy_floor_db:g} dB")
    ax_e.axhline(gate_floor, color="#999999", lw=1.0, ls=":",
                 label="noise floor (pct25)")
    ax_e.set_title("Stage 2: Sub-band energy map "
                   "(green=passes gate -> sent to decode, gray=skipped)")
    ax_e.set_xlabel("Sub-band index (ascending frequency)")
    ax_e.set_ylabel("Mean power (dB)")
    ax_e.set_xticks(idx)
    ax_e.set_xticklabels([f"{i}\n{centers[i] / 1e6:+.2f}M" for i in idx], fontsize=8)
    ax_e.legend(loc="upper right", fontsize=8)
    ax_e.grid(True, axis="y", alpha=0.3)

    # --- Panel 3: each active sub-band baseband PSD + owning region ------------
    halfwidth = scanner._owning_halfwidth_hz
    if not active:
        ax0 = fig.add_subplot(gs[3, :])
        ax0.text(0.5, 0.5, "No active sub-bands passed the energy gate.",
                 ha="center", va="center", fontsize=11)
        ax0.axis("off")
    for row, i in enumerate(active):
        ax = fig.add_subplot(gs[3 + row, :])
        sf, spdb = _psd_db(subbands[i], scanner.subband_rate)
        ax.plot(sf / 1e3, spdb, lw=0.7, color="#1f77b4")
        # owning region (primary): +/- halfwidth around sub-band center (baseband)
        ax.axvspan(-halfwidth / 1e3, halfwidth / 1e3, color="#2ca02c", alpha=0.12)
        ax.axvline(-halfwidth / 1e3, color="#2ca02c", lw=0.8, ls="--")
        ax.axvline(halfwidth / 1e3, color="#2ca02c", lw=0.8, ls="--")
        abs_center = (args.center + centers[i]) / 1e6
        ax.set_title(f"Stage 3: Active sub-band #{i}  "
                     f"(center {centers[i] / 1e6:+.2f} MHz baseband, "
                     f"abs RF {abs_center:.3f} MHz) -- "
                     f"green band = owning region (+/-{halfwidth / 1e3:.0f} kHz)",
                     fontsize=9)
        ax.set_xlabel("Sub-band baseband frequency (kHz)")
        ax.set_ylabel("PSD (dB)")
        ax.grid(True, alpha=0.3)

    # --- Panel 4: decoded calls on absolute RF axis ---------------------------
    ax_c = fig.add_subplot(gs[2, :])
    rf_lo = (args.center - args.fs / 2) / 1e6
    rf_hi = (args.center + args.fs / 2) / 1e6
    ax_c.set_xlim(rf_lo, rf_hi)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([])
    if expected is not None:
        for fo in expected:
            rf = (args.center + fo) / 1e6
            ax_c.axvline(rf, color="#d62728", lw=1.2, ls="--", alpha=0.6)
            ax_c.text(rf, 0.92, f"injected\n{rf:.3f}M", ha="center", va="top",
                      fontsize=7, color="#d62728")
    voice = [c for c in calls if c.flco == "GroupVoiceChannelUser"]
    for c in calls:
        rf = c.fo_hz / 1e6
        is_voice = c.flco == "GroupVoiceChannelUser"
        ax_c.scatter([rf], [0.4], s=120,
                     marker="v" if is_voice else "o",
                     color="#2ca02c" if is_voice else "#ff7f0e",
                     edgecolor="black", zorder=5)
        ax_c.annotate(f"{rf:.3f}M\nSRC={c.src} DST={c.dst}\n{c.flco}\n[{c.closed_by}]",
                      (rf, 0.4), textcoords="offset points", xytext=(0, -10),
                      ha="center", va="top", fontsize=6.5)
    ax_c.set_title(f"Stage 4: Decoded calls on absolute RF axis "
                   f"({len(voice)} voice / {len(calls)} total) -- "
                   f"should align with injected (red dashed)")
    ax_c.set_xlabel("Absolute RF (MHz)")
    ax_c.grid(True, axis="x", alpha=0.3)

    fig.suptitle("DMR Wideband Channelizer Pipeline -- stage-by-stage visualization",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(args.out, dpi=100)

    # --- Console summary (the sanity check) -----------------------------------
    print(f"saved {args.out}")
    print(f"  fs={args.fs/1e6:g} MHz  nsub={args.nsub}  oversample={args.oversample}  "
          f"subband_rate={scanner.subband_rate/1e6:g} MHz")
    print(f"  active sub-bands: {active}  (of {scanner.channelizer.N})")
    print(f"  decoded calls: {len(calls)}  (voice: {len(voice)})")
    for c in calls:
        print(f"    RF={c.fo_hz/1e6:.4f}MHz SRC={c.src} DST={c.dst} "
              f"FLCO={c.flco} closed_by={c.closed_by}")
    if expected is not None:
        print(f"  injected at: {[ (args.center+fo)/1e6 for fo in expected ]} MHz")


if __name__ == "__main__":
    main()
