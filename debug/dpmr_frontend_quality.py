#!/usr/bin/env python3
"""Visualize dPMR pre-decision 4-level quality.

This script intentionally does not decode CC/CCH/CRC.  It uses the dPMR FS2
sync word only as a physical-layer anchor so the plots show symbol samples from
real dPMR frames.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from core.burst_type import DEV_NOMINAL
from core.dsp import _interp, read_rawiq
from dpmr.cch import (
    air_interface_id_to_str,
    decode_cch,
    deinterleave_6x12,
    descramble,
    hamming_12_8_decode,
)
from dpmr.constants import (
    CCH_SYMBOLS,
    CC_SYMBOLS,
    FS_DEC,
    FS2_SYMBOL_COUNT,
    LEVEL_TO_DIBIT,
    SPS,
    TCH_SYMBOLS,
    VOICE_FS2_TOTAL_SYMBOLS,
)
from dpmr.dsp import DPMRSyncCandidate, find_fs2_sync, recover_voice_fs2_symbol_candidates, split_voice_fs2


LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])


@dataclass(frozen=True)
class ChainResult:
    name: str
    y: np.ndarray
    candidate: DPMRSyncCandidate | None
    positions: np.ndarray
    raw_symbols: np.ndarray
    calibrated_symbols: np.ndarray
    p90_distance: float
    ambiguous_ratio: float
    cluster_spread: float
    phase: float | None
    carrier_hz: float


@dataclass(frozen=True)
class CCHStageResult:
    name: str
    raw_bits: list[int]
    descrambled_bits: list[int]
    deinterleaved_bits: list[int]
    decoded_bits: list[int]
    block_ok: list[bool]
    corrected_bits: list[int]
    record_crc_ok: bool | None
    record_frame_number: int | None
    record_crc_value: int | None
    record_crc_computed: int | None


SEGMENTS = [
    ("FS2", 0, FS2_SYMBOL_COUNT),
    ("CCH0", FS2_SYMBOL_COUNT, FS2_SYMBOL_COUNT + CCH_SYMBOLS),
    (
        "CC",
        FS2_SYMBOL_COUNT + CCH_SYMBOLS + TCH_SYMBOLS,
        FS2_SYMBOL_COUNT + CCH_SYMBOLS + TCH_SYMBOLS + CC_SYMBOLS,
    ),
    (
        "CCH1",
        FS2_SYMBOL_COUNT + CCH_SYMBOLS + TCH_SYMBOLS + CC_SYMBOLS,
        FS2_SYMBOL_COUNT + CCH_SYMBOLS + TCH_SYMBOLS + CC_SYMBOLS + CCH_SYMBOLS,
    ),
]


def fm_frontend(
    iq: np.ndarray,
    *,
    fs: float,
    cutoff: float,
    ntaps: int = 151,
    integrate_symbols: bool = False,
    sps: int = SPS,
) -> tuple[np.ndarray, float]:
    """DMR-style FM front end with optional integrate-and-dump matched filter."""
    f, ps = signal.welch(iq, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    ps = np.fft.fftshift(ps)
    carrier_hz = float(f[np.argmax(ps)])

    n = np.arange(len(iq))
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * carrier_hz * n / fs)
    taps = signal.firwin(ntaps, cutoff, fs=fs)
    iq_filtered = signal.filtfilt(taps, [1.0], iq_shifted)

    demod = np.angle(iq_filtered[1:] * np.conj(iq_filtered[:-1]))
    amp = np.abs(iq_filtered[:-1])
    active = amp > (np.median(amp) + 0.3 * (np.mean(amp) - np.median(amp)))
    center = np.median(demod[active]) if np.any(active) else np.median(demod)
    y = (demod - center) * (3.0 / (2.0 * np.pi * DEV_NOMINAL / fs))

    if integrate_symbols:
        y = np.convolve(y, np.ones(sps) / sps, mode="same")
    return np.clip(y, -6.0, 6.0), carrier_hz


def gardner_timing_recovery(x: np.ndarray, sps: int = SPS) -> tuple[np.ndarray, np.ndarray]:
    """DMR native timing loop adapted to dPMR's 2400 sym/s at 48 kHz."""
    mu = 0.0
    idx = sps
    out: list[float] = []
    pos_out: list[int] = []
    kp = 0.01
    ki = kp / 50.0
    integ = 0.0
    prev_sym = 0.0

    while idx < len(x) - sps - 2:
        pos = idx + mu
        cur = float(_interp(x, np.array([pos]))[0])
        mid = float(_interp(x, np.array([pos - sps / 2.0]))[0])
        err = mid * (cur - prev_sym)
        integ += ki * err
        mu += kp * err + integ
        idx += sps
        while mu >= 1.0:
            mu -= 1.0
            idx += 1
        while mu < 0.0:
            mu += 1.0
            idx -= 1
        out.append(cur)
        pos_out.append(int(round(pos)))
        prev_sym = cur
    return np.asarray(out), np.asarray(pos_out)


def normalize_four_levels(symbols: np.ndarray) -> np.ndarray:
    """Affine-normalize a symbol cloud using quantiles only, no protocol fields."""
    if len(symbols) < 16:
        return symbols.astype(float)
    quantiles = np.percentile(symbols, [10, 35, 65, 90])
    a, b = np.linalg.lstsq(
        np.vstack([quantiles, np.ones(len(quantiles))]).T,
        LEVELS,
        rcond=None,
    )[0]
    calibrated = a * symbols + b
    if np.mean(np.abs(calibrated - nearest_levels(calibrated))) > np.mean(
        np.abs(-calibrated - nearest_levels(-calibrated))
    ):
        calibrated = -calibrated
    return calibrated


def nearest_levels(symbols: np.ndarray) -> np.ndarray:
    return LEVELS[np.argmin(np.abs(symbols[:, None] - LEVELS[None, :]), axis=1)]


def cloud_metrics(calibrated: np.ndarray) -> tuple[float, float, float]:
    nearest = nearest_levels(calibrated)
    distance = np.abs(calibrated - nearest)
    assigned_spreads = []
    for level in LEVELS:
        group = calibrated[nearest == level]
        if len(group) >= 3:
            assigned_spreads.append(float(np.median(np.abs(group - np.median(group)))))
    spread = float(np.median(assigned_spreads)) if assigned_spreads else float("nan")
    return (
        float(np.percentile(distance, 90)),
        float(np.mean(distance > 0.5)),
        spread,
    )


def segment_slice(name: str) -> slice:
    for seg_name, start, stop in SEGMENTS:
        if seg_name == name:
            return slice(start, stop)
    raise KeyError(name)


def symbol_error(calibrated: np.ndarray) -> np.ndarray:
    return np.abs(calibrated - nearest_levels(calibrated))


def segment_metrics(calibrated: np.ndarray) -> dict[str, tuple[float, float, float]]:
    metrics: dict[str, tuple[float, float, float]] = {}
    for name, start, stop in SEGMENTS:
        segment = calibrated[start:stop]
        if len(segment):
            metrics[name] = cloud_metrics(segment)
    return metrics


def calibrated_to_dibits(calibrated: np.ndarray) -> np.ndarray:
    nearest = nearest_levels(calibrated).astype(int)
    return np.asarray([LEVEL_TO_DIBIT[int(level)] for level in nearest], dtype=int)


def recover_fixed_phase_symbols(
    y: np.ndarray,
    candidate: DPMRSyncCandidate,
    *,
    sps: int = SPS,
) -> tuple[np.ndarray, np.ndarray, float]:
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for phase in np.linspace(-sps, sps, 81):
        positions = candidate.fs_start + phase + np.arange(VOICE_FS2_TOTAL_SYMBOLS) * sps
        if positions[0] < 0 or positions[-1] >= len(y) - 1:
            continue
        raw = _interp(y, positions)
        calibrated = normalize_four_levels(raw)
        p90, ambiguous_ratio, _ = cloud_metrics(calibrated)
        score = p90 + 0.5 * ambiguous_ratio
        if best is None or score < best[0]:
            best = (score, positions, raw, float(phase))
    if best is None:
        return np.array([]), np.array([]), float("nan")
    _, positions, raw, phase = best
    return positions, raw, phase


def recover_gardner_symbols(
    y: np.ndarray,
    candidate: DPMRSyncCandidate,
    *,
    sps: int = SPS,
) -> tuple[np.ndarray, np.ndarray]:
    symbols, positions = gardner_timing_recovery(y, sps=sps)
    lo = candidate.fs_start - sps
    hi = candidate.fs_start + VOICE_FS2_TOTAL_SYMBOLS * sps + sps
    mask = (positions >= lo) & (positions <= hi)
    return positions[mask], symbols[mask]


def analyze_chain(
    iq: np.ndarray,
    *,
    name: str,
    cutoff: float,
    integrate_symbols: bool,
    timing: str,
    fs: float,
) -> ChainResult:
    y, carrier_hz = fm_frontend(
        iq,
        fs=fs,
        cutoff=cutoff,
        integrate_symbols=integrate_symbols,
    )
    candidates = find_fs2_sync(y, threshold=0.68)
    candidate = max(candidates, key=lambda item: item.ncc) if candidates else None
    if candidate is None:
        empty = np.array([])
        return ChainResult(name, y, None, empty, empty, empty, np.nan, np.nan, np.nan, None, carrier_hz)

    if timing == "gardner":
        positions, raw_symbols = recover_gardner_symbols(y, candidate)
        phase = None
    else:
        positions, raw_symbols, phase = recover_fixed_phase_symbols(y, candidate)

    calibrated = normalize_four_levels(raw_symbols)
    p90, ambiguous_ratio, spread = cloud_metrics(calibrated)
    return ChainResult(
        name,
        y,
        candidate,
        positions,
        raw_symbols,
        calibrated,
        p90,
        ambiguous_ratio,
        spread,
        phase,
        carrier_hz,
    )


def plot_results(results: list[ChainResult], output: Path) -> None:
    fig = plt.figure(figsize=(16, 4.5 * len(results)))
    grid = fig.add_gridspec(len(results), 3, hspace=0.36, wspace=0.22)

    for row, result in enumerate(results):
        ax_wave = fig.add_subplot(grid[row, 0])
        ax_scatter = fig.add_subplot(grid[row, 1])
        ax_hist = fig.add_subplot(grid[row, 2])

        if result.candidate is not None:
            start = max(0, result.candidate.fs_start - 120)
            stop = min(len(result.y), result.candidate.fs_start + 900)
        else:
            start = 0
            stop = min(len(result.y), 1020)
        xs = np.arange(start, stop)
        ax_wave.plot(xs, result.y[start:stop], lw=0.65, color="#2f6f9f")
        for level in LEVELS:
            ax_wave.axhline(level, color="#999999", lw=0.6, ls="--", alpha=0.5)
        ax_wave.set_title(f"{result.name}: demod waveform")
        ax_wave.set_xlabel("sample")
        ax_wave.set_ylabel("level")
        ax_wave.grid(True, alpha=0.25)

        if len(result.calibrated_symbols):
            symbol_x = np.arange(len(result.calibrated_symbols))
            ax_scatter.scatter(symbol_x, result.calibrated_symbols, s=10, alpha=0.65, color="#1f77b4")
            for level in LEVELS:
                ax_scatter.axhline(level, color="#d62728", lw=0.8, ls="--", alpha=0.65)
            ax_scatter.set_ylim(-4.8, 4.8)
            phase_text = "Gardner" if result.phase is None else f"phase={result.phase:.2f}"
            ax_scatter.set_title(
                f"pre-decision samples ({phase_text}, p90={result.p90_distance:.3f}, "
                f"amb={result.ambiguous_ratio:.1%})"
            )
            ax_scatter.set_xlabel("symbol index near FS2")
            ax_scatter.set_ylabel("normalized symbol value")
            ax_scatter.grid(True, alpha=0.25)

            ax_hist.hist(result.calibrated_symbols, bins=90, color="#ff7f0e", alpha=0.78)
            for level in LEVELS:
                ax_hist.axvline(level, color="#d62728", lw=0.9, ls="--", alpha=0.75)
            ax_hist.set_xlim(-5, 5)
            ax_hist.set_title(f"4-level histogram (spread={result.cluster_spread:.3f})")
            ax_hist.set_xlabel("normalized symbol value")
            ax_hist.set_ylabel("count")
            ax_hist.grid(True, alpha=0.25)
        else:
            ax_scatter.text(0.5, 0.5, "no FS2-anchored symbols", ha="center", va="center")
            ax_hist.text(0.5, 0.5, "no histogram", ha="center", va="center")

    fig.suptitle("dPMR physical-layer 4-level quality only: no CC/CCH/CRC used", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timing_drift(results: list[ChainResult], output: Path) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 3.8 * len(results)), squeeze=False)
    colors = {"FS2": "#1f77b4", "CCH0": "#2ca02c", "CC": "#d62728", "CCH1": "#9467bd"}

    for row, result in enumerate(results):
        ax_err = axes[row, 0]
        ax_seg = axes[row, 1]
        if not len(result.calibrated_symbols):
            ax_err.text(0.5, 0.5, "no symbols", ha="center", va="center")
            ax_seg.text(0.5, 0.5, "no symbols", ha="center", va="center")
            continue

        err = symbol_error(result.calibrated_symbols)
        ax_err.plot(np.arange(len(err)), err, color="#1f77b4", lw=0.9)
        ax_err.axhline(0.5, color="#d62728", ls="--", lw=0.8, alpha=0.7)
        for name, start, stop in SEGMENTS:
            ax_err.axvspan(start, stop, color=colors[name], alpha=0.08)
            ax_err.text((start + stop) / 2.0, max(0.55, np.percentile(err, 95)), name, ha="center", va="bottom", fontsize=8)
        ax_err.set_title(f"{result.name}: symbol error vs index")
        ax_err.set_xlabel("symbol index")
        ax_err.set_ylabel("|sample - nearest level|")
        ax_err.grid(True, alpha=0.25)

        centers = []
        p90s = []
        ambs = []
        for name, start, stop in SEGMENTS:
            segment = result.calibrated_symbols[start:stop]
            if not len(segment):
                continue
            p90, ambiguous_ratio, _ = cloud_metrics(segment)
            centers.append((start + stop - 1) / 2.0)
            p90s.append(p90)
            ambs.append(ambiguous_ratio)
            ax_seg.scatter(np.arange(start, stop), segment, s=12, alpha=0.7, color=colors[name], label=name)
        for level in LEVELS:
            ax_seg.axhline(level, color="#999999", ls="--", lw=0.7, alpha=0.5)
        ax_seg.set_title(
            f"segment view: "
            + " ".join(
                f"{name} p90={cloud_metrics(result.calibrated_symbols[start:stop])[0]:.3f}"
                for name, start, stop in SEGMENTS
                if len(result.calibrated_symbols[start:stop])
            )
        )
        ax_seg.set_xlabel("symbol index")
        ax_seg.set_ylabel("normalized symbol value")
        ax_seg.set_ylim(-4.8, 4.8)
        ax_seg.grid(True, alpha=0.25)
        if row == 0:
            ax_seg.legend(loc="upper right", ncol=4, fontsize=8)

    fig.suptitle("dPMR symbol timing diagnostics: drift and segment quality", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def extract_current_cch_stages(result: ChainResult) -> list[CCHStageResult]:
    if result.candidate is None or result.phase is None:
        return []
    symbol_candidates = recover_voice_fs2_symbol_candidates(
        result.y,
        result.candidate,
        phase_search=np.array([result.phase]),
        sps_search=np.array([float(SPS)]),
        sample_windows=(0,),
        limit=1,
    )
    if not symbol_candidates:
        return []
    dibits = symbol_candidates[0].symbols
    cch0_bits, _, cch1_bits = split_voice_fs2(dibits)
    stages: list[CCHStageResult] = []
    for name, raw_bits in (("CCH0", cch0_bits), ("CCH1", cch1_bits)):
        descrambled_bits = descramble(raw_bits)
        deinterleaved_bits = deinterleave_6x12(descrambled_bits)
        decoded_bits: list[int] = []
        block_ok: list[bool] = []
        corrected_bits: list[int] = []
        for idx in range(6):
            decoded, ok, corrected = hamming_12_8_decode(
                deinterleaved_bits[idx * 12:(idx + 1) * 12]
            )
            decoded_bits.extend(decoded)
            block_ok.append(ok)
            corrected_bits.append(corrected)
        record = decode_cch(raw_bits)
        stages.append(
            CCHStageResult(
                name=name,
                raw_bits=raw_bits,
                descrambled_bits=descrambled_bits,
                deinterleaved_bits=deinterleaved_bits,
                decoded_bits=decoded_bits,
                block_ok=block_ok,
                corrected_bits=corrected_bits,
                record_crc_ok=(record.crc_ok if record is not None else None),
                record_frame_number=(record.frame_number if record is not None else None),
                record_crc_value=(record.crc_value if record is not None else None),
                record_crc_computed=(record.crc_computed if record is not None else None),
            )
        )
    return stages


def extract_current_cch_dibits(result: ChainResult) -> dict[str, np.ndarray]:
    if result.candidate is None or result.phase is None:
        return {}
    symbol_candidates = recover_voice_fs2_symbol_candidates(
        result.y,
        result.candidate,
        phase_search=np.array([result.phase]),
        sps_search=np.array([float(SPS)]),
        sample_windows=(0,),
        limit=1,
    )
    if not symbol_candidates:
        return {}
    dibits = symbol_candidates[0].symbols
    offset = FS2_SYMBOL_COUNT
    cch0 = dibits[offset:offset + CCH_SYMBOLS]
    offset += CCH_SYMBOLS + TCH_SYMBOLS + CC_SYMBOLS
    cch1 = dibits[offset:offset + CCH_SYMBOLS]
    return {"CCH0": cch0, "CCH1": cch1}


def extract_current_full_dibits(result: ChainResult) -> np.ndarray:
    if result.candidate is None or result.phase is None:
        return np.array([], dtype=int)
    symbol_candidates = recover_voice_fs2_symbol_candidates(
        result.y,
        result.candidate,
        phase_search=np.array([result.phase]),
        sps_search=np.array([float(SPS)]),
        sample_windows=(0,),
        limit=1,
    )
    if not symbol_candidates:
        return np.array([], dtype=int)
    return np.asarray(symbol_candidates[0].symbols, dtype=int)


def _bits_to_matrix(bits: list[int], rows: int, cols: int) -> np.ndarray:
    return np.asarray(bits, dtype=float).reshape(rows, cols)


def plot_cch_stage_debug(result: ChainResult, output: Path) -> None:
    stages = extract_current_cch_stages(result)
    if not stages:
        return

    fig, axes = plt.subplots(len(stages), 5, figsize=(18, 4.4 * len(stages)), squeeze=False)

    for row, stage in enumerate(stages):
        raw_matrix = _bits_to_matrix(stage.raw_bits, 12, 6)
        descrambled_matrix = _bits_to_matrix(stage.descrambled_bits, 12, 6)
        deinterleaved_matrix = _bits_to_matrix(stage.deinterleaved_bits, 6, 12)
        decoded_matrix = _bits_to_matrix(stage.decoded_bits, 6, 8)

        ax0, ax1, ax2, ax3, ax4 = axes[row]
        for ax, matrix, title in (
            (ax0, raw_matrix, f"{stage.name} raw 12x6"),
            (ax1, descrambled_matrix, f"{stage.name} descramble"),
            (ax2, deinterleaved_matrix, f"{stage.name} deinterleave 6x12"),
            (ax3, decoded_matrix, f"{stage.name} decoded 6x8"),
        ):
            ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=0, vmax=1)
            ax.set_title(title)
            ax.set_xlabel("bit")
            ax.set_ylabel("row")

        block_text = "\n".join(
            f"blk{idx}: ok={ok} corr={corr}"
            for idx, (ok, corr) in enumerate(zip(stage.block_ok, stage.corrected_bits))
        )
        crc_text = (
            f"frame={stage.record_frame_number}\n"
            f"crc_ok={stage.record_crc_ok}\n"
            f"crc={stage.record_crc_value}\n"
            f"crc_calc={stage.record_crc_computed}"
        )
        ax4.axis("off")
        ax4.set_title(f"{stage.name} block status")
        ax4.text(0.02, 0.98, block_text, va="top", ha="left", family="monospace", fontsize=10)
        ax4.text(0.02, 0.40, crc_text, va="top", ha="left", family="monospace", fontsize=10)

    fig.suptitle("dPMR current chain CCH stage diagnostics", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dibits_to_raw_bits(dibits: np.ndarray) -> list[int]:
    raw_bits: list[int] = []
    for dibit in dibits:
        dibit = int(dibit) & 0x3
        raw_bits.append((dibit >> 1) & 1)
        raw_bits.append(dibit & 1)
    return raw_bits


def transform_dibits(dibits: np.ndarray, variant: str) -> np.ndarray:
    value = np.asarray(dibits, dtype=int) & 0x3
    if variant == "id":
        return value
    if variant == "xor1":
        return value ^ 0x1
    if variant == "xor2":
        return value ^ 0x2
    if variant == "xor3":
        return value ^ 0x3
    if variant == "swap":
        return ((value & 0x1) << 1) | ((value >> 1) & 0x1)
    if variant == "swap_xor1":
        value = ((value & 0x1) << 1) | ((value >> 1) & 0x1)
        return value ^ 0x1
    if variant == "swap_xor2":
        value = ((value & 0x1) << 1) | ((value >> 1) & 0x1)
        return value ^ 0x2
    if variant == "swap_xor3":
        value = ((value & 0x1) << 1) | ((value >> 1) & 0x1)
        return value ^ 0x3
    if variant == "reverse":
        return value[::-1]
    if variant == "reverse_swap":
        value = value[::-1]
        return ((value & 0x1) << 1) | ((value >> 1) & 0x1)
    raise KeyError(variant)


def score_cch_variant(raw_bits: list[int]) -> tuple[int, int, int | None, bool | None]:
    record = decode_cch(raw_bits)
    if record is None:
        return (0, 0, None, None)
    return (
        int(sum(1 for ok in record.hamming_blocks_ok if ok)),
        int(record.corrected_bits),
        int(record.frame_number),
        bool(record.crc_ok),
    )


def plot_cch_variant_scan(result: ChainResult, output: Path) -> None:
    cch_dibits = extract_current_cch_dibits(result)
    if not cch_dibits:
        return

    variants = [
        "id",
        "xor1",
        "xor2",
        "xor3",
        "swap",
        "swap_xor1",
        "swap_xor2",
        "swap_xor3",
        "reverse",
        "reverse_swap",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, name in zip(axes, ("CCH0", "CCH1")):
        ax.axis("off")
        dibits = cch_dibits[name]
        rows = [["variant", "ham_ok", "corr", "frame", "crc"]]
        for variant in variants:
            transformed = transform_dibits(dibits, variant)
            raw_bits = dibits_to_raw_bits(transformed)
            ham_ok, corrected, frame_number, crc_ok = score_cch_variant(raw_bits)
            rows.append([
                variant,
                str(ham_ok),
                str(corrected),
                "-" if frame_number is None else str(frame_number),
                "-" if crc_ok is None else str(crc_ok),
            ])
        table = ax.table(cellText=rows, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        ax.set_title(f"{name} dibit transform scan")

    fig.suptitle("dPMR current chain CCH dibit transform hypotheses", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cch_pair_hypotheses(result: ChainResult, output: Path) -> None:
    cch_dibits = extract_current_cch_dibits(result)
    if not cch_dibits:
        return
    variants = [
        "id",
        "xor1",
        "xor2",
        "xor3",
        "swap",
        "swap_xor1",
        "swap_xor2",
        "swap_xor3",
        "reverse",
        "reverse_swap",
    ]
    rows: list[list[str]] = [["CCH0", "CCH1", "frames", "ham_sum", "crc_pair"]]
    cch0_scores: dict[str, tuple[int, int, int | None, bool | None]] = {}
    cch1_scores: dict[str, tuple[int, int, int | None, bool | None]] = {}
    for variant in variants:
        cch0_scores[variant] = score_cch_variant(dibits_to_raw_bits(transform_dibits(cch_dibits["CCH0"], variant)))
        cch1_scores[variant] = score_cch_variant(dibits_to_raw_bits(transform_dibits(cch_dibits["CCH1"], variant)))

    for v0 in variants:
        for v1 in variants:
            h0, _, f0, c0 = cch0_scores[v0]
            h1, _, f1, c1 = cch1_scores[v1]
            if f0 is None or f1 is None:
                continue
            valid = {f0, f1} == {0, 1} or {f0, f1} == {2, 3}
            if not valid:
                continue
            rows.append([
                v0,
                v1,
                f"{f0}+{f1}",
                str(h0 + h1),
                str(bool(c0) and bool(c1)),
            ])

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(rows))))
    ax.axis("off")
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax.set_title("CCH0/CCH1 transform pairs yielding valid frame-number sets")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cch_window(full_dibits: np.ndarray, segment_name: str, slide: int) -> np.ndarray | None:
    if segment_name == "CCH0":
        start = FS2_SYMBOL_COUNT + slide
    elif segment_name == "CCH1":
        start = FS2_SYMBOL_COUNT + CCH_SYMBOLS + TCH_SYMBOLS + CC_SYMBOLS + slide
    else:
        raise KeyError(segment_name)
    stop = start + CCH_SYMBOLS
    if start < 0 or stop > len(full_dibits):
        return None
    return full_dibits[start:stop]


def _variant_names() -> list[str]:
    return [
        "id",
        "xor1",
        "xor2",
        "xor3",
        "swap",
        "swap_xor1",
        "swap_xor2",
        "swap_xor3",
        "reverse",
        "reverse_swap",
    ]


def plot_cch_boundary_slide_scan(result: ChainResult, output: Path) -> None:
    full_dibits = extract_current_full_dibits(result)
    if len(full_dibits) == 0:
        return

    variants = _variant_names()
    slides = range(-2, 3)
    segment_rows: dict[str, list[list[str]]] = {}
    scores: dict[tuple[str, int, str], tuple[int, int, int | None, bool | None]] = {}

    for segment_name in ("CCH0", "CCH1"):
        ranked: list[tuple[int, int, int, str, int, int | None, bool | None]] = []
        for slide in slides:
            base = cch_window(full_dibits, segment_name, slide)
            if base is None:
                continue
            for variant in variants:
                raw_bits = dibits_to_raw_bits(transform_dibits(base, variant))
                ham_ok, corrected, frame_number, crc_ok = score_cch_variant(raw_bits)
                scores[(segment_name, slide, variant)] = (ham_ok, corrected, frame_number, crc_ok)
                ranked.append((int(bool(crc_ok)), ham_ok, -corrected, variant, slide, frame_number, crc_ok))
        ranked.sort(reverse=True)
        rows = [["slide", "variant", "ham_ok", "corr", "frame", "crc"]]
        for crc_rank, ham_ok, neg_corr, variant, slide, frame_number, crc_ok in ranked[:18]:
            rows.append([
                str(slide),
                variant,
                str(ham_ok),
                str(-neg_corr),
                "-" if frame_number is None else str(frame_number),
                str(bool(crc_ok)),
            ])
        segment_rows[segment_name] = rows

    pair_ranked: list[tuple[int, int, int, int, str, int, str, str, int, str]] = []
    for s0 in slides:
        for v0 in variants:
            score0 = scores.get(("CCH0", s0, v0))
            if score0 is None:
                continue
            h0, corr0, f0, crc0 = score0
            if f0 is None:
                continue
            for s1 in slides:
                for v1 in variants:
                    score1 = scores.get(("CCH1", s1, v1))
                    if score1 is None:
                        continue
                    h1, corr1, f1, crc1 = score1
                    if f1 is None:
                        continue
                    valid_pair = {f0, f1} == {0, 1} or {f0, f1} == {2, 3}
                    crc_count = int(bool(crc0)) + int(bool(crc1))
                    pair_ranked.append((
                        int(valid_pair),
                        crc_count,
                        h0 + h1,
                        -(corr0 + corr1),
                        v0,
                        s0,
                        v1,
                        f"{f0}+{f1}",
                        s1,
                        str(bool(crc0) and bool(crc1)),
                    ))
    pair_ranked.sort(reverse=True)
    pair_rows = [["CCH0", "s0", "CCH1", "s1", "frames", "ham", "crc_pair"]]
    for valid_pair, crc_count, ham_sum, neg_corr, v0, s0, v1, frames, s1, crc_pair in pair_ranked[:24]:
        if valid_pair == 0 and crc_count == 0:
            continue
        pair_rows.append([v0, str(s0), v1, str(s1), frames, str(ham_sum), crc_pair])
    if len(pair_rows) == 1:
        pair_rows.append(["-", "-", "-", "-", "no valid/crc candidates", "-", "-"])

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, 0.34 * max(len(pair_rows), 18))))
    for ax, title, rows in (
        (axes[0], "CCH0 slide + transform top candidates", segment_rows["CCH0"]),
        (axes[1], "CCH1 slide + transform top candidates", segment_rows["CCH1"]),
        (axes[2], "CCH0/CCH1 paired candidates", pair_rows),
    ):
        ax.axis("off")
        table = ax.table(cellText=rows, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.25)
        ax.set_title(title)
    fig.suptitle("dPMR CCH boundary slide scan around current FS2 lock", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def best_cch_pair_for_dibits(full_dibits: np.ndarray) -> tuple[int, int, int, str]:
    variants = _variant_names()
    slides = range(-2, 3)
    cch0_scores: list[tuple[int, int, int, int, str, int | None, bool | None]] = []
    cch1_scores: list[tuple[int, int, int, int, str, int | None, bool | None]] = []
    for segment_name, rows in (("CCH0", cch0_scores), ("CCH1", cch1_scores)):
        for slide in slides:
            base = cch_window(full_dibits, segment_name, slide)
            if base is None:
                continue
            for variant in variants:
                raw_bits = dibits_to_raw_bits(transform_dibits(base, variant))
                ham_ok, corrected, frame_number, crc_ok = score_cch_variant(raw_bits)
                rows.append((int(bool(crc_ok)), ham_ok, -corrected, slide, variant, frame_number, crc_ok))
    best: tuple[int, int, int, str] | None = None
    for crc0, h0, neg_corr0, s0, v0, f0, _ in cch0_scores:
        for crc1, h1, neg_corr1, s1, v1, f1, _ in cch1_scores:
            if f0 is None or f1 is None:
                continue
            valid_pair = int({f0, f1} == {0, 1} or {f0, f1} == {2, 3})
            crc_count = crc0 + crc1
            ham_sum = h0 + h1
            corrected = -(neg_corr0 + neg_corr1)
            detail = f"{f0}+{f1} c0={v0}@{s0} c1={v1}@{s1}"
            row = (crc_count, valid_pair, ham_sum, corrected, detail)
            if best is None or row[:3] > best[:3] or (row[:3] == best[:3] and row[3] < best[3]):
                best = row
    if best is None:
        return (0, 0, 0, "no candidates")
    crc_count, valid_pair, ham_sum, corrected, detail = best
    return crc_count, valid_pair, ham_sum, f"corr={corrected} {detail}"


def plot_global_cch_crc_survey(iq: np.ndarray, fs: float, output: Path, limit: int = 40) -> None:
    y, _ = fm_frontend(iq, fs=fs, cutoff=3500.0, integrate_symbols=False)
    candidates = sorted(find_fs2_sync(y, threshold=0.68), key=lambda item: item.ncc, reverse=True)[:limit]
    rows = [["fs_start", "ncc", "inv", "crc", "pair", "ham", "detail"]]
    summaries: list[tuple[int, int, int, float, DPMRSyncCandidate, str]] = []
    for candidate in candidates:
        symbol_candidates = recover_voice_fs2_symbol_candidates(y, candidate, limit=1)
        if not symbol_candidates:
            continue
        full_dibits = np.asarray(symbol_candidates[0].symbols, dtype=int)
        crc_count, valid_pair, ham_sum, detail = best_cch_pair_for_dibits(full_dibits)
        summaries.append((crc_count, valid_pair, ham_sum, candidate.ncc, candidate, detail))
    summaries.sort(reverse=True, key=lambda item: (item[0], item[1], item[2], item[3]))
    for crc_count, valid_pair, ham_sum, ncc, candidate, detail in summaries[:24]:
        rows.append([
            str(candidate.fs_start),
            f"{candidate.ncc:.3f}",
            str(candidate.polarity_inverted),
            str(crc_count),
            str(valid_pair),
            str(ham_sum),
            detail,
        ])

    fig, ax = plt.subplots(figsize=(18, max(6, 0.34 * len(rows))))
    ax.axis("off")
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    ax.set_title(f"Global dPMR CCH CRC survey across top {len(candidates)} FS2 candidates")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_official_cch_survey(iq: np.ndarray, fs: float, output: Path, limit: int = 120) -> None:
    y, _ = fm_frontend(iq, fs=fs, cutoff=3500.0, integrate_symbols=False)
    candidates = sorted(find_fs2_sync(y, threshold=0.68), key=lambda item: item.ncc, reverse=True)[:limit]
    rows = [["fs_start", "ncc", "inv", "crc", "ham_ok", "blocks", "frames", "crc_parts"]]
    summaries: list[tuple[int, int, int, float, DPMRSyncCandidate, str, str]] = []
    for candidate in candidates:
        symbol_candidates = recover_voice_fs2_symbol_candidates(y, candidate, limit=1)
        if not symbol_candidates:
            continue
        cch0_bits, _, cch1_bits = split_voice_fs2(symbol_candidates[0].symbols)
        cch0 = decode_cch(cch0_bits)
        cch1 = decode_cch(cch1_bits)
        if cch0 is None or cch1 is None:
            continue
        crc_count = int(cch0.crc_ok) + int(cch1.crc_ok)
        hamming_ok_count = int(cch0.hamming_ok) + int(cch1.hamming_ok)
        block_count = sum(cch0.hamming_blocks_ok) + sum(cch1.hamming_blocks_ok)
        frames = f"{cch0.frame_number}+{cch1.frame_number}"
        crc_parts = f"{cch0.crc_ok}/{cch1.crc_ok}"
        summaries.append((
            crc_count,
            hamming_ok_count,
            block_count,
            candidate.ncc,
            candidate,
            frames,
            crc_parts,
        ))
    summaries.sort(reverse=True, key=lambda item: (item[0], item[1], item[2], item[3]))
    for crc_count, hamming_ok_count, block_count, ncc, candidate, frames, crc_parts in summaries[:24]:
        rows.append([
            str(candidate.fs_start),
            f"{candidate.ncc:.3f}",
            str(candidate.polarity_inverted),
            str(crc_count),
            str(hamming_ok_count),
            str(block_count),
            frames,
            crc_parts,
        ])

    fig, ax = plt.subplots(figsize=(14, max(6, 0.34 * len(rows))))
    ax.axis("off")
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax.set_title(f"Official-path dPMR CCH survey across top {len(candidates)} FS2 candidates")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_force_decode_records(iq: np.ndarray, fs: float, limit: int) -> list[dict]:
    y, _ = fm_frontend(iq, fs=fs, cutoff=3500.0, integrate_symbols=False)
    candidates = sorted(find_fs2_sync(y, threshold=0.68), key=lambda item: item.ncc, reverse=True)[:limit]
    records: list[dict] = []
    for candidate in candidates:
        symbol_candidates = recover_voice_fs2_symbol_candidates(y, candidate, limit=1)
        if not symbol_candidates:
            continue
        cch0_bits, _, cch1_bits = split_voice_fs2(symbol_candidates[0].symbols)
        for part, bits in (("CCH0", cch0_bits), ("CCH1", cch1_bits)):
            record = decode_cch(bits)
            if record is None:
                continue
            records.append({
                "fs_start": candidate.fs_start,
                "part": part,
                "frame": record.frame_number,
                "id_half": record.id_half,
                "mode": record.communication_mode,
                "version": record.version,
                "format": record.comms_format,
                "emergency": record.emergency_priority,
                "reserved": record.reserved,
                "slow": record.slow_data,
                "crc_ok": record.crc_ok,
                "hamming_ok": record.hamming_ok,
                "blocks": sum(record.hamming_blocks_ok),
                "corrected": record.corrected_bits,
            })
    return records


def _force_decode_summary_rows(records: list[dict]) -> list[list[str]]:
    rows = [["set", "n", "frame_counts", "unique_fields", "top_field_repeat", "top_id_repeat"]]
    categories = [
        ("CRC_OK", lambda item: item["crc_ok"]),
        ("HAMMING_OK_NOCRC", lambda item: item["hamming_ok"]),
        ("BLOCKS_GE5_NOCRC", lambda item: item["blocks"] >= 5),
    ]
    for name, predicate in categories:
        subset = [item for item in records if predicate(item)]
        frame_counts = Counter(item["frame"] for item in subset)
        field_counts = Counter(
            (
                item["frame"],
                item["id_half"],
                item["mode"],
                item["version"],
                item["format"],
                item["emergency"],
                item["reserved"],
            )
            for item in subset
        )
        id_counts = Counter((item["frame"], item["id_half"]) for item in subset)
        top_field_repeat = field_counts.most_common(1)[0][1] if field_counts else 0
        top_id_repeat = id_counts.most_common(1)[0][1] if id_counts else 0
        rows.append([
            name,
            str(len(subset)),
            " ".join(f"{frame}:{frame_counts.get(frame, 0)}" for frame in range(4)),
            f"{len(field_counts)}/{len(subset)}" if subset else "0/0",
            str(top_field_repeat),
            str(top_id_repeat),
        ])
    return rows


def _forced_id_rows(records: list[dict]) -> list[list[str]]:
    rows = [["kind", "air_id", "count"]]
    usable = [item for item in records if item["hamming_ok"]]
    ids: Counter[tuple[str, int, str]] = Counter()
    for i, first in enumerate(usable):
        for second in usable[i + 1:i + 16]:
            if abs(second["fs_start"] - first["fs_start"]) > 120000:
                break
            frames = {first["frame"], second["frame"]}
            if frames == {0, 1}:
                hi = first if first["frame"] == 0 else second
                lo = second if first["frame"] == 0 else first
                value = (hi["id_half"] << 12) | lo["id_half"]
                ids[("dst", value, air_interface_id_to_str(value))] += 1
            if frames == {2, 3}:
                hi = first if first["frame"] == 2 else second
                lo = second if first["frame"] == 2 else first
                value = (hi["id_half"] << 12) | lo["id_half"]
                ids[("src", value, air_interface_id_to_str(value))] += 1
    for (kind, _value, text), count in ids.most_common(12):
        rows.append([kind, text, str(count)])
    if len(rows) == 1:
        rows.append(["-", "no hamming-only pairs", "-"])
    return rows


def plot_force_decode_consistency(iq: np.ndarray, fs: float, output: Path, limit: int = 160) -> None:
    records = collect_force_decode_records(iq, fs, limit)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, title, rows in (
        (axes[0], "Forced CCH decode consistency without requiring CRC", _force_decode_summary_rows(records)),
        (axes[1], "Forced hamming-only ID assembly repeat check", _forced_id_rows(records)),
    ):
        ax.axis("off")
        table = ax.table(cellText=rows, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)
        ax.set_title(title)
    fig.suptitle(f"dPMR forced decode consistency across top {limit} FS2 candidates", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cch0_local_search(
    y: np.ndarray,
    candidate: DPMRSyncCandidate,
    base_phase: float,
    *,
    base_sps: float = float(SPS),
    phase_offsets: np.ndarray | None = None,
    sps_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    if phase_offsets is None:
        phase_offsets = np.linspace(-3.0, 3.0, 49)
    if sps_offsets is None:
        sps_offsets = np.linspace(-0.25, 0.25, 41)

    cch0 = segment_slice("CCH0")
    p90_grid = np.full((len(sps_offsets), len(phase_offsets)), np.nan)
    amb_grid = np.full_like(p90_grid, np.nan, dtype=float)
    best: tuple[float, float, float, float] | None = None

    for i, sps_delta in enumerate(sps_offsets):
        sps = base_sps + float(sps_delta)
        for j, phase_delta in enumerate(phase_offsets):
            phase = base_phase + float(phase_delta)
            positions = candidate.fs_start + phase + np.arange(VOICE_FS2_TOTAL_SYMBOLS) * sps
            if positions[0] < 0 or positions[-1] >= len(y) - 1:
                continue
            raw = _interp(y, positions)
            calibrated = normalize_four_levels(raw)
            segment = calibrated[cch0]
            p90, ambiguous_ratio, _ = cloud_metrics(segment)
            p90_grid[i, j] = p90
            amb_grid[i, j] = ambiguous_ratio
            score = p90 + 0.5 * ambiguous_ratio
            if best is None or score < best[0]:
                best = (score, phase, sps, ambiguous_ratio)

    if best is None:
        best = (float("nan"), float("nan"), float("nan"), float("nan"))
    return phase_offsets, sps_offsets, p90_grid, best


def recover_drift_symbols(
    y: np.ndarray,
    candidate: DPMRSyncCandidate,
    *,
    base_phase: float = 0.0,
    base_sps: float = float(SPS),
    drift_total: float = 0.0,
) -> np.ndarray:
    symbol_idx = np.arange(VOICE_FS2_TOTAL_SYMBOLS, dtype=float)
    drift = drift_total * (symbol_idx / max(1.0, VOICE_FS2_TOTAL_SYMBOLS - 1.0))
    positions = candidate.fs_start + base_phase + symbol_idx * base_sps + drift
    return _interp(y, positions)


def cch0_drift_search(
    y: np.ndarray,
    candidate: DPMRSyncCandidate,
    base_phase: float,
    *,
    base_sps: float = float(SPS),
    phase_offsets: np.ndarray | None = None,
    drift_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    if phase_offsets is None:
        phase_offsets = np.linspace(-2.0, 2.0, 49)
    if drift_offsets is None:
        drift_offsets = np.linspace(-4.0, 4.0, 49)

    cch0 = segment_slice("CCH0")
    p90_grid = np.full((len(drift_offsets), len(phase_offsets)), np.nan)
    best: tuple[float, float, float, float] | None = None

    for i, drift_offset in enumerate(drift_offsets):
        for j, phase_offset in enumerate(phase_offsets):
            phase = base_phase + float(phase_offset)
            raw = recover_drift_symbols(
                y,
                candidate,
                base_phase=phase,
                base_sps=base_sps,
                drift_total=float(drift_offset),
            )
            calibrated = normalize_four_levels(raw)
            segment = calibrated[cch0]
            p90, ambiguous_ratio, _ = cloud_metrics(segment)
            p90_grid[i, j] = p90
            score = p90 + 0.5 * ambiguous_ratio
            if best is None or score < best[0]:
                best = (score, phase, float(drift_offset), ambiguous_ratio)

    if best is None:
        best = (float("nan"), float("nan"), float("nan"), float("nan"))
    return phase_offsets, drift_offsets, p90_grid, best


def plot_cch0_local_search(results: list[ChainResult], output: Path) -> None:
    fixed_results = [result for result in results if result.candidate is not None and result.phase is not None]
    if not fixed_results:
        return

    fig, axes = plt.subplots(len(fixed_results), 2, figsize=(14, 4.2 * len(fixed_results)), squeeze=False)

    for row, result in enumerate(fixed_results):
        ax_heat = axes[row, 0]
        ax_cmp = axes[row, 1]
        phase_offsets, sps_offsets, p90_grid, best = cch0_local_search(
            result.y,
            result.candidate,
            result.phase,
        )
        im = ax_heat.imshow(
            p90_grid,
            aspect="auto",
            origin="lower",
            extent=[phase_offsets[0], phase_offsets[-1], sps_offsets[0], sps_offsets[-1]],
            cmap="viridis",
        )
        ax_heat.set_title(f"{result.name}: CCH0 local p90 search")
        ax_heat.set_xlabel("phase offset from current")
        ax_heat.set_ylabel("sps offset from 20.0")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        best_score, best_phase, best_sps, best_amb = best
        ax_heat.scatter([best_phase - result.phase], [best_sps - SPS], marker="x", s=70, color="red")

        base_positions = result.candidate.fs_start + result.phase + np.arange(VOICE_FS2_TOTAL_SYMBOLS) * float(SPS)
        best_positions = result.candidate.fs_start + best_phase + np.arange(VOICE_FS2_TOTAL_SYMBOLS) * best_sps
        base_segment = normalize_four_levels(_interp(result.y, base_positions))[segment_slice("CCH0")]
        best_segment = normalize_four_levels(_interp(result.y, best_positions))[segment_slice("CCH0")]
        base_p90, base_amb, _ = cloud_metrics(base_segment)
        best_p90, _, _ = cloud_metrics(best_segment)

        x = np.arange(len(base_segment))
        ax_cmp.scatter(x, base_segment, s=14, alpha=0.65, color="#7f7f7f", label=f"current p90={base_p90:.3f}")
        ax_cmp.scatter(x, best_segment, s=14, alpha=0.65, color="#1f77b4", label=f"best p90={best_p90:.3f}")
        for level in LEVELS:
            ax_cmp.axhline(level, color="#999999", lw=0.7, ls="--", alpha=0.5)
        ax_cmp.set_title(
            f"CCH0 compare: best phase={best_phase:.2f}, sps={best_sps:.3f}, amb={best_amb:.1%}"
        )
        ax_cmp.set_xlabel("CCH0 symbol index")
        ax_cmp.set_ylabel("normalized symbol value")
        ax_cmp.set_ylim(-4.8, 4.8)
        ax_cmp.grid(True, alpha=0.25)
        ax_cmp.legend(loc="upper right")

    fig.suptitle("dPMR CCH0 local timing search around current FS2 lock", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cch0_drift_search(results: list[ChainResult], output: Path) -> None:
    fixed_results = [result for result in results if result.candidate is not None and result.phase is not None]
    if not fixed_results:
        return

    fig, axes = plt.subplots(len(fixed_results), 2, figsize=(14, 4.2 * len(fixed_results)), squeeze=False)

    for row, result in enumerate(fixed_results):
        ax_heat = axes[row, 0]
        ax_cmp = axes[row, 1]
        phase_offsets, drift_offsets, p90_grid, best = cch0_drift_search(
            result.y,
            result.candidate,
            result.phase,
        )
        im = ax_heat.imshow(
            p90_grid,
            aspect="auto",
            origin="lower",
            extent=[phase_offsets[0], phase_offsets[-1], drift_offsets[0], drift_offsets[-1]],
            cmap="magma",
        )
        ax_heat.set_title(f"{result.name}: CCH0 drift search")
        ax_heat.set_xlabel("phase offset from current")
        ax_heat.set_ylabel("total frame drift (samples)")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        best_score, best_phase, best_drift, best_amb = best
        ax_heat.scatter([best_phase - result.phase], [best_drift], marker="x", s=70, color="cyan")

        base_raw = recover_drift_symbols(
            result.y,
            result.candidate,
            base_phase=result.phase,
            base_sps=float(SPS),
            drift_total=0.0,
        )
        best_raw = recover_drift_symbols(
            result.y,
            result.candidate,
            base_phase=best_phase,
            base_sps=float(SPS),
            drift_total=best_drift,
        )
        base_segment = normalize_four_levels(base_raw)[segment_slice("CCH0")]
        best_segment = normalize_four_levels(best_raw)[segment_slice("CCH0")]
        base_p90, _, _ = cloud_metrics(base_segment)
        best_p90, _, _ = cloud_metrics(best_segment)

        x = np.arange(len(base_segment))
        ax_cmp.scatter(x, base_segment, s=14, alpha=0.6, color="#7f7f7f", label=f"current p90={base_p90:.3f}")
        ax_cmp.scatter(x, best_segment, s=14, alpha=0.65, color="#17becf", label=f"drift p90={best_p90:.3f}")
        for level in LEVELS:
            ax_cmp.axhline(level, color="#999999", lw=0.7, ls="--", alpha=0.5)
        ax_cmp.set_title(
            f"CCH0 drift compare: best phase={best_phase:.2f}, drift={best_drift:.2f}, amb={best_amb:.1%}"
        )
        ax_cmp.set_xlabel("CCH0 symbol index")
        ax_cmp.set_ylabel("normalized symbol value")
        ax_cmp.set_ylim(-4.8, 4.8)
        ax_cmp.grid(True, alpha=0.25)
        ax_cmp.legend(loc="upper right")

    fig.suptitle("dPMR CCH0 drift search around current FS2 lock", fontsize=14)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_segment_summary(results: list[ChainResult]) -> None:
    for result in results:
        print(result.name)
        print(
            f"  sync carrier={result.carrier_hz:.1f}Hz "
            f"phase={'Gardner' if result.phase is None else f'{result.phase:.2f}'} "
            f"global p90={result.p90_distance:.3f} amb={result.ambiguous_ratio:.1%} "
            f"spread={result.cluster_spread:.3f}"
        )
        for name, start, stop in SEGMENTS:
            segment = result.calibrated_symbols[start:stop]
            if not len(segment):
                continue
            p90, ambiguous_ratio, spread = cloud_metrics(segment)
            print(
                f"  {name:>4}: p90={p90:.3f} amb={ambiguous_ratio:.1%} spread={spread:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("iq_file", nargs="?", default="data/dpmr_1_48000.rawiq")
    parser.add_argument("--fs", type=float, default=FS_DEC)
    parser.add_argument("--output", default="output/dpmr_demod_quality.png")
    parser.add_argument("--timing-output", default="output/dpmr_timing_drift.png")
    parser.add_argument("--cch0-output", default="output/dpmr_cch0_local_search.png")
    parser.add_argument("--drift-output", default="output/dpmr_cch0_drift_search.png")
    parser.add_argument("--cch-stage-output", default="output/dpmr_cch_stage_debug.png")
    parser.add_argument("--variant-output", default="output/dpmr_cch_variant_scan.png")
    parser.add_argument("--pair-output", default="output/dpmr_cch_pair_hypotheses.png")
    parser.add_argument("--slide-output", default="output/dpmr_cch_boundary_slide_scan.png")
    parser.add_argument("--survey-output", default="output/dpmr_global_cch_crc_survey.png")
    parser.add_argument("--official-survey-output", default="output/dpmr_official_cch_survey.png")
    parser.add_argument("--force-output", default="output/dpmr_force_decode_consistency.png")
    parser.add_argument("--survey-limit", type=int, default=40)
    parser.add_argument("--official-survey-limit", type=int, default=120)
    parser.add_argument("--force-limit", type=int, default=160)
    args = parser.parse_args()

    iq = read_rawiq(args.iq_file)
    results = [
        analyze_chain(
            iq,
            name="current dPMR: cutoff 3.5k, fixed phase",
            cutoff=3500.0,
            integrate_symbols=False,
            timing="fixed",
            fs=args.fs,
        ),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_results(results, output)
    timing_output = Path(args.timing_output)
    timing_output.parent.mkdir(parents=True, exist_ok=True)
    plot_timing_drift(results, timing_output)
    cch0_output = Path(args.cch0_output)
    cch0_output.parent.mkdir(parents=True, exist_ok=True)
    plot_cch0_local_search(results, cch0_output)
    drift_output = Path(args.drift_output)
    drift_output.parent.mkdir(parents=True, exist_ok=True)
    plot_cch0_drift_search(results, drift_output)
    cch_stage_output = Path(args.cch_stage_output)
    cch_stage_output.parent.mkdir(parents=True, exist_ok=True)
    plot_cch_stage_debug(results[0], cch_stage_output)
    variant_output = Path(args.variant_output)
    variant_output.parent.mkdir(parents=True, exist_ok=True)
    plot_cch_variant_scan(results[0], variant_output)
    pair_output = Path(args.pair_output)
    pair_output.parent.mkdir(parents=True, exist_ok=True)
    plot_cch_pair_hypotheses(results[0], pair_output)
    slide_output = Path(args.slide_output)
    slide_output.parent.mkdir(parents=True, exist_ok=True)
    plot_cch_boundary_slide_scan(results[0], slide_output)
    survey_output = Path(args.survey_output)
    survey_output.parent.mkdir(parents=True, exist_ok=True)
    plot_global_cch_crc_survey(iq, args.fs, survey_output, limit=args.survey_limit)
    official_survey_output = Path(args.official_survey_output)
    official_survey_output.parent.mkdir(parents=True, exist_ok=True)
    plot_official_cch_survey(iq, args.fs, official_survey_output, limit=args.official_survey_limit)
    force_output = Path(args.force_output)
    force_output.parent.mkdir(parents=True, exist_ok=True)
    plot_force_decode_consistency(iq, args.fs, force_output, limit=args.force_limit)

    print(f"saved {output}")
    print(f"saved {timing_output}")
    print(f"saved {cch0_output}")
    print(f"saved {drift_output}")
    print(f"saved {cch_stage_output}")
    print(f"saved {variant_output}")
    print(f"saved {pair_output}")
    print(f"saved {slide_output}")
    print(f"saved {survey_output}")
    print(f"saved {official_survey_output}")
    print(f"saved {force_output}")
    for result in results:
        sync = "none"
        if result.candidate is not None:
            sync = (
                f"fs_start={result.candidate.fs_start} "
                f"inv={result.candidate.polarity_inverted} "
                f"ncc={result.candidate.ncc:.3f}"
            )
        print(f"{result.name}: {sync}")
    print_segment_summary(results)


if __name__ == "__main__":
    main()
