from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

from core.dsp import _interp, frontend
from dpmr.constants import (
    CC_SYMBOLS,
    CCH_SYMBOLS,
    DIBIT_TO_BITS,
    DIBIT_TO_LEVEL,
    DPMR_FRONTEND_CUTOFF,
    DPMR_FRAME_SYMBOLS,
    FS1_SYMBOLS,
    FS2_SYMBOLS,
    FS3_SYMBOLS,
    FS4_SYMBOLS,
    INV_FS1_SYMBOLS,
    INV_FS2_SYMBOLS,
    INV_FS3_SYMBOLS,
    INV_FS4_SYMBOLS,
    SPS,
    TCH_SYMBOLS,
    VOICE_FS2_TOTAL_SYMBOLS,
)

DIBIT_LEVELS = np.array([DIBIT_TO_LEVEL[idx] for idx in range(4)], dtype=float)


@dataclass(frozen=True)
class DPMRSyncCandidate:
    fs_start: int
    polarity_inverted: bool
    ncc: float
    sync_type: str = "FS2"


@dataclass(frozen=True)
class DPMRSymbolCandidate:
    symbols: np.ndarray
    resid: float
    sps: float
    phase: float
    sample_window: int
    decision_error_p90: float
    ambiguous_symbols: int


def frontend_dpmr(iq_dec: np.ndarray, fs: float = 48_000) -> np.ndarray:
    return frontend(iq_dec, fo=0.0, fs=fs, cutoff=DPMR_FRONTEND_CUTOFF)


def _ncc(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    wave = np.repeat(ref, SPS)
    corr = signal.correlate(y, wave, mode="same")
    energy = np.convolve(y ** 2, np.ones(len(wave)), mode="same")
    energy = np.where(energy <= 0, 1e-9, energy)
    return corr / np.sqrt(energy * np.sum(wave ** 2))


def _dibits_to_levels(dibits: np.ndarray) -> np.ndarray:
    return DIBIT_LEVELS[dibits.astype(int)]


def _zero_mean_ncc(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref_levels = _dibits_to_levels(ref)
    wave = np.repeat(ref_levels - np.mean(ref_levels), SPS)
    window = len(wave)
    kernel = np.ones(window)
    local_mean = signal.convolve(y, kernel, mode="same") / window
    centered = y - local_mean
    corr = signal.correlate(centered, wave, mode="same")
    energy = signal.convolve(centered ** 2, kernel, mode="same")
    energy = np.where(energy <= 0, 1e-9, energy)
    return corr / np.sqrt(energy * np.sum(wave ** 2))


def _sync_error(y: np.ndarray, fs_start: int, ref: np.ndarray) -> tuple[int, float]:
    best = (len(ref), float("inf"))
    ref_levels = _dibits_to_levels(ref)
    for phase in np.linspace(-12, 12, 25):
        pos = fs_start + phase + np.arange(len(ref)) * SPS
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue
        seg = _sample_symbols(y, pos, 0)
        a, b = np.linalg.lstsq(
            np.vstack([seg, np.ones(len(seg))]).T,
            ref_levels,
            rcond=None,
        )[0]
        calibrated = a * seg + b
        nearest = np.argmin(
            np.abs(calibrated[:, None] - DIBIT_LEVELS[None, :]),
            axis=1,
        ).astype(int)
        item = (
            int(np.sum(nearest != ref.astype(int))),
            float(np.mean((calibrated - ref_levels) ** 2)),
        )
        if item < best:
            best = item
    return best


def find_dpmr_sync(
    y: np.ndarray,
    threshold: float = 0.82,
    max_symbol_errors: int = 0,
) -> list[DPMRSyncCandidate]:
    refs = (
        ("FS1", False, FS1_SYMBOLS),
        ("FS2", False, FS2_SYMBOLS),
        ("FS3", False, FS3_SYMBOLS),
        ("FS4", False, FS4_SYMBOLS),
        ("FS1", True, INV_FS1_SYMBOLS),
        ("FS2", True, INV_FS2_SYMBOLS),
        ("FS3", True, INV_FS3_SYMBOLS),
        ("FS4", True, INV_FS4_SYMBOLS),
    )
    candidates: list[tuple[int, str, bool, float, int, float]] = []
    for sync_type, inverted, ref in refs:
        ncc = _zero_mean_ncc(y, ref)
        peaks, props = signal.find_peaks(ncc, height=threshold, distance=1200)
        for peak, height in zip(peaks, props["peak_heights"]):
            fs_start = int(round(peak - (len(ref) * SPS) / 2))
            if fs_start < 0:
                continue
            errors, resid = _sync_error(y, fs_start, ref)
            if errors <= max_symbol_errors:
                candidates.append((fs_start, sync_type, inverted, float(height), errors, resid))
    candidates.sort(key=lambda item: (item[0], item[4], item[5], -item[3]))
    deduped: list[tuple[int, str, bool, float, int, float]] = []
    for cand in candidates:
        if deduped and abs(cand[0] - deduped[-1][0]) < SPS * 3:
            if (cand[4], cand[5], -cand[3]) < (
                deduped[-1][4],
                deduped[-1][5],
                -deduped[-1][3],
            ):
                deduped[-1] = cand
            continue
        deduped.append(cand)
    return [
        DPMRSyncCandidate(fs_start, inverted, ncc, sync_type)
        for fs_start, sync_type, inverted, ncc, _, _ in deduped
    ]


def find_fs2_sync(y: np.ndarray, threshold: float = 0.82) -> list[DPMRSyncCandidate]:
    return [
        candidate for candidate in find_dpmr_sync(y, threshold=threshold)
        if candidate.sync_type == "FS2"
    ]


def find_fs1_sync(y: np.ndarray, threshold: float = 0.82) -> list[DPMRSyncCandidate]:
    return [
        candidate for candidate in find_dpmr_sync(y, threshold=threshold)
        if candidate.sync_type == "FS1"
    ]


def recover_voice_fs2_symbols(
    y: np.ndarray,
    candidate: DPMRSyncCandidate,
    phase_search: np.ndarray | None = None,
) -> np.ndarray | None:
    recovered = recover_voice_fs2_symbol_candidates(y, candidate, phase_search=phase_search, limit=1)
    return recovered[0].symbols if recovered else None


def _sample_symbols(y: np.ndarray, positions: np.ndarray, half_window: int) -> np.ndarray:
    if half_window <= 0:
        return _interp(y, positions)
    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    samples = np.vstack([_interp(y, positions + offset) for offset in offsets])
    return np.mean(samples, axis=0)


def _sync_ref_for_candidate(candidate: DPMRSyncCandidate) -> np.ndarray:
    refs = {
        ("FS1", False): FS1_SYMBOLS,
        ("FS2", False): FS2_SYMBOLS,
        ("FS3", False): FS3_SYMBOLS,
        ("FS4", False): FS4_SYMBOLS,
        ("FS1", True): INV_FS1_SYMBOLS,
        ("FS2", True): INV_FS2_SYMBOLS,
        ("FS3", True): INV_FS3_SYMBOLS,
        ("FS4", True): INV_FS4_SYMBOLS,
    }
    return refs[(candidate.sync_type, candidate.polarity_inverted)]


def recover_frame_symbol_candidates(
    y: np.ndarray,
    candidate: DPMRSyncCandidate,
    total_symbols: int = DPMR_FRAME_SYMBOLS,
    phase_search: np.ndarray | None = None,
    sps_search: np.ndarray | None = None,
    sample_windows: tuple[int, ...] = (0, 3),
    limit: int = 8,
) -> list[DPMRSymbolCandidate]:
    if phase_search is None:
        phase_search = np.linspace(-10, 10, 41)
    if sps_search is None:
        sps_search = np.linspace(19.0, 21.0, 21)
    ref_dibits = _sync_ref_for_candidate(candidate)
    ref = _dibits_to_levels(ref_dibits)
    recovered: list[tuple[float, np.ndarray, float, float, int]] = []
    for sps in sps_search:
        for phase in phase_search:
            pos = candidate.fs_start + phase + np.arange(total_symbols) * float(sps)
            if pos[0] < 0 or pos[-1] >= len(y) - 1:
                continue
            for sample_window in sample_windows:
                if pos[0] - sample_window < 0 or pos[-1] + sample_window >= len(y) - 1:
                    continue
                seg = _sample_symbols(y, pos, sample_window)
                fs_seg = seg[:len(ref)]
                a, b = np.linalg.lstsq(
                    np.vstack([fs_seg, np.ones(len(fs_seg))]).T,
                    ref,
                    rcond=None,
                )[0]
                calibrated = a * seg + b
                nearest = np.argmin(
                    np.abs(calibrated[:, None] - DIBIT_LEVELS[None, :]),
                    axis=1,
                ).astype(int)
                decision_error = np.abs(calibrated - DIBIT_LEVELS[nearest])
                resid = float(np.mean((calibrated[:len(ref)] - ref) ** 2))
                resid += 0.03 * float(np.mean((calibrated - nearest) ** 2))
                if candidate.polarity_inverted:
                    nearest = nearest ^ 2
                recovered.append((
                    resid,
                    nearest,
                    float(sps),
                    float(phase),
                    sample_window,
                    float(np.percentile(decision_error, 90)),
                    int(np.sum(decision_error > 0.35)),
                ))
    recovered.sort(key=lambda item: item[0])
    return [
        DPMRSymbolCandidate(
            symbols=symbols,
            resid=resid,
            sps=sps,
            phase=phase,
            sample_window=sample_window,
            decision_error_p90=decision_error_p90,
            ambiguous_symbols=ambiguous_symbols,
        )
        for (
            resid,
            symbols,
            sps,
            phase,
            sample_window,
            decision_error_p90,
            ambiguous_symbols,
        ) in recovered[:limit]
    ]


def recover_voice_fs2_symbol_candidates(
    y: np.ndarray,
    candidate: DPMRSyncCandidate,
    phase_search: np.ndarray | None = None,
    sps_search: np.ndarray | None = None,
    sample_windows: tuple[int, ...] = (0, 3),
    limit: int = 8,
) -> list[DPMRSymbolCandidate]:
    return recover_frame_symbol_candidates(
        y,
        candidate,
        total_symbols=VOICE_FS2_TOTAL_SYMBOLS,
        phase_search=phase_search,
        sps_search=sps_search,
        sample_windows=sample_windows,
        limit=limit,
    )


def symbols_to_bits(symbols: np.ndarray) -> list[int]:
    out: list[int] = []
    for sym in symbols:
        out.extend(DIBIT_TO_BITS[int(sym) & 3])
    return out


def split_voice_fs2(symbols: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    offset = len(FS2_SYMBOLS)
    cch0 = symbols_to_bits(symbols[offset:offset + CCH_SYMBOLS])
    offset += CCH_SYMBOLS + TCH_SYMBOLS
    cc = symbols_to_bits(symbols[offset:offset + CC_SYMBOLS])
    offset += CC_SYMBOLS
    cch1 = symbols_to_bits(symbols[offset:offset + CCH_SYMBOLS])
    return cch0, cc, cch1
