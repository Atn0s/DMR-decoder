from __future__ import annotations

from math import gcd

import numpy as np
import scipy.signal as signal

from common.io import detect_sample_rate, read_rawiq
from dmr.constants import (
    DOWN_FACTOR,
    Fs_dec,
    Fs_wide,
    SPS,
    SYNC_TEMPLATES,
    UP_FACTOR,
)
from dmr.config import DEFAULT_DMR_CONFIG, DMRConfig
from dmr.decoder import LateEntryCollector, decode_burst
from dmr.dsp import _interp, adaptive_slice_bits, find_sync_positions, frontend, recover_burst


PSD_PEAK_THRESHOLD_DB = 15
BURST_STRIDE = 2880


def _psd_blind_search(iq: np.ndarray, fs: float) -> list[float]:
    f, psd = signal.welch(iq, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-12)
    nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=nf + PSD_PEAK_THRESHOLD_DB, distance=20)
    return [float(f[p]) for p in peaks]


def _lock_voice_phase(y: np.ndarray, anchor: int, polarity: float, sync_type: str) -> float:
    ref = SYNC_TEMPLATES[sync_type]
    levels = np.array([-3, -1, 1, 3])
    best = (1e18, 0.0)
    for ph in np.linspace(-8, 8, 65):
        start = anchor - (54 + 12) * SPS + ph
        pos = start + np.arange(132) * SPS
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue
        seg = polarity * _interp(y, pos)
        sy = seg[54:78]
        a, b = np.linalg.lstsq(np.vstack([sy, np.ones(24)]).T, ref, rcond=None)[0]
        segc = a * seg + b
        near = levels[np.argmin(np.abs(segc[:, None] - levels[None, :]), axis=1)]
        resid = np.mean((segc - near) ** 2)
        if resid < best[0]:
            best = (resid, ph)
    return best[1]


def _recover_stepped_burst(
    y: np.ndarray,
    anchor: int,
    j: int,
    ph: float,
    polarity: float,
    burst_stride_samples: int = BURST_STRIDE,
):
    start = anchor + burst_stride_samples * j - (54 + 12) * SPS + ph
    pos = start + np.arange(132) * SPS
    if pos[0] < 0 or pos[-1] >= len(y) - 1:
        return None
    seg = polarity * _interp(y, pos)
    return adaptive_slice_bits(seg)


def _decode_dmr_loop(y: np.ndarray, config: DMRConfig | None = None) -> list[dict]:
    config = config or DEFAULT_DMR_CONFIG
    positions = find_sync_positions(
        y,
        voice_threshold=config.sync_threshold_voice,
        data_threshold=config.sync_threshold_data,
        peak_distance_samples=config.sync_peak_distance_samples,
    )
    results = []
    seen_bursts: set[tuple] = set()

    for center, polarity, sync_type in positions:
        dedup_key = (round(center / 50), sync_type)
        if dedup_key in seen_bursts:
            continue
        seen_bursts.add(dedup_key)

        if "VOICE" in sync_type:
            ph = _lock_voice_phase(y, center, polarity, sync_type)
            collector = LateEntryCollector()
            for j in range(6):
                ba = _recover_stepped_burst(
                    y,
                    center,
                    j,
                    ph,
                    polarity,
                    burst_stride_samples=config.voice_burst_stride_samples,
                )
                if ba is None:
                    break
                pdu = collector.feed(ba, sync_type)
                if pdu is not None:
                    results.append(dict(pdu))
                    break
        else:
            symbols = recover_burst(y, center, polarity, sync_type)
            if symbols is None:
                continue
            pdu = decode_burst(symbols, sync_type)
            if pdu is not None:
                results.append(dict(pdu))

    return results


def decode(y: np.ndarray, config: DMRConfig | None = None) -> list[dict]:
    pdus = _decode_dmr_loop(y, config)
    for pdu in pdus:
        pdu.setdefault("protocol", "DMR")
    return pdus


def _resample_factors(source_sample_rate: float, target: float = Fs_dec) -> tuple[int, int]:
    up = int(round(target))
    down = int(round(source_sample_rate))
    g = gcd(up, down)
    return up // g, down // g


def _process_candidate(iq: np.ndarray, fo: float, fs_in: float) -> list[dict]:
    t = np.arange(len(iq)) / fs_in
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * fo * t)
    if abs(fs_in - Fs_dec) < 1:
        iq_dec = iq_shifted
    elif abs(fs_in - Fs_wide) < 1:
        iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)
    else:
        up, down = _resample_factors(fs_in)
        iq_dec = signal.resample_poly(iq_shifted, up, down)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
    results = decode(y)
    for pdu in results:
        pdu["_fo_hz"] = fo
    return results


def _process_narrowband(iq: np.ndarray, fs_in: float | None = None) -> list[dict]:
    fs = fs_in or Fs_dec
    if abs(fs - Fs_dec) < 1:
        iq_dec = iq
    else:
        up, down = _resample_factors(fs)
        iq_dec = signal.resample_poly(iq, up, down)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
    return decode(y)


def _dedup_pdus(pdus: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for pdu in pdus:
        fo_bucket = round(pdu.get("_fo_hz", 0) / 5000) * 5000
        key = (
            pdu.get("protocol", "DMR"),
            pdu.get("src", 0),
            pdu.get("dst", 0),
            pdu.get("type", ""),
            fo_bucket,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(pdu)
    return unique


def scan_file(path: str, freq_list: list[float] | None = None) -> list[dict]:
    iq = read_rawiq(path)
    fs = detect_sample_rate(path)

    if freq_list is not None:
        fs_in = fs or Fs_wide
        all_pdus = []
        for fo in freq_list:
            all_pdus.extend(_process_candidate(iq, fo, fs_in))
        return _dedup_pdus(all_pdus)
    if fs is None or fs > 200_000:
        fs_in = fs or Fs_wide
        all_pdus = []
        for fo in _psd_blind_search(iq, fs_in):
            all_pdus.extend(_process_candidate(iq, fo, fs_in))
        return _dedup_pdus(all_pdus)
    return _dedup_pdus(_process_narrowband(iq, fs))
