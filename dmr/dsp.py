import numpy as np
import scipy.signal as signal
from bitarray import bitarray
from common.dsp import fsk_frontend, interp
from common.io import read_rawiq
from dmr.constants import (
    Fs_dec, SPS, DEV_NOMINAL,
    NCC_THRESHOLD_VOICE, NCC_THRESHOLD_DATA,
    SYNC_TEMPLATES,
)


def _interp(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return interp(arr, pos)


def adaptive_slice_bits(seg: np.ndarray) -> bitarray:
    hi = np.percentile(seg, 90)
    lo = np.percentile(seg, 10)
    if hi == lo:
        # degenerate: cannot distinguish levels, map everything to +1
        return bitarray([0, 0] * len(seg))
    center = 0.5 * (hi + lo)
    umid = 0.5 * (hi + center)
    lmid = 0.5 * (lo + center)
    bits = []
    for v in seg:
        if v >= umid:
            bits.extend([0, 1])
        elif v >= center:
            bits.extend([0, 0])
        elif v >= lmid:
            bits.extend([1, 0])
        else:
            bits.extend([1, 1])
    return bitarray(bits)


def frontend(iq_dec: np.ndarray, fo: float = 0.0, fs: float = Fs_dec,
             cutoff: float = 9500.0, ntaps: int = 151) -> np.ndarray:
    """DDC (fo!=0) + channel filter + FM discriminator + DC removal."""
    return fsk_frontend(
        iq_dec,
        fo=fo,
        fs=fs,
        cutoff=cutoff,
        ntaps=ntaps,
        dev_nominal=DEV_NOMINAL,
    )


def lc_front_end_compat(iq_dec: np.ndarray, cutoff: float = 9500.0,
                         ntaps: int = 151) -> np.ndarray:
    """Drop-in replacement for dmr_pipeline_v2.lc_front_end."""
    return frontend(iq_dec, fo=0.0, fs=Fs_dec, cutoff=cutoff, ntaps=ntaps)


def find_sync_positions(y: np.ndarray) -> list[tuple[int, float, str]]:
    """NCC scan all sync templates. Returns [(center_sample, polarity, sync_type)].
    sync_type in {'MS_VOICE','BS_VOICE','DATA_MS','DATA_BS'}"""
    results = []
    thresholds = {
        "MS_VOICE": NCC_THRESHOLD_VOICE,
        "BS_VOICE": NCC_THRESHOLD_VOICE,
        "DATA_MS": NCC_THRESHOLD_DATA,
        "DATA_BS": NCC_THRESHOLD_DATA,
    }
    for name, ref in SYNC_TEMPLATES.items():
        rwave = np.repeat(ref, SPS)
        c = signal.correlate(y, rwave, mode='same')
        e = np.convolve(y ** 2, np.ones(len(rwave)), mode='same')
        e = np.where(e <= 0, 1e-9, e)
        ncc = c / np.sqrt(e * np.sum(rwave ** 2))
        thr = thresholds[name]
        pos_peaks, _ = signal.find_peaks(ncc, height=thr, distance=800)
        neg_peaks, _ = signal.find_peaks(-ncc, height=thr, distance=800)
        for p in pos_peaks:
            results.append((int(p), 1.0, name))
        for p in neg_peaks:
            results.append((int(p), -1.0, name))
    results.sort(key=lambda x: x[0])
    return results


def recover_burst(y: np.ndarray, center: int, polarity: float,
                  sync_type: str) -> np.ndarray | None:
    """Sub-symbol phase sweep [-8,8] 65 steps, pick best phase by residual.
    Returns 132-symbol calibrated array or None."""
    ref = SYNC_TEMPLATES[sync_type]
    levels = np.array([-3, -1, 1, 3])
    best = (1e18, None)
    for ph in np.linspace(-8, 8, 65):
        start = center - (54 + 12) * SPS + ph
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
            best = (resid, segc)
    return best[1]
