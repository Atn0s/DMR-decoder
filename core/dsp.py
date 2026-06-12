import numpy as np
import scipy.signal as signal
from bitarray import bitarray
from core.burst_type import (
    Fs_dec, SPS, DEV_NOMINAL,
    NCC_THRESHOLD_VOICE, NCC_THRESHOLD_DATA,
    SYNC_TEMPLATES,
)


def read_rawiq(filename: str) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.int16)
    I, Q = data[0::2], data[1::2]
    n = min(len(I), len(Q))
    return (I[:n] + 1j * Q[:n]) / 32768.0


def _interp(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    i = np.floor(pos).astype(int)
    fr = pos - i
    i = np.clip(i, 0, len(arr) - 2)
    return arr[i] * (1 - fr) + arr[i + 1] * fr


def adaptive_slice_bits(seg: np.ndarray) -> bitarray:
    hi = np.percentile(seg, 90)
    lo = np.percentile(seg, 10)
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
    if fo != 0.0:
        n = np.arange(len(iq_dec))
        iq_dec = iq_dec * np.exp(-1j * 2 * np.pi * fo * n / fs)
    f, ps = signal.welch(iq_dec, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    ps = np.fft.fftshift(ps)
    cf = f[np.argmax(ps)]
    n = np.arange(len(iq_dec))
    iqf = iq_dec * np.exp(-1j * 2 * np.pi * cf * n / fs)
    iqf = signal.filtfilt(signal.firwin(ntaps, cutoff, fs=fs), [1.0], iqf)
    yd = np.angle(iqf[1:] * np.conj(iqf[:-1]))
    amp = np.abs(iqf[:-1])
    active = amp > (np.median(amp) + 0.3 * (np.mean(amp) - np.median(amp)))
    center = np.median(yd[active]) if np.any(active) else np.median(yd)
    return (yd - center) * (3.0 / (2.0 * np.pi * DEV_NOMINAL / fs))


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
