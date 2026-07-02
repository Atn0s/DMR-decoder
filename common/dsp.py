from __future__ import annotations

import numpy as np
import scipy.signal as signal


def interp(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    i = np.floor(pos).astype(int)
    fr = pos - i
    i = np.clip(i, 0, len(arr) - 2)
    return arr[i] * (1 - fr) + arr[i + 1] * fr


def fsk_frontend(
    iq_dec: np.ndarray,
    fo: float = 0.0,
    fs: float = 48_000.0,
    cutoff: float = 9500.0,
    ntaps: int = 151,
    dev_nominal: float = 1944.0,
    min_samples: int = 512,
    psd_nperseg: int = 4096,
) -> np.ndarray:
    """DDC + channel filter + FM discriminator + DC removal for narrowband FSK."""
    if len(iq_dec) < min_samples:
        raise ValueError(
            f"frontend: signal too short ({len(iq_dec)} samples), need >= {min_samples}"
        )
    if fo != 0.0:
        n = np.arange(len(iq_dec))
        iq_dec = iq_dec * np.exp(-1j * 2 * np.pi * fo * n / fs)
    f, ps = signal.welch(iq_dec, fs=fs, nperseg=psd_nperseg, return_onesided=False)
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
    return (yd - center) * (3.0 / (2.0 * np.pi * dev_nominal / fs))
