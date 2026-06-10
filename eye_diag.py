"""Step-1 diagnostic: eye diagram + 4-level histogram on the CLEAN isolated DMR file.
Vectorized only (no Gardner loop) so it runs in a few seconds.
Goal: find why +-1 / +-3 don't separate cleanly."""
import numpy as np, scipy.signal as signal
import dmr_pipeline_v2 as P

Fs = 48000.0
SPS = 10

def demod(slice_n=300000, cutoff=6500.0, mf='box'):
    raw = P.read_rawiq('dmr_1_78125.rawiq')[:slice_n]
    iqd = signal.resample_poly(raw, 384, 625)          # 78125 -> 48000
    # residual carrier removal
    f, ps = signal.welch(iqd, fs=Fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f); ps = np.fft.fftshift(ps); cf = f[np.argmax(ps)]
    iqd = iqd * np.exp(-1j * 2 * np.pi * cf * np.arange(len(iqd)) / Fs)
    iqf = signal.filtfilt(signal.firwin(101, cutoff, fs=Fs), [1.0], iqd)
    yd = np.angle(iqf[1:] * np.conj(iqf[:-1]))
    pw = np.abs(iqf[:-1]) ** 2
    act = pw > np.median(pw) * 1.5
    yd = yd - np.mean(yd[act])
    hz = yd / (2 * np.pi) * Fs
    y = yd * (3.0 / (2 * np.pi * P.DEV_NOMINAL / Fs))   # scale to nominal +-3
    if mf == 'box':
        yf = np.convolve(y, np.ones(SPS) / SPS, mode='same')
    elif mf == 'half':                                   # narrow boxcar, center 6 taps
        w = np.zeros(SPS); w[2:8] = 1.0 / 6
        yf = np.convolve(y, w, mode='same')
    else:
        yf = y
    return y, yf, hz, cf, act

def find_syncs(yf):
    pk_all = {}
    for name in ['MS Sourced', 'BS Sourced']:
        ref = P.data_sync_sym[name]; rwave = np.repeat(ref, SPS)
        c = signal.correlate(yf, rwave, mode='same')
        pk, _ = signal.find_peaks(np.abs(c), height=0.5 * np.max(np.abs(c)), distance=800)
        pk_all[name] = (pk, c)
    return pk_all
