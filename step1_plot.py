"""Step-1 visual diagnostic on the CLEAN isolated DMR file -> PNG.
Answers: is the 4-level eye closed because of low SNR, or wrong demod interpretation?"""
import numpy as np, scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dmr_pipeline_v2 as P

Fs = 48000.0
SPS = 10
raw = P.read_rawiq('dmr_1_78125.rawiq')[:200000]
iqd = signal.resample_poly(raw, 384, 625)                  # 78125 -> 48000
f, ps = signal.welch(iqd, fs=Fs, nperseg=4096, return_onesided=False)
fc = np.fft.fftshift(f); psc = np.fft.fftshift(ps); cf = fc[np.argmax(psc)]
iqd = iqd * np.exp(-1j * 2 * np.pi * cf * np.arange(len(iqd)) / Fs)
iqf = signal.filtfilt(signal.firwin(151, 9000.0, fs=Fs), [1.0], iqd)
yd = np.angle(iqf[1:] * np.conj(iqf[:-1]))
pw = np.abs(iqf[:-1]) ** 2
act = pw > np.median(pw) * 1.5
yd = yd - np.mean(yd[act])
hz = yd / (2 * np.pi) * Fs                                  # instantaneous freq in Hz

fig, axs = plt.subplots(2, 2, figsize=(14, 9))

# spectrum
fb, pb = signal.welch(iqd, fs=Fs, nperseg=2048, return_onesided=False)
fb = np.fft.fftshift(fb); pb = np.fft.fftshift(pb)
axs[0, 0].plot(fb / 1e3, 10 * np.log10(pb))
axs[0, 0].set_title('Baseband spectrum (clean DMR file)')
axs[0, 0].set_xlabel('kHz'); axs[0, 0].grid(True)

# instantaneous-frequency histogram on active samples
hzact = hz[act]
axs[0, 1].hist(hzact, bins=120, range=(-4000, 4000))
for lv in [-1944, -648, 648, 1944]:
    axs[0, 1].axvline(lv, color='r', ls='--', alpha=0.6)
axs[0, 1].set_title('Inst-freq histogram (red=expected +-648/+-1944 Hz)')
axs[0, 1].set_xlabel('Hz'); axs[0, 1].grid(True)

# eye diagram: overlay 2-symbol windows, aligned to active start
idx = np.where(act)[0]
seg = hz[idx[0]:idx[0] + 4000]
win = 2 * SPS
n = (len(seg) // win) * win
mat = seg[:n].reshape(-1, win)
for row in mat[:200]:
    axs[1, 0].plot(row, color='b', alpha=0.05)
for lv in [-1944, -648, 648, 1944]:
    axs[1, 0].axhline(lv, color='r', ls='--', alpha=0.5)
axs[1, 0].set_title('Eye diagram (2-symbol = 20 samples, freq Hz)')
axs[1, 0].set_xlabel('sample within window'); axs[1, 0].grid(True)

# time series snippet
axs[1, 1].plot(hz[idx[0]:idx[0] + 300], '.-', ms=3)
for lv in [-1944, -648, 648, 1944]:
    axs[1, 1].axhline(lv, color='r', ls='--', alpha=0.5)
axs[1, 1].set_title('Inst-freq time series (300 samples)')
axs[1, 1].set_xlabel('sample'); axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('step1_diag.png', dpi=90)
print('saved step1_diag.png  residual_carrier=%.0f Hz' % cf)
print('active inst-freq: std=%.0f Hz  p1=%.0f p99=%.0f' % (np.std(hzact), np.percentile(hzact,1), np.percentile(hzact,99)))
