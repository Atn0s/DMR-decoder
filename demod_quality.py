"""4FSK 解调质量可视化：星座直方图 + 眼图 + 符号散点 + 时域判决。
用已验证的 lc_front_end + recover_burst_symbols 路径，证明符号层已经收敛。"""
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dmr_pipeline_v2 as P

iq = P.read_rawiq('synthesized_wideband_2.5MHz.rawiq')
f_w, psd_w = signal.welch(iq, fs=P.Fs_wide, nperseg=4096, return_onesided=False)
f_w = np.fft.fftshift(f_w); psd_w = np.fft.fftshift(psd_w)
psd_db = 10 * np.log10(psd_w); nf = np.median(psd_db)
peaks, _ = signal.find_peaks(psd_db, height=nf + 15, distance=20)
fo = f_w[peaks[0]]
t = np.arange(len(iq)) / P.Fs_wide
iqd = signal.resample_poly(iq * np.exp(-1j * 2 * np.pi * fo * t), P.UP_FACTOR, P.DOWN_FACTOR)

y = P.lc_front_end(iqd)

# 收集所有 Golay-OK 突发的校准符号，构成星座统计
all_syms = []
vlc_seg = None
for name in ['MS Sourced', 'BS Sourced']:
    for sc, sgn in P.find_data_sync_positions(y, name):
        seg = P.recover_burst_symbols(y, sc, sgn, name)
        if seg is None:
            continue
        ba = P.adaptive_slice_bits(seg)
        slot = ba[98:108] + ba[156:166]
        if P.Golay2087.check(slot.copy()):
            all_syms.append(seg)
            if vlc_seg is None and P.ba2int(slot[4:8]) == 1:
                vlc_seg = seg
allc = np.concatenate(all_syms) if all_syms else np.array([])
levels = np.array([-3, -1, 1, 3])
near = levels[np.argmin(np.abs(allc[:, None] - levels[None, :]), axis=1)]
rmse = np.sqrt(np.mean((allc - near) ** 2))
frac = np.mean(np.min(np.abs(allc[:, None] - levels[None, :]), axis=1) < 0.5)

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('DMR 4FSK demod quality (Golay-OK bursts: %d, RMSE=%.3f, %%within0.5=%.0f%%)'
             % (len(all_syms), rmse, 100 * frac), fontsize=13, fontweight='bold')

# [0,0] 四电平直方图
axs[0, 0].hist(allc, bins=120, range=(-4, 4), color='steelblue')
for lv in levels:
    axs[0, 0].axvline(lv, color='r', ls='--', alpha=0.7)
axs[0, 0].set_title('Constellation histogram (red = ideal -3/-1/+1/+3)')
axs[0, 0].set_xlabel('symbol value'); axs[0, 0].grid(True, alpha=0.3)

# [0,1] 符号散点（前若干突发）
flat = allc[:2000]
axs[0, 1].plot(flat, np.zeros_like(flat) + np.random.uniform(-0.3, 0.3, len(flat)),
               '.', ms=2, alpha=0.4)
for lv in levels:
    axs[0, 1].axvline(lv, color='r', ls='--', alpha=0.7)
axs[0, 1].set_title('Symbol scatter (jittered)')
axs[0, 1].set_xlabel('symbol value'); axs[0, 1].set_yticks([]); axs[0, 1].grid(True, alpha=0.3)

# [1,0] 眼图：在 VLC 突发附近的样点流，按 2 符号折叠
if vlc_seg is not None:
    # 用样点域眼图更直观：取一段 active 样点
    sc0, sgn0 = P.find_data_sync_positions(y, 'MS Sourced')[0]
    s = sgn0 * y[sc0 - 70 * P.SPS: sc0 + 70 * P.SPS]
    win = 2 * P.SPS
    n = (len(s) // win) * win
    mat = s[:n].reshape(-1, win)
    for row in mat:
        axs[1, 0].plot(row, color='b', alpha=0.15)
    axs[1, 0].set_title('Eye diagram (2-symbol window, sample domain)')
    axs[1, 0].set_xlabel('sample within window'); axs[1, 0].grid(True, alpha=0.3)

# [1,1] 一个 VLC 突发的 132 符号时域 + 判决
if vlc_seg is not None:
    axs[1, 1].plot(vlc_seg, '.-', ms=4, color='darkgreen', label='calibrated symbols')
    near_v = levels[np.argmin(np.abs(vlc_seg[:, None] - levels[None, :]), axis=1)]
    axs[1, 1].step(np.arange(132), near_v, color='orange', alpha=0.7, where='mid', label='decision')
    axs[1, 1].axvspan(54, 78, color='gray', alpha=0.2, label='SYNC')
    for lv in levels:
        axs[1, 1].axhline(lv, color='r', ls=':', alpha=0.4)
    axs[1, 1].set_title('One Voice-LC-Header burst (132 symbols)')
    axs[1, 1].set_xlabel('symbol index'); axs[1, 1].legend(fontsize=8); axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('demod_quality.png', dpi=90)
print('saved demod_quality.png')
print('Golay-OK bursts=%d  total symbols=%d  RMSE=%.3f  %%within0.5=%.1f%%'
      % (len(all_syms), len(allc), rmse, 100 * frac))
