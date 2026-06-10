import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Matplotlib 标准渲染设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True

def read_rawiq(filename, target_len):
    """
    读取 s16le 格式的原始窄带 IQ 文件并截取前 10s 数据
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: {filename} not found in this directory.")
    
    data = np.fromfile(filename, dtype=np.int16)
    I, Q = data[0::2], data[1::2]
    length = min(len(I), len(Q), target_len)
    
    # 转换为归一化复数 float
    return (I[:length] + 1j * Q[:length]) / 32768.0

def plot_narrowband_psd(filename):
    Fs = 78125.0       # 原始窄带采样率
    duration = 10.0    # 10s 时长
    target_len = int(Fs * duration)
    
    print(f"Loading {filename} (Extracting first {duration}s)...")
    try:
        iq = read_rawiq(filename, target_len)
    except Exception as e:
        print(e)
        return
    
    # ---- 1. 高分辨率 Welch 功率谱估计 ----
    # 16384 点 FFT 提供约 4.77 Hz 的谱分辨率
    nperseg = 16384
    f, psd = signal.welch(iq, fs=Fs, nperseg=nperseg, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd)
    
    # ---- 2. 锁定中心 [-2000, 2000] Hz 区域并进行高精度插值 ----
    mask = (f >= -2000) & (f <= 2000)
    f_sub = f[mask]
    psd_sub = psd_db[mask]
    sub_peak_idx = np.argmax(psd_sub)
    
    # 找到该点在全局数组中的对应索引
    peak_idx = np.where(f == f_sub[sub_peak_idx])[0][0]
    
    # 二次抛物线插值，估算极高精度的物理中心
    f_refined = f[peak_idx]
    if 0 < peak_idx < len(psd_db) - 1:
        y0 = psd_db[peak_idx]
        ym1 = psd_db[peak_idx - 1]
        yp1 = psd_db[peak_idx + 1]
        denom = ym1 - 2 * y0 + yp1
        if abs(denom) > 1e-10:
            d = 0.5 * (ym1 - yp1) / denom
            f_refined += d * (Fs / nperseg)
            
    print(f"\nAnalysis Results for {filename}:")
    print(f"  Coarse Peak Location : {f[peak_idx]:.2f} Hz")
    print(f"  Refined CFO Offset   : {f_refined:+.2f} Hz")
    
    # ---- 3. 精细化绘图 ----
    plt.figure(figsize=(11, 6), dpi=120)
    
    # 绘制完整的 PSD 曲线
    plt.plot(f, psd_db, color='#1f77b4', linewidth=1.2, label='Baseband PSD')
    
    # 标出理论零频位置（红色虚线）
    plt.axvline(0, color='crimson', linestyle='--', linewidth=1.2, alpha=0.8,
                label='Theoretical Center (0.00 Hz)')
    
    # 标出算法算出的实际 CFO 峰值位置（绿色实线）
    plt.axvline(f_refined, color='#2ca02c', linestyle='-', linewidth=1.5, alpha=0.9,
                label=f'Actual CFO Peak ({f_refined:+.2f} Hz)')
    
    # 高倍率缩放中心区域，精细展现几十到几百 Hz 的偏差
    plt.xlim(-2500, 2500)
    
    # 动态适应 y 轴显示区间
    zoom_psd = psd_db[(f >= -2500) & (f <= 2500)]
    plt.ylim(np.min(zoom_psd) - 3, np.max(zoom_psd) + 5)
    
    # 文本与箭头的指向标注
    plt.annotate(f'CFO Offset: {f_refined:+.1f} Hz',
                 xy=(f_refined, np.max(zoom_psd)),
                 xytext=(f_refined + 250, np.max(zoom_psd) - 1.5),
                 arrowprops=dict(facecolor='darkgreen', shrink=0.08, width=1, headwidth=6, headlength=6),
                 fontsize=10, fontweight='semibold', color='darkgreen')
    
    # 图表细节修饰
    plt.title(f'Narrowband PSD Zoom-In: {filename} (10s Average)', fontsize=12, pad=12)
    plt.xlabel('Frequency Offset relative to Baseband (Hz)', fontsize=10)
    plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=10)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 你可以随时在此切换想要观察的原始窄带文件
    # 1) dmr_1_78125.rawiq   (对应 DUC -300kHz 候选信号 1)
    # 2) dmr_2_78125.rawiq   (对应 DUC +150kHz 候选信号 2)
    # 3) p25_1_78125.rawiq   (对应 DUC +600kHz 候选信号 3)
    target_file = 'p25_1_78125.rawiq' 
    plot_narrowband_psd(target_file)