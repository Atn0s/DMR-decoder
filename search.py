import os
import numpy as np
import scipy.signal as signal

# ================== Constants ==================
Fs_wide = 2500000.0        # 2.5 MHz Wideband sampling rate
Fs_dec = 48000.0           # Baseband sampling rate (SPS = 10)
SPS = 10

# Resampling factors: 2.5 MHz * 12 / 625 = 48 kHz
UP_FACTOR = 12
DOWN_FACTOR = 625

# NCC Threshold
NCC_THRESHOLD = 0.68

def hex_to_symbols(hex_str):
    bin_str = "".join(f"{int(c, 16):04b}" for c in hex_str)
    symbols = []
    for i in range(0, len(bin_str), 2):
        dibit = bin_str[i:i+2]
        if dibit == '01': symbols.append(3)
        elif dibit == '00': symbols.append(1)
        elif dibit == '10': symbols.append(-1)
        elif dibit == '11': symbols.append(-3)
    return np.array(symbols)

# Multi-template Dictionary (DMR Voice Sync, Data Sync is detected via negative peak)
templates_wave = {
    "BS Sourced": np.repeat(hex_to_symbols("755FD7DF75F7"), 10), # Base Station Downlink
    "MS Sourced": np.repeat(hex_to_symbols("7F7D5DD57DFD"), 10)  # Handheld Uplink
}

def read_rawiq(filename):
    data = np.fromfile(filename, dtype=np.int16)
    I, Q = data[0::2], data[1::2]
    length = min(len(I), len(Q))
    return (I[:length] + 1j * Q[:length]) / 32768.0

def verify_periodicity(peaks, target_period_ms, tolerance_ms=15.0):
    """
    DMR 专属协议指纹校验：
    检查检测到的同步峰，其相邻间距是否符合 DMR 协议规定的周期 (如 30ms, 60ms 或 360ms) 的整数倍
    """
    if len(peaks) < 3:
        # 如果 10 秒内连 3 个同步峰都凑不齐，绝对是噪声或非DMR信号触发的偶发虚警
        return False
        
    diffs = np.diff(peaks) / Fs_dec * 1000.0  # 转换为毫秒(ms)
    valid_count = 0
    
    for diff in diffs:
        # 检查是否为目标周期的近似整数倍
        remainder = diff % target_period_ms
        if remainder < tolerance_ms or remainder > (target_period_ms - tolerance_ms):
            valid_count += 1
            
    # 如果有超过 60% 的间距符合协议周期，通过校验
    return (valid_count / len(diffs)) >= 0.6

def main():
    target_file = "synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found. Please run synthesis.py first!")
        return
        
    print("="*80)
    print("      Automated DMR Signal Retrieval with Protocol Periodicity Check")
    print("="*80)
    print(f"Analyzing Wideband Scenario: {target_file}")
    
    iq = read_rawiq(target_file)
    
    # ---- Stage 1: Coarse Screening (Welch PSD) ----
    f_welch, psd_welch = signal.welch(iq, fs=Fs_wide, nperseg=4096, return_onesided=False)
    f_welch = np.fft.fftshift(f_welch)
    psd_welch = np.fft.fftshift(psd_welch)
    psd_db = 10 * np.log10(psd_welch)
    
    noise_floor = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=noise_floor + 15, distance=20)
    
    if len(peaks) == 0:
        print("  -> Status: Silent. No active signals detected.")
        return
        
    print(f"  -> Coarse Stage: Detected {len(peaks)} active narrowband candidates:")
    
    for idx, peak_idx in enumerate(peaks):
        f_offset = f_welch[peak_idx]
        print(f"\n  [Candidate {idx+1}] Offset: {f_offset/1e3:+.2f} kHz")
        
        # ---- Stage 2: Digital Down-Conversion & Decimation ----
        t = np.arange(len(iq)) / Fs_wide
        iq_shifted = iq * np.exp(-1j * 2 * np.pi * f_offset * t)
        iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)
        
        # ---- Stage 3: NFM Demodulation & CFO Cancellation ----
        y_demod = np.angle(iq_dec[1:] * np.conj(iq_dec[:-1]))
        
        # 【核心修正 1】：自适应频偏消除，将波形拉回对称零点
        y_demod_centered = y_demod - np.mean(y_demod)
        
        y_scaled = y_demod_centered * (3.0 / (2.0 * np.pi * 1944.0 / Fs_dec))
        y_clipped = np.clip(y_scaled, -5.0, 5.0)
        
        # ---- Stage 4: Multi-Template NCC Match ----
        is_identified = False
        
        for t_name, t_wave in templates_wave.items():
            window_len = len(t_wave)
            corr_linear = signal.correlate(y_clipped, t_wave, mode='same')
            
            y_sq = y_clipped ** 2
            y_energy = np.convolve(y_sq, np.ones(window_len), mode='same')
            y_energy = np.where(y_energy == 0, 1e-10, y_energy)
            t_energy = np.sum(t_wave ** 2)
            
            ncc = corr_linear / np.sqrt(y_energy * t_energy)
            
            peaks_pos, _ = signal.find_peaks(ncc, height=NCC_THRESHOLD, distance=800)
            peaks_neg, _ = signal.find_peaks(-ncc, height=NCC_THRESHOLD, distance=800)
            
            if len(peaks_pos) > 0:
                # 【核心修正 2】：如果是语音，校验相邻峰间距是否符合 360 ms 超级帧周期
                if verify_periodicity(peaks_pos, target_period_ms=360.0):
                    print(f"     └─ [MATCH CONFIRMED]: DMR 【{t_name}】 VOICE Signal!")
                    print(f"        Aligned Sync Centers (sample index at 48k): {peaks_pos[:10]} ...")
                    is_identified = True
                    break
            if len(peaks_neg) > 0:
                # 【核心修正 2】：如果是数据，校验相邻峰间距是否符合 30 ms 或 60 ms 时隙周期
                if verify_periodicity(peaks_neg, target_period_ms=30.0) or verify_periodicity(peaks_neg, target_period_ms=60.0):
                    print(f"     └─ [MATCH CONFIRMED]: DMR 【{t_name}】 DATA/CONTROL Signal!")
                    print(f"        Aligned Sync Centers (sample index at 48k): {peaks_neg[:10]} ...")
                    is_identified = True
                    break
                
        if not is_identified:
            print("     └─ [REJECTED]: Non-DMR signal (Successfully rejected P25 Phase 1 or noise)")

    print("\n" + "="*80)
    print("                               Search Completed")
    print("="*80)

if __name__ == '__main__':
    main()