import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Use standard Matplotlib font settings for perfect English rendering
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True

# ================== Parameters ==================
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

# File and target candidate offset for visualization
file_to_debug = "synthesized_wideband_2.5MHz.rawiq"
f_offset_candidate = -300000.0  # Analyze the DMR 1 signal at -300 kHz

def main():
    if not os.path.exists(file_to_debug):
        print(f"Error: {file_to_debug} not found. Please run synthesis.py first!")
        return

    print(f"Analyzing {file_to_debug} at offset {f_offset_candidate/1e3:.1f} kHz ...")
    
    # 1. Read binary rawiq data
    data = np.fromfile(file_to_debug, dtype=np.int16)
    I, Q = data[0::2], data[1::2]
    length = min(len(I), len(Q))
    iq = (I[:length] + 1j * Q[:length]) / 32768.0

    # 2. Digital Down-Conversion (DDC) to baseband
    t = np.arange(len(iq)) / Fs_wide
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * f_offset_candidate * t)
    
    # 3. Polyphase Decimation (2.5 MHz -> 48 kHz)
    iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)
    
    # 4. NFM Demodulation & Calibration
    y_demod = np.angle(iq_dec[1:] * np.conj(iq_dec[:-1]))
    y_demod_centered = y_demod - np.mean(y_demod)
    y_scaled = y_demod * (3.0 / (2.0 * np.pi * 1944.0 / Fs_dec))
    
    # [Clipped Signal]: Prevents noise phase jumps from bloating the NCC denominator
    y_clipped = np.clip(y_scaled, -4.0, 4.0)

    # 5. Ideal BS Sourced Voice SYNC Wave Template (Table 9.2)
    t_symbols = hex_to_symbols("7F7D5DD57DFD")  # MS Sourced SYNC (Uplink)
    t_wave = np.repeat(t_symbols, SPS)
    window_len = len(t_wave)

    # 6. Compute Linear and Normalized Cross-Correlation (NCC)
    corr_linear = signal.correlate(y_clipped, t_wave, mode='same')
    
    # Calculate moving sum of squares (sliding window energy)
    y_sq = y_clipped ** 2
    y_energy = np.convolve(y_sq, np.ones(window_len), mode='same')
    y_energy = np.where(y_energy == 0, 1e-10, y_energy)
    t_energy = np.sum(t_wave ** 2)
    
    # Mathematically rigorous NCC bounded in [-1.0, 1.0]
    ncc = corr_linear / np.sqrt(y_energy * t_energy)

    # ---------------- Plotting Main Interface (All English) ----------------
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # Subplot 1: Demodulated Frequency Deviation
    # Slice the plot to a smaller window if 10 seconds is too dense, or plot full
    # For 10s, there are 480k samples. We plot the whole array to see overall activity.
    axs[0].plot(y_clipped, color='tab:blue', alpha=0.7, label="Demodulated Freq Deviation (y_clipped)")
    axs[0].axhline(3, color='r', linestyle='--', alpha=0.5, label="Level +3 (1944Hz)")
    axs[0].axhline(-3, color='r', linestyle='--', alpha=0.5, label="Level -3 (-1944Hz)")
    axs[0].set_title("1. Demodulated Frequency Deviation (NFM Discriminator Output)")
    axs[0].set_ylabel("Normalized Level")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)
    axs[0].set_ylim([-5, 5])

    # Subplot 2: Linear Cross-Correlation
    axs[1].plot(corr_linear, color='tab:orange', label="Linear Cross-Correlation")
    axs[1].set_title("2. Linear Cross-Correlation (Linearly dependent on signal amplitude)")
    axs[1].set_ylabel("Correlation Value")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    # Subplot 3: Normalized Cross-Correlation (NCC)
    axs[2].plot(ncc, color='tab:green', label="Normalized Correlation (NCC)")
    axs[2].axhline(0.5, color='gray', linestyle=':', label="0.5 Threshold")
    axs[2].axhline(NCC_THRESHOLD, color='red', linestyle='-.', label=f"{NCC_THRESHOLD} Threshold")
    axs[2].set_title("3. Normalized Cross-Correlation (NCC) (Shape similarity metric bounded in [-1.0, 1.0])")
    axs[2].set_xlabel("Sample Index (at 48 kHz)")
    axs[2].set_ylabel("Correlation Coefficient")
    axs[2].legend(loc="upper right")
    axs[2].grid(True)
    axs[2].set_ylim([-1.1, 1.1])

    # Find the peak of the NCC to pop up a zoomed staircase waveform window
    peaks, _ = signal.find_peaks(ncc, height=0.6, distance=800)
    if len(peaks) > 0:
        target_peak = peaks[0]
        print(f"DMR Signal Aligned! First SYNC Center Index: {target_peak}, NCC Score: {ncc[target_peak]:.3f}")
        
        # Mark all detected peaks in Plot 3
        axs[2].plot(peaks, ncc[peaks], "x", color='red', markersize=10, label="Detected SYNCs")
        axs[2].legend(loc="upper right")
        
        # Pop up Sub-window: Zoomed Staircase Wave
        plt.figure("DMR SYNC Staircase Waveform Zoom")
        # Crop 240 samples around the peak (120 samples left, 120 samples right)
        zoom_range = range(max(0, target_peak - 120), min(len(y_clipped), target_peak + 120))
        plt.plot(zoom_range, y_clipped[zoom_range], 'o-', color='tab:blue', label="Demodulated Waveform")
        
        # Draw theoretical 4FSK modulation lines
        plt.axhline(3, color='r', linestyle='--', alpha=0.7, label="Level +3 (+1944Hz)")
        plt.axhline(1, color='orange', linestyle=':', alpha=0.5, label="Level +1 (+648Hz)")
        plt.axhline(-1, color='orange', linestyle=':', alpha=0.5, label="Level -1 (-648Hz)")
        plt.axhline(-3, color='r', linestyle='--', alpha=0.7, label="Level -3 (-1944Hz)")
        
        plt.title(f"Zoomed DMR SYNC Code (Staircase Waveform at Center Index: {target_peak})")
        plt.xlabel("Sample Index")
        plt.ylabel("4FSK Level")
        plt.grid(True)
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()