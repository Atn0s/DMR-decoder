import os
import numpy as np
from scipy.signal import resample_poly
import scipy.signal as signal
import matplotlib.pyplot as plt

# Standard Matplotlib font settings for English rendering
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True

# ================== Helper Functions ==================
def read_rawiq(filename):
    """
    Read s16le format rawiq files and convert to complex float normalized in [-1, 1]
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: {filename} not found.")
    
    data = np.fromfile(filename, dtype=np.int16)
    I = data[0::2]
    Q = data[1::2]
    length = min(len(I), len(Q))
    iq = (I[:length] + 1j * Q[:length]) / 32768.0
    return iq

def extract_or_pad(sig, target_len):
    """
    Truncate or pad zeros to ensure the signal is exactly target_len
    """
    if len(sig) >= target_len:
        return sig[:target_len]
    else:
        return np.pad(sig, (0, target_len - len(sig)), mode='constant')

# ================== Core Parameters ==================
Fs_in = 78125              # Input narrowband sample rate (78.125 kHz)
L = 32                     # Simple integer interpolation factor (32x)
Fs_out = Fs_in * L         # Output wideband sample rate (2.5 MHz)
duration = 10.0             # Total duration of the simulation (seconds)

num_samples_in = int(Fs_in * duration)   # Narrowband samples needed (approx 39,062)
num_samples_out = int(Fs_out * duration) # Wideband samples generated (approx 1,250,000)

# Target offset frequencies inside the 2.5 MHz spectrum
f_offset_dmr1 = -300000.0   # DMR 1 shifted to -300 kHz
f_offset_dmr2 =  150000.0   # DMR 2 shifted to +150 kHz
f_offset_p25  =  600000.0   # P25 1 shifted to +600 kHz

SNR_dB = 20                # Simulated SNR in dB

# ================== Main Process ==================
def main():
    print("="*60)
    print("      DMR/P25 Single-File 2.5 MHz Wideband Scenario Generator")
    print("="*60)

    print("Loading narrowband IQ recordings...")
    try:
        sig_dmr1 = read_rawiq('dmr_1_78125.rawiq')
        sig_dmr2 = read_rawiq('dmr_2_78125.rawiq')
        sig_p25  = read_rawiq('p25_1_78125.rawiq')
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Please verify dmr_1_78125.rawiq, dmr_2_78125.rawiq, p25_1_78125.rawiq exist in this directory.")
        return

    # Memory Optimization: Cut the signals to the required duration first!
    print(f"Truncating narrowband inputs to first {duration} seconds ({num_samples_in} samples)...")
    sig_dmr1_cut = extract_or_pad(sig_dmr1, num_samples_in)
    sig_dmr2_cut = extract_or_pad(sig_dmr2, num_samples_in)
    sig_p25_cut  = extract_or_pad(sig_p25,  num_samples_in)

    print(f"Interpolating narrowband segments by 32x to reach 2.5 MHz...")
    # resample_poly performs polyphase FIR interpolation with zero-insertion, very fast!
    sig_dmr1_up = resample_poly(sig_dmr1_cut, L, 1)
    sig_dmr2_up = resample_poly(sig_dmr2_cut, L, 1)
    sig_p25_up  = resample_poly(sig_p25_cut,  L, 1)

    # Force strict sample alignment in case resample_poly output deviates slightly
    sig1 = extract_or_pad(sig_dmr1_up, num_samples_out)
    sig2 = extract_or_pad(sig_dmr2_up, num_samples_out)
    sig3 = extract_or_pad(sig_p25_up,  num_samples_out)

    print("Modulating baseband signals onto high-frequency carriers...")
    t = np.arange(num_samples_out) / Fs_out
    
    # Digital Up-Conversion (DUC)
    sig1_mod = sig1 * np.exp(1j * 2 * np.pi * f_offset_dmr1 * t)
    sig2_mod = sig2 * np.exp(1j * 2 * np.pi * f_offset_dmr2 * t)
    sig3_mod = sig3 * np.exp(1j * 2 * np.pi * f_offset_p25 * t)

    # Combine channels
    wideband_base = sig1_mod + sig2_mod + sig3_mod

    print("Adding AWGN background channel noise...")
    sig_power = np.mean(np.abs(wideband_base)**2)
    noise_power = sig_power / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples_out) + 1j * np.random.randn(num_samples_out))
    
    wideband_noisy = wideband_base + noise

    # Scale to prevent clipping during s16le saving
    wideband_scaled = (wideband_noisy / np.max(np.abs(wideband_noisy))) * 0.9

    # Save to synthesized rawiq file
    I_out = np.clip(np.round(wideband_scaled.real * 32767), -32768, 32767).astype(np.int16)
    Q_out = np.clip(np.round(wideband_scaled.imag * 32767), -32768, 32767).astype(np.int16)

    out_data = np.empty((2 * num_samples_out,), dtype=np.int16)
    out_data[0::2] = I_out
    out_data[1::2] = Q_out

    output_filename = "synthesized_wideband_2.5MHz.rawiq"
    out_data.tofile(output_filename)
    print(f"\nScenario successfully generated: {output_filename}")
    print(f"Sample Rate: {Fs_out / 1e6:.3f} MHz, Samples: {num_samples_out}")

    # ================== Plotting Spectrum (All English) ==================
    print("Plotting Power Spectral Density (PSD)...")
    # Estimate PSD using Welch's method (Centered spectrum for complex signals)
    f_welch, psd_welch = signal.welch(wideband_scaled, fs=Fs_out, nperseg=4096, return_onesided=False)
    f_welch = np.fft.fftshift(f_welch)
    psd_welch = np.fft.fftshift(psd_welch)
    psd_db = 10 * np.log10(psd_welch)

    plt.figure(figsize=(10, 6))
    plt.plot(f_welch / 1e6, psd_db, color='teal', linewidth=1)
    plt.grid(True)
    plt.xlabel('Offset Frequency (MHz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title(f'Synthesized 2.5 MHz SDR Spectrum (SNR={SNR_dB}dB)')
    
    # Text Annotations on the plot
    plt.text(f_offset_dmr1 / 1e6 - 0.1, -45, r'$\leftarrow$ DMR 1 (-300kHz)', color='red', fontsize=10)
    plt.text(f_offset_dmr2 / 1e6 + 0.05, -45, r'$\leftarrow$ DMR 2 (+150kHz)', color='green', fontsize=10)
    plt.text(f_offset_p25  / 1e6 + 0.05, -45, r'$\leftarrow$ P25 1 (+600kHz)', color='blue', fontsize=10)
    
    plt.show()

def synthesize_wideband_grid(placements, out_path, fs_out, dur_sec,
                             fs_in=78125, snr_db=20, data_dir="data"):
    """Synthesize a wideband IQ file with several narrowband signals placed on a
    frequency grid, all present for the whole duration.

    placements: list of (fo_hz, src_filename). Each source is truncated/padded to
    dur_sec, upsampled to fs_out, shifted to fo_hz, summed; then wideband AWGN at
    snr_db is added and the result is scaled and saved as interleaved int16.
    Returns out_path."""
    L = int(round(fs_out / fs_in))
    n_out = int(dur_sec * fs_out)
    wideband = np.zeros(n_out, dtype=np.complex128)
    t = np.arange(n_out) / fs_out

    for (fo_hz, fname) in placements:
        narrow = read_rawiq(os.path.join(data_dir, fname))
        seg = extract_or_pad(narrow, int(dur_sec * fs_in))
        up = extract_or_pad(resample_poly(seg, L, 1), n_out)
        wideband += up * np.exp(1j * 2 * np.pi * fo_hz * t)

    sig_power = np.mean(np.abs(wideband) ** 2)
    if sig_power > 0:
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(n_out) + 1j * np.random.randn(n_out))
        wideband = wideband + noise

    peak = np.max(np.abs(wideband))
    if peak > 0:
        wideband = (wideband / peak) * 0.9
    I_out = np.clip(np.round(wideband.real * 32767), -32768, 32767).astype(np.int16)
    Q_out = np.clip(np.round(wideband.imag * 32767), -32768, 32767).astype(np.int16)
    out_data = np.empty(2 * n_out, dtype=np.int16)
    out_data[0::2] = I_out
    out_data[1::2] = Q_out
    out_data.tofile(out_path)
    return out_path


if __name__ == '__main__':
    main()


def synthesize_scenario(scenario, out_path, fs_out=2_500_000.0, fs_in=78125,
                        snr_db=20, data_dir="data"):
    """Synthesize a wideband IQ file from a time-scripted scenario.

    scenario: list of (start_sec, dur_sec, fo_hz, src_filename).
    Each signal appears ONLY within its time window (zeros outside), is upsampled
    to fs_out, shifted to fo_hz, summed, then AWGN is added at snr_db (wideband).
    Returns out_path."""
    L = int(round(fs_out / fs_in))
    total_sec = max(s + d for (s, d, _, _) in scenario)
    n_out = int(total_sec * fs_out)
    wideband = np.zeros(n_out, dtype=np.complex128)

    for (start_sec, dur_sec, fo_hz, fname) in scenario:
        narrow = read_rawiq(os.path.join(data_dir, fname))
        n_in_needed = int(dur_sec * fs_in)
        seg = extract_or_pad(narrow, n_in_needed)
        up = resample_poly(seg, L, 1)
        n_seg = len(up)
        start_idx = int(start_sec * fs_out)
        end_idx = min(start_idx + n_seg, n_out)
        t = np.arange(end_idx - start_idx) / fs_out
        carrier = np.exp(1j * 2 * np.pi * fo_hz * t)
        wideband[start_idx:end_idx] += up[:end_idx - start_idx] * carrier

    sig_power = np.mean(np.abs(wideband) ** 2)
    if sig_power > 0:
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(n_out) + 1j * np.random.randn(n_out))
        wideband = wideband + noise

    peak = np.max(np.abs(wideband))
    if peak > 0:
        wideband = (wideband / peak) * 0.9
    I_out = np.clip(np.round(wideband.real * 32767), -32768, 32767).astype(np.int16)
    Q_out = np.clip(np.round(wideband.imag * 32767), -32768, 32767).astype(np.int16)
    out_data = np.empty(2 * n_out, dtype=np.int16)
    out_data[0::2] = I_out
    out_data[1::2] = Q_out
    out_data.tofile(out_path)
    return out_path