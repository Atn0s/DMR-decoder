import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# --- DMR L2 decode (ok-dmrlib 0.8.0) ---
try:
    from bitarray import bitarray
    from bitarray.util import ba2int
    from okdmr.dmrlib.etsi.fec.golay_20_8_7 import Golay2087
    from okdmr.dmrlib.etsi.fec.bptc_196_96 import BPTC19696
    from okdmr.dmrlib.etsi.fec.reed_solomon_12_9_4 import ReedSolomon1294
    from okdmr.dmrlib.etsi.layer2.pdu.full_link_control import FullLinkControl
    OKDMR_AVAILABLE = True
except ImportError:
    OKDMR_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True

# ================== Constants ==================
Fs_wide = 2500000.0        # 2.5 MHz wideband sampling rate
Fs_dec = 48000.0           # baseband rate (SPS = 10)
SPS = 10
UP_FACTOR = 12             # 2.5 MHz * 12 / 625 = 48 kHz
DOWN_FACTOR = 625
NCC_THRESHOLD = 0.68
DEV_NOMINAL = 1944.0       # Hz, outer-symbol (+3/-3) frequency deviation


def hex_to_symbols(hex_str):
    bin_str = "".join(f"{int(c, 16):04b}" for c in hex_str)
    symbols = []
    for i in range(0, len(bin_str), 2):
        dibit = bin_str[i:i + 2]
        if dibit == '01': symbols.append(3)
        elif dibit == '00': symbols.append(1)
        elif dibit == '10': symbols.append(-1)
        elif dibit == '11': symbols.append(-3)
    return np.array(symbols)


# symbol-level reference (for sync-aided calibration) and waveform (for NCC)
templates_sym = {
    "BS Sourced": hex_to_symbols("755FD7DF75F7"),
    "MS Sourced": hex_to_symbols("7F7D5DD57DFD"),
}
templates_wave = {k: np.repeat(v, SPS) for k, v in templates_sym.items()}

# DATA sync words (Voice LC Header is a DATA-type burst, uses these, not voice sync)
data_sync_sym = {
    "BS Sourced": hex_to_symbols("DFF57D75DF5D"),
    "MS Sourced": hex_to_symbols("D5D7F77FD757"),
}


def read_rawiq(filename):
    data = np.fromfile(filename, dtype=np.int16)
    I, Q = data[0::2], data[1::2]
    length = min(len(I), len(Q))
    return (I[:length] + 1j * Q[:length]) / 32768.0


def verify_periodicity(peaks, target_period_ms, tolerance_ms=15.0):
    if len(peaks) < 3:
        return False
    diffs = np.diff(peaks) / Fs_dec * 1000.0
    valid = 0
    for d in diffs:
        r = d % target_period_ms
        if r < tolerance_ms or r > (target_period_ms - tolerance_ms):
            valid += 1
    return (valid / len(diffs)) >= 0.6


def integrate_and_dump(x, sps):
    """Matched filter for staircase instantaneous-frequency output:
    running boxcar of length sps. Normalized so a flat plateau keeps its level."""
    return np.convolve(x, np.ones(sps) / sps, mode='same')


def gardner_timing_recovery(x, sps):
    """Gardner timing-error-detector with a 2nd-order loop. x is real (post-discriminator).
    Returns (symbols, positions): the interpolated symbol-center samples and the
    integer sample position each symbol was taken from (for burst gating)."""
    mu = 0.0              # fractional interpolation offset
    idx = sps             # integer sample pointer (start one symbol in)
    out = []
    pos_out = []
    Kp = 0.01             # loop gain (proportional)
    Ki = Kp / 50.0        # integral term
    integ = 0.0
    n = len(x)

    def interp(pos):
        # linear interpolation at fractional sample position
        i = int(np.floor(pos))
        f = pos - i
        if i < 0 or i + 1 >= n:
            return 0.0
        return x[i] * (1 - f) + x[i + 1] * f

    prev_sym = 0.0
    while idx < n - sps - 2:
        pos = idx + mu
        cur = interp(pos)                 # symbol center
        mid = interp(pos - sps / 2.0)     # halfway to previous symbol
        # Gardner TED (works without carrier phase, ideal for FSK discriminator output)
        err = mid * (cur - prev_sym)
        integ += Ki * err
        adj = Kp * err + integ
        idx += sps
        mu += adj
        # keep mu bounded, fold integer part into idx
        while mu >= 1.0:
            mu -= 1.0
            idx += 1
        while mu < 0.0:
            mu += 1.0
            idx -= 1
        out.append(cur)
        pos_out.append(int(round(pos)))   # sample index this symbol came from
        prev_sym = cur
    return np.array(out), np.array(pos_out)


def sync_aided_calibration(syms, ref_sym, peak_sym_idx):
    """Use the known sync sequence to estimate residual DC (CFO) and gain.
    Solve  ref = a*syms + b  over the sync region by least squares.
    Returns (gain, offset) so that (syms - offset_in_raw) * gain hits +-1/+-3."""
    L = len(ref_sym)
    s = peak_sym_idx - L // 2
    if s < 0 or s + L > len(syms):
        return None
    seg = syms[s:s + L]
    # least squares: ref ~= a*seg + b
    A = np.vstack([seg, np.ones(L)]).T
    a, b = np.linalg.lstsq(A, ref_sym, rcond=None)[0]
    return a, b


def gate_and_calibrate(syms, positions, ref_sym, ncc_peaks, burst_half_samples):
    """Keep only symbols inside detected DMR bursts, and calibrate gain/offset
    PER BURST using the sync word found near each NCC peak.
    Returns the calibrated symbols belonging to bursts only (noise/idle removed)."""
    L = len(ref_sym)
    kept = []
    gains = []
    for pk in ncc_peaks:
        lo = pk - burst_half_samples
        hi = pk + burst_half_samples
        # symbols whose source sample falls inside this burst window
        mask = (positions >= lo) & (positions <= hi)
        if not np.any(mask):
            continue
        burst_syms = syms[mask]
        # locate the sync sequence within this burst and calibrate locally
        if len(burst_syms) >= L:
            c = np.correlate(burst_syms, ref_sym, mode='same')
            local_pk = int(np.argmax(np.abs(c)))
            cal = sync_aided_calibration(burst_syms, ref_sym, local_pk)
            if cal is not None:
                a, b = cal
                gains.append(a)
                kept.append(a * burst_syms + b)
                continue
        kept.append(burst_syms)  # fallback: keep uncalibrated
    if not kept:
        return np.array([]), None
    return np.concatenate(kept), (np.median(gains) if gains else None)



# --- Voice LC Header verification: Reed-Solomon(12,9,4), NOT CRC-24 ---
# ETSI TS 102 361-1 B.3.6: the 24 bits after the 72-bit FLC are RS(12,9,4) parity,
# masked with the Voice-LC-Header data-type mask 0x969696 (B.3.12). ok-dmrlib ships
# ReedSolomon1294.check(data12, mask). (The earlier CRC-24 assumption was wrong; that
# is why crc24_selftest.py could never find matching params.)
VLC_RS_MASK = bytes([0x96, 0x96, 0x96])


def symbol_to_dibit(val):
    """4FSK hard decision -> 2 bits (ETSI Table). +3->01 +1->00 -1->10 -3->11."""
    if val > 2.0:
        return [0, 1]
    elif val > 0.0:
        return [0, 0]
    elif val > -2.0:
        return [1, 0]
    else:
        return [1, 1]


def adaptive_slice_bits(seg):
    """dsd-fme-style adaptive 4-level slicer. Per-burst max/min/center/umid/lmid
    decision thresholds (robust to residual gain/DC drift), -> 264 bits."""
    hi = np.percentile(seg, 90)
    lo = np.percentile(seg, 10)
    center = 0.5 * (hi + lo)
    umid = 0.5 * (hi + center)
    lmid = 0.5 * (lo + center)
    bits = []
    for v in seg:
        if v >= umid:
            bits.extend([0, 1])     # +3
        elif v >= center:
            bits.extend([0, 0])     # +1
        elif v >= lmid:
            bits.extend([1, 0])     # -1
        else:
            bits.extend([1, 1])     # -3
    return bitarray(bits)


def decode_lc_header_from_symbols(seg132):
    """Decode a Voice LC Header from 132 sync-calibrated burst symbols.
    Verifies Slot-Type via Golay(20,8,7) and the FLC via Reed-Solomon(12,9,4).
    No silent pass: RS failure is reported honestly."""
    if not OKDMR_AVAILABLE:
        return {"ok": False, "reason": "okdmr not installed"}
    if seg132 is None or len(seg132) < 132:
        return {"ok": False, "reason": "burst out of range"}

    ba = adaptive_slice_bits(seg132)
    # field slicing: Info(98)|Slot(10)|SYNC(48)|Slot(10)|Info(98)  (bit domain)
    slot_type = ba[98:108] + ba[156:166]
    info = ba[0:98] + ba[166:264]

    golay_ok = Golay2087.check(slot_type.copy())
    color_code = ba2int(slot_type[0:4])
    data_type = ba2int(slot_type[4:8])

    res = {"ok": False, "color_code": color_code, "data_type": data_type,
           "golay_ok": golay_ok}
    if not golay_ok:
        res["reason"] = "Slot Type Golay check failed (burst corrupt)"
        return res
    if data_type != 1:  # 0001 = Voice LC Header
        res["reason"] = f"not Voice LC Header (data_type={data_type:04b})"
        return res

    # BPTC(196,96): deinterleave + repair -> 96 bits = 72-bit FLC + 24-bit RS parity
    decoded = BPTC19696.deinterleave_data_bits(info, repair_if_necessary=True)
    data12 = decoded[0:96].tobytes()
    rs_ok = ReedSolomon1294.check(data12, VLC_RS_MASK)

    res["rs_ok"] = rs_ok
    if not rs_ok:
        res["reason"] = "Reed-Solomon(12,9,4) mismatch (frame corrupt)"
        return res

    flc = FullLinkControl.from_bits(decoded[0:96])
    dst = flc.group_address or flc.target_address
    res.update({
        "ok": True,
        "flco": int(flc.full_link_control_opcode.value),
        "flco_name": flc.full_link_control_opcode.name,
        "fid": int(flc.feature_set_id.value),
        "fid_name": flc.feature_set_id.name,
        "dst_id": dst,
        "src_id": flc.source_address,
    })
    return res


def _interp(arr, pos):
    """Linear interpolation of arr at fractional sample positions pos (array)."""
    i = np.floor(pos).astype(int)
    fr = pos - i
    i = np.clip(i, 0, len(arr) - 2)
    return arr[i] * (1 - fr) + arr[i + 1] * fr


def lc_front_end(iq_dec, cutoff=9500.0, ntaps=151):
    """Dedicated front-end for LC decode: residual-carrier removal + WIDER channel
    filter (9.5 kHz, vs 6.5 kHz which clips outer +-1944 Hz symbols) + FM discriminator,
    centered on the active-region median (robust to 4FSK spectral-peak bias)."""
    f, ps = signal.welch(iq_dec, fs=Fs_dec, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f); ps = np.fft.fftshift(ps)
    cf = f[np.argmax(ps)]
    n = np.arange(len(iq_dec))
    iqf = iq_dec * np.exp(-1j * 2 * np.pi * cf * n / Fs_dec)
    iqf = signal.filtfilt(signal.firwin(ntaps, cutoff, fs=Fs_dec), [1.0], iqf)
    yd = np.angle(iqf[1:] * np.conj(iqf[:-1]))
    amp = np.abs(iqf[:-1])
    active = amp > (np.median(amp) + 0.3 * (np.mean(amp) - np.median(amp)))
    center = np.median(yd[active]) if np.any(active) else np.median(yd)
    y = (yd - center) * (3.0 / (2.0 * np.pi * DEV_NOMINAL / Fs_dec))
    return y


def find_data_sync_positions(y, name, thr_ratio=0.55):
    """Sample-domain normalized cross-correlation to locate DATA-sync bursts.
    Returns [(sync_center_sample, polarity_sign)]."""
    ref = data_sync_sym[name]
    rwave = np.repeat(ref, SPS)
    c = signal.correlate(y, rwave, mode='same')
    e = np.convolve(y ** 2, np.ones(len(rwave)), mode='same')
    e = np.where(e <= 0, 1e-9, e)
    ncc = c / np.sqrt(e * np.sum(rwave ** 2))
    pos, _ = signal.find_peaks(np.abs(ncc), height=thr_ratio, distance=800)
    return [(int(p), float(np.sign(ncc[p]))) for p in pos]


def recover_burst_symbols(y, sync_center, sgn, name):
    """Recover 132 burst symbols. Burst layout (132 sym): Info(49) Slot(5) SYNC(24)
    Slot(5) Info(49); sync occupies symbols [54,78). Sweeps sub-symbol phase, picking
    the phase whose sync-region affine fit gives the cleanest 4-level constellation.
    Returns the calibrated 132-symbol array (or None)."""
    ref = data_sync_sym[name]
    levels = np.array([-3, -1, 1, 3])
    best = (1e18, None)
    for ph in np.linspace(-4, 4, 33):
        start = sync_center - (54 + 12) * SPS + ph
        pos = start + np.arange(132) * SPS
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue
        seg = sgn * _interp(y, pos)
        sy = seg[54:78]                       # 24 sync symbols, known +-3
        a, b = np.linalg.lstsq(np.vstack([sy, np.ones(24)]).T, ref, rcond=None)[0]
        segc = a * seg + b
        near = levels[np.argmin(np.abs(segc[:, None] - levels[None, :]), axis=1)]
        resid = np.mean((segc - near) ** 2)
        if resid < best[0]:
            best = (resid, segc)
    return best[1]


def process_candidate(iq, f_offset, idx):
    print(f"\nAnalyzing Candidate {idx+1} at Offset: {f_offset/1e3:+.2f} kHz...")

    # --- Stage 2: DDC + decimate + linear-phase channel filter ---
    t = np.arange(len(iq)) / Fs_wide
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * f_offset * t)
    iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)

    # linear-phase FIR (filtfilt) instead of Butterworth to avoid group-delay drift
    fir = signal.firwin(101, 6500.0, fs=Fs_dec)
    iq_filtered = signal.filtfilt(fir, [1.0], iq_dec)

    # --- Stage 3a: FM discriminator ---
    y_demod = np.angle(iq_filtered[1:] * np.conj(iq_filtered[:-1]))

    # gated DC cancellation (coarse CFO removal on active samples only)
    power = np.abs(iq_filtered) ** 2
    nf = np.median(power)
    ap = np.mean(power)
    thr = nf + 0.15 * (ap - nf)
    active = power > thr
    cfo_dc = np.mean(y_demod[active[:-1]]) if np.any(active[:-1]) else np.mean(y_demod)
    y_centered = y_demod - cfo_dc

    # coarse scaling to nominal +-3
    y_scaled = y_centered * (3.0 / (2.0 * np.pi * DEV_NOMINAL / Fs_dec))

    # --- Stage 3b: integrate-and-dump matched filter (NOT RRC) ---
    y_if = integrate_and_dump(y_scaled, SPS)
    y_clipped = np.clip(y_if, -5.0, 5.0)

    # --- Stage 4: multi-template NCC ---
    ncc_results, peaks_found = {}, {}
    for name, wave in templates_wave.items():
        wlen = len(wave)
        corr = signal.correlate(y_clipped, wave, mode='same')
        energy = np.convolve(y_clipped ** 2, np.ones(wlen), mode='same')
        energy = np.where(energy == 0, 1e-10, energy)
        ncc = corr / np.sqrt(energy * np.sum(wave ** 2))
        p_pos, _ = signal.find_peaks(ncc, height=NCC_THRESHOLD, distance=800)
        p_neg, _ = signal.find_peaks(-ncc, height=NCC_THRESHOLD, distance=800)
        ncc_results[name] = ncc
        peaks_found[name] = (p_pos, p_neg)

    # --- Stage 5: protocol periodicity verification ---
    best = {"name": None, "type": None, "peaks": []}
    for name, (p_pos, p_neg) in peaks_found.items():
        if len(p_pos) > 0 and verify_periodicity(p_pos, 360.0):
            best = {"name": name, "type": "VOICE", "peaks": p_pos}
            break
        if len(p_neg) > 0 and (verify_periodicity(p_neg, 30.0) or verify_periodicity(p_neg, 60.0)):
            best = {"name": name, "type": "DATA/CONTROL", "peaks": p_neg}
            break

    # --- Stage 6: Voice LC Header decode (DATA-sync burst at call start) ---
    # Uses a dedicated wider-filter front-end + sample-domain framing + per-burst
    # sync calibration + adaptive slicer + Reed-Solomon(12,9,4) verification.
    lc_result = None
    if best["name"] is not None and OKDMR_AVAILABLE:
        y_lc = lc_front_end(iq_dec)
        seen = set()
        for name in (best["name"], "BS Sourced", "MS Sourced"):
            for sc, sgn in find_data_sync_positions(y_lc, name):
                key = round(sc / 50)
                if key in seen:
                    continue
                seen.add(key)
                seg = recover_burst_symbols(y_lc, sc, sgn, name)
                r = decode_lc_header_from_symbols(seg)
                if r.get("ok"):
                    lc_result = r
                    break
                if lc_result is None:
                    lc_result = r       # keep last attempt for reporting
            if lc_result is not None and lc_result.get("ok"):
                break

    return iq_dec, y_clipped, ncc_results, best, lc_result



def plot_candidate(iq_dec, y_clipped, ncc_results, best, f_offset, idx):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    _title = "Candidate %d (LO %+.1f kHz)" % (idx + 1, f_offset / 1e3)
    fig.suptitle(_title, fontsize=14, fontweight='bold')

    # [0,0] baseband spectrum
    f_d, psd_d = signal.welch(iq_dec, fs=Fs_dec, nperseg=512, return_onesided=False)
    f_d = np.fft.fftshift(f_d)
    psd_d = np.fft.fftshift(psd_d)
    axs[0, 0].plot(f_d / 1e3, 10 * np.log10(psd_d), color='tab:blue')
    axs[0, 0].axvspan(-6.25, 6.25, color='cyan', alpha=0.15, label='DMR 12.5kHz')
    axs[0, 0].set_title("STAGE 2: Decimated Baseband Spectrum")
    axs[0, 0].set_xlabel("Baseband Freq (kHz)")
    axs[0, 0].set_ylabel("Power (dB)")
    axs[0, 0].grid(True)
    axs[0, 0].legend(loc="upper right")

    # [0,1] burst-gated, per-burst calibrated symbol histogram
    if best["name"] is not None:
        syms, positions = gardner_timing_recovery(y_clipped, SPS)
        ref = templates_sym[best["name"]]
        burst_half = int(0.5 * 264 * SPS)
        syms_cal, _ = gate_and_calibrate(syms, positions, ref, best["peaks"], burst_half)
        if len(syms_cal) > 0:
            axs[0, 1].hist(syms_cal, bins=120, color='tab:orange', alpha=0.85)
            for lv, col in [(3, 'r'), (1, 'orange'), (-1, 'orange'), (-3, 'r')]:
                axs[0, 1].axvline(lv, color=col, linestyle='--', alpha=0.6)
            axs[0, 1].set_title("STAGE 3: Burst-Gated Symbol Histogram")
            axs[0, 1].set_xlabel("Decided Level (target +-1 / +-3)")
            axs[0, 1].set_ylabel("Count")
            axs[0, 1].set_xlim([-5, 5])
        else:
            axs[0, 1].text(0.5, 0.5, "No burst symbols gated", ha='center',
                           va='center', transform=axs[0, 1].transAxes)
    else:
        axs[0, 1].plot(y_clipped, color='tab:orange', alpha=0.8)
        axs[0, 1].set_title("STAGE 3: Demod (No Match)")
        axs[0, 1].set_ylim([-5, 5])
    axs[0, 1].grid(True)

    # [1,0] NCC
    axs[1, 0].plot(ncc_results["BS Sourced"], label="BS", color='tab:green', alpha=0.8)
    axs[1, 0].plot(ncc_results["MS Sourced"], label="MS", color='tab:purple', alpha=0.8)
    axs[1, 0].axhline(NCC_THRESHOLD, color='red', linestyle='-.', label='Thr')
    axs[1, 0].set_title("STAGE 4: Multi-Template NCC")
    axs[1, 0].set_xlabel("Sample Index")
    axs[1, 0].set_ylabel("Score")
    axs[1, 0].grid(True)
    axs[1, 0].legend(loc="lower left")

    # [1,1] confirm / reject
    if best["name"] is not None:
        tp = best["peaks"][0]
        rng = range(max(0, tp - 120), min(len(y_clipped), tp + 120))
        axs[1, 1].plot(rng, y_clipped[rng], 'o-', color='tab:green', ms=3, label="Matched")
        for lv, col in [(3, 'r'), (1, 'orange'), (-1, 'orange'), (-3, 'r')]:
            axs[1, 1].axhline(lv, color=col, linestyle='--', alpha=0.6)
        axs[1, 1].set_title("STAGE 5: DMR CONFIRMED (%s) - %s" % (best['type'], best['name']))
        axs[1, 1].set_xlabel("Sample Index")
        axs[1, 1].set_ylabel("Level")
        axs[1, 1].legend(loc="lower left")
    else:
        axs[1, 1].set_facecolor('mistyrose')
        axs[1, 1].text(0.1, 0.5, "REJECTED:\nNon-DMR (P25 or Noise)", fontsize=12,
                       color='darkred', fontweight='bold', ha='left', va='center',
                       transform=axs[1, 1].transAxes)
        axs[1, 1].set_title("STAGE 5: REJECTED", color='red', fontweight='bold')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    target_file = "data/synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(target_file):
        print("Error: %s not found. Run synthesis.py first!" % target_file)
        return

    print("=" * 80)
    print("   DMR Pipeline v2 (Timing Recovery + I&D + Sync Calib + LC Decode)")
    print("=" * 80)
    if not OKDMR_AVAILABLE:
        print("  [note] ok-dmrlib not importable -- LC header decode will be skipped.")
    iq = read_rawiq(target_file)

    # Stage 1: coarse Welch scan
    f_w, psd_w = signal.welch(iq, fs=Fs_wide, nperseg=4096, return_onesided=False)
    f_w = np.fft.fftshift(f_w)
    psd_w = np.fft.fftshift(psd_w)
    psd_db = 10 * np.log10(psd_w)
    nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=nf + 15, distance=20)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f_w / 1e6, psd_db, color='teal', label='Wideband Spectrum')
    ax.axhline(nf + 15, color='red', linestyle='--', label='Threshold (NF+15dB)')
    if len(peaks) > 0:
        ax.plot(f_w[peaks] / 1e6, psd_db[peaks], "x", color='red', ms=10, label='Candidates')
    ax.set_title("STAGE 1: Coarse Spectrum Scan")
    ax.set_xlabel("Frequency Offset (MHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show(block=False)

    if len(peaks) == 0:
        print("  -> Silent. No active signals.")
        plt.show()
        return
    print("  -> Detected %d candidates." % len(peaks))

    for idx, pk in enumerate(peaks):
        f_offset = f_w[pk]
        iq_dec, y_clipped, ncc_results, best, lc = process_candidate(iq, f_offset, idx)
        if best["name"]:
            print("  -> [MATCH] DMR %s %s" % (best['name'], best['type']))
            if lc is not None:
                cc = lc.get('color_code')
                dt = lc.get('data_type')
                gok = lc.get('golay_ok')
                print("     Slot Type: CC=%s DataType=%s Golay_ok=%s" % (cc, dt, gok))
                if lc.get("ok"):
                    print("     [LC HEADER DECODED -- Reed-Solomon(12,9,4) OK]")
                    print("       Source ID      : %s" % lc['src_id'])
                    print("       Destination ID : %s" % lc['dst_id'])
                    print("       FID            : %s (0x%02X) %s" % (
                        lc['fid'], lc['fid'], lc.get('fid_name', '')))
                    print("       FLCO           : %s (%s)" % (lc['flco'], lc.get('flco_name', '')))
                else:
                    print("     [LC HEADER NOT DECODED] reason: %s" % lc.get('reason'))
            elif not OKDMR_AVAILABLE:
                print("     (okdmr not installed -- skipped LC decode)")
        else:
            print("  -> [REJECTED] Non-DMR (P25 or noise)")
        plot_candidate(iq_dec, y_clipped, ncc_results, best, f_offset, idx)

    print("\n" + "=" * 80)
    print("                     Search & Visualization Completed")
    print("=" * 80)


if __name__ == '__main__':
    main()
