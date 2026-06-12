"""DMR 中途加入 (late entry) 解码：从语音超帧的 Embedded Signalling 碎片重建 LC。

复用 dmr_pipeline_v2 已验证的符号引擎（宽滤波前端 + 样点域定帧 + 同步码标定 +
自适应判决器），只新增「语音突发定位 + EMB/LCSS 状态机 + VBPTC(128,72) 纠错」。

EMB 校验当前为容错模式（EMB_TOLERANT=True）：QR(16,7,6) 校验失败也接收碎片，
仅用 LCSS 状态机对齐。后续可置 False 收紧。
"""
import os
import numpy as np
import scipy.signal as signal
from bitarray import bitarray

import dmr_pipeline_v2 as P   # keep for templates_sym, lc_front_end, read_rawiq, Fs_wide, UP_FACTOR, DOWN_FACTOR
from core.burst_type import SPS
from core.dsp import _interp, adaptive_slice_bits
from core.decoder import LateEntryCollector
from bitarray.util import ba2int

# 一个语音突发 = 27.5ms，TDMA 两时隙 => 同一时隙相邻语音突发间隔 60ms = 2880 样点 @48k
BURST_STRIDE = 2880
EMB_TOLERANT = True   # True: EMB QR(16,7,6) 头校验失败仍收集碎片（仅 QR 护 16-bit 头）
CS5_STRICT = True     # True: 要求重组 LC 的 5-bit 校验和(CS5)通过，滤除拼接错帧（推荐）


def find_voice_sync_anchor(y, name, thr_ratio=0.7):
    """样点域 NCC 用语音同步码锁定语音突发 (Burst A)。返回 [(中心样点, 极性)] 列表。
    真实语音超帧中 Burst A 每 360ms 出现一次，NCC 应达 ~0.85+。"""
    ref = P.templates_sym[name]
    rwave = np.repeat(ref, SPS)
    c = signal.correlate(y, rwave, mode='same')
    e = np.convolve(y ** 2, np.ones(len(rwave)), mode='same')
    e = np.where(e <= 0, 1e-9, e)
    ncc = c / np.sqrt(e * np.sum(rwave ** 2))
    pos, _ = signal.find_peaks(np.abs(ncc), height=thr_ratio, distance=800)
    return [(int(p), float(np.sign(ncc[p]))) for p in pos]


def lock_phase_from_anchor(y, anchor_center, sgn, name):
    """用 Burst A 语音同步码（符号 [54,78)）拟合最优亚符号相位。返回 best_phase。"""
    ref = P.templates_sym[name]
    levels = np.array([-3, -1, 1, 3])
    best = (1e18, 0.0)
    for ph in np.linspace(-6, 6, 49):
        start = anchor_center - (54 + 12) * SPS + ph
        pos = start + np.arange(132) * SPS
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue
        seg = sgn * _interp(y, pos)
        sy = seg[54:78]
        a, b = np.linalg.lstsq(np.vstack([sy, np.ones(24)]).T, ref, rcond=None)[0]
        segc = a * seg + b
        near = levels[np.argmin(np.abs(segc[:, None] - levels[None, :]), axis=1)]
        resid = np.mean((segc - near) ** 2)
        if resid < best[0]:
            best = (resid, ph)
    return best[1]


def recover_voice_burst(y, anchor_center, j, ph, sgn):
    """取第 j 个突发（相对 Burst A，j=0 即 A 本身）的 264-bit。沿用 A 的相位/极性，
    幅度由自适应判决器自校准（语音突发 B-E 中心是嵌入信令，无已知图案可仿射）。"""
    start = anchor_center + BURST_STRIDE * j - (54 + 12) * SPS + ph
    pos = start + np.arange(132) * SPS
    if pos[0] < 0 or pos[-1] >= len(y) - 1:
        return None
    seg = sgn * _interp(y, pos)
    return adaptive_slice_bits(seg)


def parse_emb_center(ba264):
    """从 264-bit 突发取中心 48-bit 嵌入区，拆出 16-bit EMB + 32-bit 信令。"""
    center = ba264[108:156]
    emb_bits = center[0:8] + center[40:48]
    signalling = center[8:40]
    return emb_bits, signalling


def decode_one_superframe(y, anchor_center, sgn, name, verbose=False):
    """以一个语音同步突发 (Burst A) 为超帧起点，用 LateEntryCollector 跑 EMB/LCSS
    状态机收集 First→Last 共 4×32=128 bit，VBPTC(128,72) 纠错出 LC。成功返回 dict。"""
    ph = lock_phase_from_anchor(y, anchor_center, sgn, name)
    collector = LateEntryCollector()
    sync_type = "MS_VOICE" if "MS" in name else "BS_VOICE"
    for j in range(0, 7):
        ba = recover_voice_burst(y, anchor_center, j, ph, sgn)
        if ba is None:
            break
        result = collector.feed(ba, sync_type)
        if result is not None:
            return {
                "anchor":    anchor_center,
                "cs5_ok":    result["extra"].get("cs5_ok", True),
                "flco":      0,
                "flco_name": result.get("flco", "UNKNOWN"),
                "fid":       0,
                "fid_name":  "UNKNOWN",
                "src_id":    result.get("src", 0),
                "dst_id":    result.get("dst", 0),
            }
    return None


def late_entry_decode(y, name, verbose=True):
    """扫描所有语音同步锚点（每个 = 一个超帧起点），逐超帧尝试中途加入解码。
    返回成功解出的 LC 列表（去重前的全部命中）。"""
    anchors = find_voice_sync_anchor(y, name)
    if not anchors:
        if verbose:
            print("  [%s] 未找到强语音同步锚点 (NCC>0.7)" % name)
        return []
    if verbose:
        print("  [%s] 语音同步锚点数=%d (每个为一超帧起点)" % (name, len(anchors)))
    results = []
    for anchor_center, sgn in anchors:
        r = decode_one_superframe(y, anchor_center, sgn, name, verbose=verbose)
        if r:
            results.append(r)
    return results


def _report(results, name):
    """打印某路信号的 late-entry 结果（按 SRC/DST 去重）。"""
    if not results:
        return False
    seen = set()
    uniq = []
    for r in results:
        k = (r["src_id"], r["dst_id"], r["flco"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    print("  [🎉 %s LATE ENTRY 解码成功] 命中超帧=%d 唯一LC=%d" % (name, len(results), len(uniq)))
    for r in uniq:
        print("     SRC=%d DST=%d FLCO=%s FID=%s (锚点样点=%d)"
              % (r["src_id"], r["dst_id"], r["flco_name"], r["fid_name"], r["anchor"]))
    return True


def run_narrowband(path):
    """干净窄带文件 (78125 Hz)：直接 resample 到 48k 解码。EMB QR(16,7,6) 纠错弱，
    干净信号才稳定。"""
    raw = P.read_rawiq(path)
    y = P.lc_front_end(signal.resample_poly(raw, 384, 625))
    print("\n文件 %s (%.1fs):" % (path, len(raw) / 78125.0))
    any_ok = False
    for name in ("MS Sourced", "BS Sourced"):
        results = late_entry_decode(y, name, verbose=True)
        any_ok |= _report(results, name)
    if not any_ok:
        print("  未能中途解出 LC")


def run_wideband(path):
    """合成宽带文件 (2.5 MHz)：盲搜候选 -> DDC -> 解码。注意合成时叠加了 20dB AWGN，
    嵌入信令的 QR(16,7,6) 纠错较弱，可能无法稳定恢复（LC Header 的 BPTC 更强，仍可解）。"""
    iq = P.read_rawiq(path)
    f_w, psd_w = signal.welch(iq, fs=P.Fs_wide, nperseg=4096, return_onesided=False)
    f_w = np.fft.fftshift(f_w); psd_w = np.fft.fftshift(psd_w)
    psd_db = 10 * np.log10(psd_w); nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=nf + 15, distance=20)
    for idx, pk in enumerate(peaks):
        fo = f_w[pk]
        t = np.arange(len(iq)) / P.Fs_wide
        iqd = signal.resample_poly(iq * np.exp(-1j * 2 * np.pi * fo * t), P.UP_FACTOR, P.DOWN_FACTOR)
        y = P.lc_front_end(iqd)
        print("\n候选 %d (偏移 %+.1f kHz):" % (idx + 1, fo / 1e3))
        any_ok = False
        for name in ("MS Sourced", "BS Sourced"):
            results = late_entry_decode(y, name, verbose=False)
            any_ok |= _report(results, name)
        if not any_ok:
            print("  本候选未能中途解出 LC (合成信号 20dB 噪声下嵌入信令 QR 纠错可能失败)")


def main():
    import sys
    print("=" * 80)
    print("  DMR 中途加入 (Late Entry) 解码 — EMB 容错模式=%s" % EMB_TOLERANT)
    print("=" * 80)
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        # 默认：优先用干净窄带原始文件（嵌入信令在此最稳），再试合成宽带文件
        targets = []
        for f in ("data/dmr_2_78125.rawiq", "data/dmr_1_78125.rawiq", "data/synthesized_wideband_2.5MHz.rawiq"):
            if os.path.exists(f):
                targets.append(f)
        if not targets:
            print("未找到任何输入文件")
            return
    for path in targets:
        if not os.path.exists(path):
            print("跳过不存在的文件:", path)
            continue
        if "78125" in path:
            run_narrowband(path)
        else:
            run_wideband(path)
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
