"""dsd-fme 风格的 DMR 符号引擎（方案第 1-6 步）。
保留 dmr_pipeline_v2 的 DDC/抽取/鉴频思路，只重写 "鉴频输出 -> 符号 -> bits"。
验证指标：Golay(20,8,7) 通过率。"""
import numpy as np
import scipy.signal as signal
import dmr_pipeline_v2 as P

from bitarray import bitarray
from bitarray.util import ba2int
from okdmr.dmrlib.etsi.fec.golay_20_8_7 import Golay2087
from okdmr.dmrlib.etsi.fec.bptc_196_96 import BPTC19696

Fs = 48000.0
SPS = 10

# 132 符号突发布局: Info(49) Slot(5) SYNC(24) Slot(5) Info(49)
# 同步中心符号位于突发起点后 49+5+12 = 66 符号；同步占符号 [54,78)
SYNC_OFF = 54
BURST_LEN = 132


def front_end(iq_dec, cutoff=9500.0, ntaps=151):
    """残余载波去除 + 放宽信道滤波 + FM 鉴频 -> 每样点瞬时频率(标称符号刻度)。
    方案第 1、3-CFO 步：滤波放宽到 ~9.5kHz，center 用鉴频中位数(对 4FSK 比谱峰稳健)。"""
    f, ps = signal.welch(iq_dec, fs=Fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f); ps = np.fft.fftshift(ps)
    cf = f[np.argmax(ps)]
    n = np.arange(len(iq_dec))
    iq_dec = iq_dec * np.exp(-1j * 2 * np.pi * cf * n / Fs)

    fir = signal.firwin(ntaps, cutoff, fs=Fs)
    iqf = signal.filtfilt(fir, [1.0], iq_dec)

    yd = np.angle(iqf[1:] * np.conj(iqf[:-1]))
    amp = np.abs(iqf[:-1])
    active = amp > (np.median(amp) + 0.3 * (np.mean(amp) - np.median(amp)))
    center = np.median(yd[active]) if np.any(active) else np.median(yd)
    y = yd - center
    y = y * (3.0 / (2.0 * np.pi * P.DEV_NOMINAL / Fs))
    return y, amp, active, cf


def find_sync_positions(y, name, sps=SPS, thr_ratio=0.55):
    """样点域归一化互相关定位数据同步码。返回 [(中心样点, 极性)], ncc。"""
    ref = P.data_sync_sym[name]
    rwave = np.repeat(ref, sps)
    c = signal.correlate(y, rwave, mode='same')
    e = np.convolve(y ** 2, np.ones(len(rwave)), mode='same')
    e = np.where(e <= 0, 1e-9, e)
    ncc = c / np.sqrt(e * np.sum(rwave ** 2))
    pos, _ = signal.find_peaks(np.abs(ncc), height=thr_ratio, distance=800)
    return [(int(p), float(np.sign(ncc[p]))) for p in pos], ncc


def _interp(arr, pos):
    i = np.floor(pos).astype(int)
    fr = pos - i
    i = np.clip(i, 0, len(arr) - 2)
    return arr[i] * (1 - fr) + arr[i + 1] * fr


def recover_burst(y, sync_center, sgn, name, sps=SPS, phase_search=np.linspace(-4, 4, 33)):
    """取 132 符号；用同步码 24 符号(已知 ±3)为参考扫亚符号相位，选 4 电平残差最小者。
    方案第 3 步(轻量闭环) + 第 2 步(同步码重标定 center)。返回校准后的 132 符号数组或 None。"""
    ref = P.data_sync_sym[name]
    best = (1e18, None, None)
    for ph in phase_search:
        start = sync_center - (SYNC_OFF + 12) * sps + ph
        pos = start + np.arange(BURST_LEN) * sps
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue
        seg = sgn * _interp(y, pos)
        # 同步码区仿射标定 (gain, offset)
        sy = seg[SYNC_OFF:SYNC_OFF + 24]
        A = np.vstack([sy, np.ones(24)]).T
        a, b = np.linalg.lstsq(A, ref, rcond=None)[0]
        segc = a * seg + b
        levels = np.array([-3, -1, 1, 3])
        near = levels[np.argmin(np.abs(segc[:, None] - levels[None, :]), axis=1)]
        resid = np.mean((segc - near) ** 2)
        if resid < best[0]:
            best = (resid, segc, a)
    return best[1], best[0]


def adaptive_slice(segc):
    """方案第 4 步：自适应四电平判决器。用本突发符号分布估 max/min/center/umid/lmid，
    生成 264 bits。映射: +3->01 +1->00 -1->10 -3->11。"""
    hi = np.percentile(segc, 90)
    lo = np.percentile(segc, 10)
    center = 0.5 * (hi + lo)
    umid = 0.5 * (hi + center)
    lmid = 0.5 * (lo + center)
    bits = []
    for v in segc:
        if v >= umid:
            bits.extend([0, 1])     # +3
        elif v >= center:
            bits.extend([0, 0])     # +1
        elif v >= lmid:
            bits.extend([1, 0])     # -1
        else:
            bits.extend([1, 1])     # -3
    return bitarray(bits)


def decode_burst_bits(ba):
    """切片 Slot Type，Golay 校验，提取 CC/DataType；若是 Voice LC Header 再做 BPTC。"""
    slot = ba[98:108] + ba[156:166]
    golay_ok = Golay2087.check(slot.copy())
    cc = ba2int(slot[0:4])
    dt = ba2int(slot[4:8])
    res = {"golay_ok": golay_ok, "color_code": cc, "data_type": dt}
    if not golay_ok or dt != 1:
        return res
    info = ba[0:98] + ba[166:264]
    decoded = BPTC19696.deinterleave_data_bits(info, repair_if_necessary=True)
    lc = decoded[0:72]
    lcb = lc.tobytes()
    res.update({
        "is_vlc": True,
        "flco": lcb[0] & 0x3F,
        "fid": lcb[1],
        "dst_id": (lcb[3] << 16) | (lcb[4] << 8) | lcb[5],
        "src_id": (lcb[6] << 16) | (lcb[7] << 8) | lcb[8],
        "rx_crc": ba2int(decoded[72:96]),
    })
    return res


def process(iq_dec, names=('MS Sourced', 'BS Sourced'), verbose=False):
    """完整: 前端 -> 找同步 -> 逐突发恢复 -> 自适应判决 -> 解码。返回统计与解码列表。"""
    y, amp, active, cf = front_end(iq_dec)
    out = {"cf": cf, "golay_total": 0, "golay_ok": 0, "vlc": [], "bursts": 0}
    seen = set()
    for name in names:
        syncs, ncc = find_sync_positions(y, name)
        for sc, sgn in syncs:
            key = round(sc / 50)
            if key in seen:
                continue
            seen.add(key)
            segc, resid = recover_burst(y, sc, sgn, name)
            if segc is None:
                continue
            out["bursts"] += 1
            ba = adaptive_slice(segc)
            r = decode_burst_bits(ba)
            out["golay_total"] += 1
            if r["golay_ok"]:
                out["golay_ok"] += 1
            if r.get("is_vlc"):
                out["vlc"].append(r)
                if verbose:
                    print("  VLC @%d %s: flco=%d fid=%d src=%d dst=%d crc=0x%06X" % (
                        sc, name, r["flco"], r["fid"], r["src_id"], r["dst_id"], r["rx_crc"]))
    return out
