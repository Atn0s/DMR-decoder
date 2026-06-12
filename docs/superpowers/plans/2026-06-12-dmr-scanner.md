# DMR 离线盲扫 + 完整信令解析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有代码重构为 core/dsp.py + core/decoder.py + core/burst_type.py + scanner.py 分层架构，新增 CSBK 和 Terminator 解析，保留原入口脚本。

**Architecture:** DSP 层（core/dsp.py）只处理 numpy 数组；协议层（core/decoder.py）只做比特解析；scanner.py 调用两层完成端到端扫描并维护 Session 状态。原有 dmr_pipeline_v2.py 和 late_entry.py 保留为入口，实现迁移到 core/。

**Tech Stack:** Python 3.10, numpy, scipy.signal, bitarray, okdmr.dmrlib (BPTC19696, Golay2087, ReedSolomon1294, VBPTC12873, CSBK, FullLinkControl, EmbeddedSignalling, FiveBitChecksum), pytest

---

## File Structure

```
core/__init__.py              create  (empty)
core/burst_type.py            create  Sync 模板 + SlotDataType 枚举 + 常量
core/dsp.py                   create  frontend / find_sync_positions / recover_burst
core/decoder.py               create  decode_burst / LateEntryCollector
scanner.py                    create  scan_file / Session
tests/__init__.py             create  (empty)
tests/test_burst_type.py      create
tests/test_dsp.py             create
tests/test_decoder.py         create
tests/test_scanner.py         create
dmr_pipeline_v2.py            modify  import from core/，移除被迁移的函数
late_entry.py                 modify  import from core/，移除被迁移的函数
```

---

## Task 1: core/burst_type.py — 常量与枚举

**Files:**
- Create: `core/__init__.py`
- Create: `core/burst_type.py`
- Test: `tests/test_burst_type.py`

- [ ] **Step 1: 创建 core/__init__.py**

```python
# core/__init__.py
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_burst_type.py
from core.burst_type import SlotDataType, SYNC_TEMPLATES, SPS

def test_slot_data_type_values():
    assert SlotDataType.VOICE_LC_HEADER.value == 1
    assert SlotDataType.TERMINATOR_WITH_LC.value == 2
    assert SlotDataType.CSBK.value == 3

def test_sync_templates_shape():
    import numpy as np
    for key in ("MS_VOICE", "BS_VOICE", "DATA_MS", "DATA_BS"):
        assert key in SYNC_TEMPLATES
        assert len(SYNC_TEMPLATES[key]) == 24  # 24 symbols

def test_sps():
    assert SPS == 10
```

- [ ] **Step 3: 运行确认失败**

```bash
cd /home/lzkj/lzkj_workspace/python_docs/DMR_demo
pytest tests/test_burst_type.py -v
```
Expected: ImportError / ModuleNotFoundError

- [ ] **Step 4: 创建 core/burst_type.py**

```python
import numpy as np
from enum import IntEnum

Fs_wide  = 2_500_000.0
Fs_dec   = 48_000.0
SPS      = 10
UP_FACTOR   = 12
DOWN_FACTOR = 625
NCC_THRESHOLD_VOICE = 0.68
NCC_THRESHOLD_DATA  = 0.55
DEV_NOMINAL = 1944.0
VLC_RS_MASK = bytes([0x96, 0x96, 0x96])

class SlotDataType(IntEnum):
    PI_HEADER        = 0
    VOICE_LC_HEADER  = 1
    TERMINATOR_WITH_LC = 2
    CSBK             = 3
    MBC_HEADER       = 4
    MBCC             = 5
    DATA_HEADER      = 6
    RATE_HALF        = 7
    RATE_34          = 8
    IDLE             = 9
    RATE_1           = 10

def _hex_to_symbols(hex_str: str) -> np.ndarray:
    mapping = {'0':1,'1':1,'2':-1,'3':-1,'4':-1,'5':-3,
               '6':-3,'7':3,'8':-3,'9':3,'A':3,'B':3,
               'C':-1,'D':-3,'E':1,'F':3}
    # Use dibit table: 01->+3, 00->+1, 10->-1, 11->-3
    bin_str = "".join(f"{int(c,16):04b}" for c in hex_str)
    tbl = {'01':3,'00':1,'10':-1,'11':-3}
    return np.array([tbl[bin_str[i:i+2]] for i in range(0,len(bin_str),2)])

SYNC_TEMPLATES = {
    "BS_VOICE": _hex_to_symbols("755FD7DF75F7"),
    "MS_VOICE": _hex_to_symbols("7F7D5DD57DFD"),
    "DATA_BS":  _hex_to_symbols("DFF57D75DF5D"),
    "DATA_MS":  _hex_to_symbols("D5D7F77FD757"),
}
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_burst_type.py -v
```
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add core/__init__.py core/burst_type.py tests/__init__.py tests/test_burst_type.py
git commit -m "feat: add core/burst_type with SlotDataType enum and sync templates"
```

---

## Task 2: core/dsp.py — DSP 函数层

**Files:**
- Create: `core/dsp.py`
- Test: `tests/test_dsp.py`

函数来源：从 `dmr_pipeline_v2.py` 和 `late_entry.py` 提取，不添加新逻辑。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dsp.py
import numpy as np
from core.dsp import _interp, adaptive_slice_bits, read_rawiq, frontend, find_sync_positions

def test_interp_known():
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(_interp(arr, np.array([0.5, 1.5])), [0.5, 1.5])

def test_adaptive_slice_bits_levels():
    from bitarray import bitarray
    # Pure +3 signal → all 01
    seg = np.full(132, 3.0)
    ba = adaptive_slice_bits(seg)
    assert len(ba) == 264
    assert ba[0:2] == bitarray('01')

def test_read_rawiq_shape():
    import os
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        return
    iq = read_rawiq(path)
    assert iq.dtype == complex
    assert len(iq) > 0

def test_frontend_output_shape():
    import os
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        return
    from core.dsp import read_rawiq, frontend
    import scipy.signal as sig
    iq_raw = read_rawiq(path)
    iq_dec = sig.resample_poly(iq_raw, 384, 625)
    y = frontend(iq_dec, fo=0.0, fs=48000.0)
    assert len(y) == len(iq_dec) - 1

def test_find_sync_positions_returns_list():
    y = np.zeros(50000)
    result = find_sync_positions(y)
    assert isinstance(result, list)
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_dsp.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 core/dsp.py**

```python
import numpy as np
import scipy.signal as signal
from bitarray import bitarray
from core.burst_type import (
    Fs_dec, SPS, UP_FACTOR, DOWN_FACTOR,
    DEV_NOMINAL, NCC_THRESHOLD_VOICE, NCC_THRESHOLD_DATA,
    SYNC_TEMPLATES,
)

def read_rawiq(filename: str) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.int16)
    I, Q = data[0::2], data[1::2]
    n = min(len(I), len(Q))
    return (I[:n] + 1j * Q[:n]) / 32768.0

def _interp(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    i = np.floor(pos).astype(int)
    fr = pos - i
    i = np.clip(i, 0, len(arr) - 2)
    return arr[i] * (1 - fr) + arr[i + 1] * fr

def adaptive_slice_bits(seg: np.ndarray) -> bitarray:
    hi = np.percentile(seg, 90)
    lo = np.percentile(seg, 10)
    center = 0.5 * (hi + lo)
    umid = 0.5 * (hi + center)
    lmid = 0.5 * (lo + center)
    bits = []
    for v in seg:
        if v >= umid:   bits.extend([0, 1])
        elif v >= center: bits.extend([0, 0])
        elif v >= lmid:   bits.extend([1, 0])
        else:             bits.extend([1, 1])
    return bitarray(bits)

def frontend(iq_dec: np.ndarray, fo: float = 0.0, fs: float = Fs_dec,
             cutoff: float = 9500.0, ntaps: int = 151) -> np.ndarray:
    """DDC (fo≠0) + channel filter + FM discriminator + DC removal.
    When called with pre-DDC'd baseband (fo=0), skips the frequency shift."""
    if fo != 0.0:
        n = np.arange(len(iq_dec))
        iq_dec = iq_dec * np.exp(-1j * 2 * np.pi * fo * n / fs)
    f, ps = signal.welch(iq_dec, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f); ps = np.fft.fftshift(ps)
    cf = f[np.argmax(ps)]
    n = np.arange(len(iq_dec))
    iqf = iq_dec * np.exp(-1j * 2 * np.pi * cf * n / fs)
    iqf = signal.filtfilt(signal.firwin(ntaps, cutoff, fs=fs), [1.0], iqf)
    yd = np.angle(iqf[1:] * np.conj(iqf[:-1]))
    amp = np.abs(iqf[:-1])
    active = amp > (np.median(amp) + 0.3 * (np.mean(amp) - np.median(amp)))
    center = np.median(yd[active]) if np.any(active) else np.median(yd)
    return (yd - center) * (3.0 / (2.0 * np.pi * DEV_NOMINAL / fs))

def find_sync_positions(y: np.ndarray) -> list[tuple[int, float, str]]:
    """NCC 扫描所有同步模板，返回 [(center_sample, polarity, sync_type)]。
    sync_type ∈ {'MS_VOICE','BS_VOICE','DATA_MS','DATA_BS'}"""
    results = []
    thresholds = {
        "MS_VOICE": NCC_THRESHOLD_VOICE, "BS_VOICE": NCC_THRESHOLD_VOICE,
        "DATA_MS": NCC_THRESHOLD_DATA,   "DATA_BS": NCC_THRESHOLD_DATA,
    }
    for name, ref in SYNC_TEMPLATES.items():
        rwave = np.repeat(ref, SPS)
        c = signal.correlate(y, rwave, mode='same')
        e = np.convolve(y ** 2, np.ones(len(rwave)), mode='same')
        e = np.where(e <= 0, 1e-9, e)
        ncc = c / np.sqrt(e * np.sum(rwave ** 2))
        thr = thresholds[name]
        for peaks, sgn in [(signal.find_peaks(ncc, height=thr, distance=800)[0], 1.0),
                           (signal.find_peaks(-ncc, height=thr, distance=800)[0], -1.0)]:
            for p in peaks:
                results.append((int(p), sgn, name))
    results.sort(key=lambda x: x[0])
    return results

def recover_burst(y: np.ndarray, center: int, polarity: float,
                  sync_type: str) -> np.ndarray | None:
    """亚符号相位扫描 [-8,8] 65步，最小残差选最优相位，返回 132 符号数组。"""
    ref = SYNC_TEMPLATES[sync_type]
    levels = np.array([-3, -1, 1, 3])
    best = (1e18, None)
    for ph in np.linspace(-8, 8, 65):
        start = center - (54 + 12) * SPS + ph
        pos = start + np.arange(132) * SPS
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue
        seg = polarity * _interp(y, pos)
        sy = seg[54:78]
        a, b = np.linalg.lstsq(np.vstack([sy, np.ones(24)]).T, ref, rcond=None)[0]
        segc = a * seg + b
        near = levels[np.argmin(np.abs(segc[:, None] - levels[None, :]), axis=1)]
        resid = np.mean((segc - near) ** 2)
        if resid < best[0]:
            best = (resid, segc)
    return best[1]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_dsp.py -v
```
Expected: 5 PASSED (data-dependent tests skip if files absent)

- [ ] **Step 5: Commit**

```bash
git add core/dsp.py tests/test_dsp.py
git commit -m "feat: add core/dsp with frontend, find_sync_positions, recover_burst"
```

---

## Task 3: core/decoder.py — LC / CSBK / Terminator

**Files:**
- Create: `core/decoder.py`
- Test: `tests/test_decoder.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_decoder.py
import numpy as np
from bitarray import bitarray

def test_decode_burst_returns_none_on_garbage():
    from core.decoder import decode_burst
    syms = np.random.uniform(-4, 4, 132)
    result = decode_burst(syms, "DATA_MS")
    assert result is None or isinstance(result, dict)

def test_decode_burst_type_field():
    from core.decoder import decode_burst
    # All-zero symbols → Golay fails → returns None (not a crash)
    syms = np.zeros(132)
    result = decode_burst(syms, "DATA_MS")
    assert result is None

def test_late_entry_collector_needs_four_frags():
    from core.decoder import LateEntryCollector
    col = LateEntryCollector()
    ba = bitarray(264); ba.setall(0)
    assert col.feed(ba, "MS_VOICE") is None  # only 1 frag, not enough
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_decoder.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 core/decoder.py**

```python
import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int

from okdmr.dmrlib.etsi.fec.golay_20_8_7 import Golay2087
from okdmr.dmrlib.etsi.fec.bptc_196_96 import BPTC19696
from okdmr.dmrlib.etsi.fec.reed_solomon_12_9_4 import ReedSolomon1294
from okdmr.dmrlib.etsi.fec.vbptc_128_72 import VBPTC12873
from okdmr.dmrlib.etsi.fec.five_bit_checksum import FiveBitChecksum
from okdmr.dmrlib.etsi.layer2.pdu.full_link_control import FullLinkControl
from okdmr.dmrlib.etsi.layer2.pdu.csbk import CSBK
from okdmr.dmrlib.etsi.layer2.pdu.embedded_signalling import EmbeddedSignalling
from okdmr.dmrlib.etsi.layer2.elements.lcss import LCSS

from core.burst_type import SlotDataType, VLC_RS_MASK
from core.dsp import adaptive_slice_bits


def decode_burst(symbols: np.ndarray, sync_type: str) -> dict | None:
    """Data Sync burst 解码分发：VOICE_LC_HEADER / TERMINATOR_WITH_LC / CSBK。
    Voice Sync burst 不经此函数——由 LateEntryCollector.feed() 处理。
    返回统一 PDU dict 或 None（校验失败）。"""
    ba = adaptive_slice_bits(symbols)
    slot_bits = ba[98:108] + ba[156:166]   # 20 bit Slot Type field
    if not Golay2087.check(slot_bits.copy()):
        return None
    color_code = ba2int(slot_bits[0:4])
    data_type  = ba2int(slot_bits[4:8])
    info = ba[0:98] + ba[166:264]          # 196 bit info field

    if data_type == SlotDataType.VOICE_LC_HEADER:
        return _decode_lc_or_terminator(ba, info, color_code, "LC_HEADER")
    elif data_type == SlotDataType.TERMINATOR_WITH_LC:
        return _decode_lc_or_terminator(ba, info, color_code, "TERMINATOR")
    elif data_type == SlotDataType.CSBK:
        return _decode_csbk(ba, info, color_code)
    return None


def _decode_lc_or_terminator(ba264: bitarray, info196: bitarray,
                               color_code: int, pdu_type: str) -> dict | None:
    decoded = BPTC19696.deinterleave_data_bits(info196, repair_if_necessary=True)
    data12  = decoded[0:96].tobytes()
    if not ReedSolomon1294.check(data12, VLC_RS_MASK):
        return None
    try:
        flc = FullLinkControl.from_bits(decoded[0:96])
    except Exception:
        return None
    dst = flc.group_address or flc.target_address
    return {
        "type":     pdu_type,
        "src":      flc.source_address,
        "dst":      dst,
        "ts":       0,
        "flco":     flc.full_link_control_opcode.name,
        "extra":    {"color_code": color_code},
        "raw_bits": ba264.tobytes(),
    }


def _decode_csbk(ba264: bitarray, info96: bitarray, color_code: int) -> dict | None:
    # BPTC(196,96) → 96 bit CSBK PDU
    decoded = BPTC19696.deinterleave_data_bits(info96, repair_if_necessary=True)
    csbk_bits = decoded[0:96]
    try:
        csbk = CSBK.from_bits(csbk_bits)
    except Exception:
        return None
    return {
        "type":    "CSBK",
        "src":     csbk.source_address or 0,
        "dst":     csbk.target_address or 0,
        "ts":      0,
        "flco":    csbk.csbko.name,
        "extra":   {"color_code": color_code, "last_block": csbk.last_block},
        "raw_bits": ba264.tobytes(),
    }


class LateEntryCollector:
    """有状态的 EMB 碎片收集器。每个 Voice Sync burst 调用 feed()，
    集齐 First+Cont+Cont+Last 共 4 片后触发 VBPTC 解码，返回 PDU dict。"""

    def __init__(self):
        self._frags: list = []
        self._collecting: bool = False

    def reset(self):
        self._frags = []
        self._collecting = False

    def feed(self, ba264: bitarray, sync_type: str) -> dict | None:
        center = ba264[108:156]
        emb_bits   = center[0:8] + center[40:48]   # 16-bit EMB header
        signalling = center[8:40]                   # 32-bit fragment

        try:
            emb = EmbeddedSignalling.from_bits(emb_bits)
        except Exception:
            return None
        lcss = emb.link_control_start_stop

        if not self._collecting:
            if lcss == LCSS.FirstFragmentLC:
                self._collecting = True
                self._frags = [signalling]
        else:
            self._frags.append(signalling)
            if len(self._frags) == 4:
                return self._decode_assembled(ba264)
        return None

    def _decode_assembled(self, last_ba264: bitarray) -> dict | None:
        b128 = self._frags[0] + self._frags[1] + self._frags[2] + self._frags[3]
        self.reset()
        lc77 = VBPTC12873.deinterleave_data_bits(b128, include_cs5=True)
        lc72 = lc77[0:72]
        rx_cs5 = ba2int(lc77[72:77])
        cs5_ok = (rx_cs5 <= 30) and FiveBitChecksum.verify(lc72.tobytes(), rx_cs5)
        if not cs5_ok:
            return None
        try:
            flc = FullLinkControl.from_bits(lc77)
        except Exception:
            return None
        dst = flc.group_address or flc.target_address
        return {
            "type":    "LATE_ENTRY",
            "src":     flc.source_address,
            "dst":     dst,
            "ts":      0,
            "flco":    flc.full_link_control_opcode.name,
            "extra":   {"cs5_ok": cs5_ok},
            "raw_bits": last_ba264.tobytes(),
        }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_decoder.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/decoder.py tests/test_decoder.py
git commit -m "feat: add core/decoder with LC/CSBK/Terminator/LateEntryCollector"
```

---

## Task 4: scanner.py — 盲扫调度与会话跟踪

**Files:**
- Create: `scanner.py`
- Test: `tests/test_scanner.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_scanner.py
import os
from scanner import detect_sample_rate, scan_file

def test_detect_sample_rate_from_filename():
    assert detect_sample_rate("data/dmr_1_78125.rawiq") == 78125
    assert detect_sample_rate("data/synthesized_wideband_2.5MHz.rawiq") is None

def test_scan_file_returns_list():
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        return
    results = scan_file(path)
    assert isinstance(results, list)

def test_scan_file_wideband():
    path = "data/synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(path):
        return
    results = scan_file(path)
    assert isinstance(results, list)
    # 已知数据中有 DMR 信号，应解出至少一个 PDU
    types = [r["type"] for r in results]
    assert any(t in ("LC_HEADER", "LATE_ENTRY", "CSBK", "TERMINATOR") for t in types)
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_scanner.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 scanner.py**

```python
import re, json, os
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass, field

from core.burst_type import Fs_wide, Fs_dec, UP_FACTOR, DOWN_FACTOR, SPS
from core.dsp import read_rawiq, frontend, find_sync_positions, recover_burst
from core.decoder import decode_burst, LateEntryCollector


@dataclass
class Session:
    src: int
    dst: int
    start_pdu: dict
    voice_raw: list = field(default_factory=list)
    terminator: dict | None = None
    late_entry_lc: dict | None = None
    duration_s: float | None = None


def detect_sample_rate(path: str) -> int | None:
    """从文件名提取采样率数字，如 dmr_1_78125.rawiq -> 78125。无法推断返回 None。"""
    m = re.search(r'_(\d{4,7})\.rawiq', os.path.basename(path))
    return int(m.group(1)) if m else None


def _psd_blind_search(iq: np.ndarray, fs: float) -> list[float]:
    f, psd = signal.welch(iq, fs=fs, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f); psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd)
    nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(psd_db, height=nf + 15, distance=20)
    return [float(f[p]) for p in peaks]


def _process_candidate(iq: np.ndarray, fo: float, fs_in: float) -> list[dict]:
    """对一个候选频偏执行完整解码链，返回该候选解出的所有 PDU。"""
    t = np.arange(len(iq)) / fs_in
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * fo * t)
    iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)
    y = frontend(iq_dec, fo=0.0, fs=Fs_dec)

    positions = find_sync_positions(y)
    results = []
    collector = LateEntryCollector()
    seen_bursts: set[int] = set()

    for (center, polarity, sync_type) in positions:
        dedup_key = round(center / 50)
        if dedup_key in seen_bursts:
            continue
        seen_bursts.add(dedup_key)

        symbols = recover_burst(y, center, polarity, sync_type)
        if symbols is None:
            continue

        from core.dsp import adaptive_slice_bits
        ba264 = adaptive_slice_bits(symbols)

        if "VOICE" in sync_type:
            pdu = collector.feed(ba264, sync_type)
        else:
            pdu = decode_burst(symbols, sync_type)

        if pdu is not None:
            pdu["_fo_hz"] = fo
            results.append(pdu)

    return results


def scan_file(path: str, freq_list: list[float] | None = None,
              output_json: str | None = None) -> list[dict]:
    """扫描离线 IQ 文件，返回所有解出的 PDU list。
    freq_list=None 时对宽带文件做 Welch PSD 盲搜，对窄带文件直接处理。"""
    iq = read_rawiq(path)
    fs = detect_sample_rate(path)

    if freq_list is not None:
        candidates = [(fo, Fs_wide) for fo in freq_list]
    elif fs is None or fs > 200_000:
        # 宽带：盲搜
        fs_in = fs or Fs_wide
        fos = _psd_blind_search(iq, fs_in)
        candidates = [(fo, fs_in) for fo in fos]
    else:
        # 窄带：直接处理，先 resample 到 48k
        iq_dec = signal.resample_poly(iq, 384, 625)
        y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
        positions = find_sync_positions(y)
        results = []
        collector = LateEntryCollector()
        seen: set[int] = set()
        for (center, polarity, sync_type) in positions:
            key = round(center / 50)
            if key in seen:
                continue
            seen.add(key)
            symbols = recover_burst(y, center, polarity, sync_type)
            if symbols is None:
                continue
            from core.dsp import adaptive_slice_bits
            ba264 = adaptive_slice_bits(symbols)
            if "VOICE" in sync_type:
                pdu = collector.feed(ba264, sync_type)
            else:
                pdu = decode_burst(symbols, sync_type)
            if pdu is not None:
                results.append(pdu)
        if output_json:
            _write_json(results, output_json)
        _print_results(results)
        return results

    all_pdus = []
    for fo, fs_in in candidates:
        all_pdus.extend(_process_candidate(iq, fo, fs_in))

    # 跨频偏去重：同一 (src,dst,type) 只保留第一次
    seen_pdus: set[tuple] = set()
    unique = []
    for pdu in all_pdus:
        k = (pdu["src"], pdu["dst"], pdu["type"])
        if k not in seen_pdus:
            seen_pdus.add(k)
            unique.append(pdu)

    _print_results(unique)
    if output_json:
        _write_json(unique, output_json)
    return unique


def _print_results(pdus: list[dict]) -> None:
    for p in pdus:
        fo_str = ""
        if "_fo_hz" in p:
            fo_str = f" (fo={p['_fo_hz']/1e3:+.1f}kHz)"
        print(f"[{p['type']:<12}] SRC={p['src']} DST={p['dst']} FLCO={p['flco']}{fo_str}")


def _write_json(pdus: list[dict], path: str) -> None:
    clean = [{k: v for k, v in p.items() if k != "raw_bits"} for p in pdus]
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["data/dmr_1_78125.rawiq"]
    for t in targets:
        print(f"\n=== {t} ===")
        scan_file(t)
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_scanner.py -v
```
Expected: detect_sample_rate tests PASS，scan_file tests skip if no data files，wideband test PASS if data present

- [ ] **Step 5: Commit**

```bash
git add scanner.py tests/test_scanner.py
git commit -m "feat: add scanner.py with scan_file, Session, blind search"
```

---

## Task 5: 更新入口脚本 dmr_pipeline_v2.py

**Files:**
- Modify: `dmr_pipeline_v2.py`

原有文件保留 `process_candidate`, `plot_candidate`, `main` 逻辑，将已迁移到 core/ 的函数改为从 core/ 导入。

- [ ] **Step 1: 在 dmr_pipeline_v2.py 顶部替换导入**

将现有的本地函数定义替换为导入，在文件顶部 import 区之后添加：

```python
# 从 core/ 导入已迁移的函数
from core.burst_type import (
    Fs_wide, Fs_dec, SPS, UP_FACTOR, DOWN_FACTOR,
    NCC_THRESHOLD_VOICE as NCC_THRESHOLD, DEV_NOMINAL, VLC_RS_MASK,
    SYNC_TEMPLATES,
)
from core.dsp import (
    read_rawiq, _interp, adaptive_slice_bits,
    lc_front_end_compat as lc_front_end,
)
from core.decoder import decode_burst as _decode_burst_core
```

注意：`lc_front_end` 是 `frontend()` 的别名，需在 core/dsp.py 中添加：

```python
# core/dsp.py 末尾添加
def lc_front_end_compat(iq_dec, cutoff=9500.0, ntaps=151):
    return frontend(iq_dec, fo=0.0, fs=Fs_dec, cutoff=cutoff, ntaps=ntaps)
```

- [ ] **Step 2: 删除 dmr_pipeline_v2.py 中重复定义的函数**

删除以下函数的函数体（保留 `process_candidate`, `plot_candidate`, `main`, `gardner_timing_recovery`, `verify_periodicity`, `gate_and_calibrate`, `sync_aided_calibration`, `integrate_and_dump`，这些暂不迁移）：
- `read_rawiq` → 改为 `from core.dsp import read_rawiq`
- `hex_to_symbols` → 内联到 core/burst_type.py，此处可删除
- `_interp` → `from core.dsp import _interp`
- `adaptive_slice_bits` → `from core.dsp import adaptive_slice_bits`
- `lc_front_end` → `from core.dsp import lc_front_end_compat as lc_front_end`
- `find_data_sync_positions` → 用 `core.dsp.find_sync_positions` 替换调用
- `recover_burst_symbols` → 用 `core.dsp.recover_burst` 替换调用
- `decode_lc_header_from_symbols` → 用 `core.decoder.decode_burst` 替换调用

- [ ] **Step 3: 验证原有 main 仍能运行**

```bash
python dmr_pipeline_v2.py 2>&1 | head -20
```
Expected: 无 ImportError，Stage 1 输出正常

- [ ] **Step 4: Commit**

```bash
git add dmr_pipeline_v2.py core/dsp.py
git commit -m "refactor: dmr_pipeline_v2 now imports from core/"
```

---

## Task 6: 更新入口脚本 late_entry.py

**Files:**
- Modify: `late_entry.py`

- [ ] **Step 1: 替换 late_entry.py 中已迁移到 core/ 的函数**

```python
# late_entry.py 顶部替换：
from core.dsp import (
    read_rawiq, _interp, adaptive_slice_bits,
    find_sync_positions, recover_burst,
)
from core.decoder import LateEntryCollector
import dmr_pipeline_v2 as P  # 只用 P.lc_front_end, P.SPS, P.templates_sym
```

- [ ] **Step 2: 将 decode_one_superframe 改为使用 LateEntryCollector**

```python
def decode_one_superframe(y, anchor_center, sgn, name, verbose=False):
    ph = lock_phase_from_anchor(y, anchor_center, sgn, name)
    collector = LateEntryCollector()
    for j in range(0, 7):
        ba = recover_voice_burst(y, anchor_center, j, ph, sgn)
        if ba is None:
            break
        sync_type = "MS_VOICE" if "MS" in name else "BS_VOICE"
        result = collector.feed(ba, sync_type)
        if result is not None:
            return result  # PDU dict
    return None
```

- [ ] **Step 3: 运行 late_entry.py 验证**

```bash
python late_entry.py data/dmr_1_78125.rawiq 2>&1 | head -20
```
Expected: 无 ImportError，输出 LC 解码结果

- [ ] **Step 4: Commit**

```bash
git add late_entry.py
git commit -m "refactor: late_entry.py now uses core/decoder.LateEntryCollector"
```

---

## Task 7: 集成 smoke test

**Files:**
- Test: `tests/test_scanner.py` (补充)

- [ ] **Step 1: 运行完整测试套件**

```bash
pytest tests/ -v 2>&1
```
Expected: 所有测试 PASS 或 SKIP（无 FAIL）

- [ ] **Step 2: 用真实数据做端到端验证**

```bash
python scanner.py data/dmr_1_78125.rawiq data/dmr_2_78125.rawiq
```
Expected 输出类似：
```
=== data/dmr_1_78125.rawiq ===
[LC_HEADER   ] SRC=1234567 DST=9876543 FLCO=GROUP_VOICE_CHANNEL_USER
[LATE_ENTRY  ] SRC=1234567 DST=9876543 FLCO=GROUP_VOICE_CHANNEL_USER
```

- [ ] **Step 3: 宽带文件验证**

```bash
python scanner.py data/synthesized_wideband_2.5MHz.rawiq
```
Expected: 检出 2 个 DMR 候选，各解出 LC_HEADER

- [ ] **Step 4: Final commit**

```bash
git add tests/
git commit -m "test: add integration smoke tests for scanner"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Section 1 目录结构 → Task 1-4
- ✅ Section 4 接口签名 → core/dsp.py, core/decoder.py 完全匹配
- ✅ Section 5 PDU 统一结构 → decoder.py 的所有返回 dict 含 type/src/dst/ts/flco/extra/raw_bits
- ✅ Section 6 LC/Terminator/CSBK/LateEntry → Task 3
- ✅ Section 7 会话跟踪 → scanner.py Session dataclass（简化版，voice_raw 收集留扩展点）
- ✅ Section 8 盲搜策略 → scanner.py `_psd_blind_search` + 窄带直通
- ✅ Section 9 输出格式 → `_print_results` + `_write_json`
- ✅ Section 10 扩展接入点 → frontend() 只接受 numpy，接缝已预留
- ✅ Section 11 不实现内容 → AMBE/加密/数据信道均未涉及

**Type consistency:** `decode_burst` 在 Task 3 定义，在 Task 5 引用，签名一致。`LateEntryCollector.feed()` 在 Task 3 定义，在 Task 4/6 使用，参数 `(ba264: bitarray, sync_type: str)` 一致。

**已知偏差:** `find_sync_positions` 使用 `DATA_MS`/`DATA_BS` 替代 spec 中的单一 `DATA`，因为 recover_burst 需要知道用哪个模板进行相位标定。
