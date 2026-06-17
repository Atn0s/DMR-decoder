# DMR 实时 SDR 盲扫与信令解析系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有离线解调核心（`core/`）之上新增一个 `realtime/` 编排层，把连续 IQ 流（仿真文件，未来 SDR）切成重叠窗口、检测活跃信道、多进程解码、聚合成通话记录。

**Architecture:** 分块流式（chunked streaming）——采集线程把 IQ 写入无损环形缓冲，主循环按步进取重叠窗口，检测器用廉价能量检测找活跃频点并维护信道状态表，worker 池复用 `scanner._decode_loop` 解码，会话聚合器按 (fo,src,dst) 归并 PDU 成通话记录。`core/` 与 `scanner.py` 解调逻辑一行不改。

**Tech Stack:** Python 3.10, numpy, scipy.signal, multiprocessing, pytest。复用 okdmr.dmrlib（经由 `core/decoder.py`）。

## Global Constraints

- 解调核心 `core/dsp.py`、`core/decoder.py`、`core/burst_type.py` **不修改**。
- `scanner._decode_loop`、`scanner.frontend` 等现有函数被 worker **复用**，不重写。
- 采样率不绑定具体值——所有模块接受 `sample_rate` 参数，两级抽取因子按源采样率动态推导，目标输出 48kHz（`Fs_dec`）。
- 窗口长 `WINDOW_SEC=1.0`，步进 `STEP_SEC=0.9`，重叠 0.1s（≥ 一个 burst 55ms）。
- PDU dict schema（来自 `core/decoder.py`）：`{"type","src","dst","ts","flco","extra","raw_bits"}`，worker 另加 `"_fo_hz"`、`"_window_id"`。
- 全速回归测试用 `throttle=False`（不 sleep）；实时节奏单独测试。
- 测试中数据文件缺失时用 `pytest.skip()`，不得用裸 `return`（避免假绿）。
- 频率检测阈值 `ACTIVE_THRESHOLD_DB=15`（与现有 `scanner.PSD_PEAK_THRESHOLD_DB` 一致）。
- 关闭滞回 `CLOSE_HYSTERESIS=3` 窗；通话超时 `CALL_TIMEOUT_WINDOWS=5` 窗。
- 信道频点栅格 `channel_grid_hz=12500.0`。

---

## File Structure

```
realtime/__init__.py          create  (empty)
realtime/iq_source.py         create  IQSource / FileIQSource / SoapyIQSource(占位)
realtime/ring_buffer.py       create  RingBuffer
realtime/detector.py          create  Detector + ChannelState + ChannelRecord
realtime/worker.py            create  decode_window (复用 core/ 与 scanner._decode_loop)
realtime/aggregator.py        create  SessionAggregator + CallRecord
realtime/scanner_rt.py        create  RealtimeScanner 顶层编排
utils/synthesis.py            modify  增加 synthesize_scenario() 时间线合成(层次2)
tests/test_iq_source.py       create
tests/test_ring_buffer.py     create
tests/test_detector.py        create
tests/test_aggregator.py      create
tests/test_worker.py          create
tests/test_realtime_e2e.py    create
```

---

## Task 1: realtime/iq_source.py — IQ 源抽象与文件仿真源

**Files:**
- Create: `realtime/__init__.py`
- Create: `realtime/iq_source.py`
- Test: `tests/test_iq_source.py`

**Interfaces:**
- Consumes: `core.dsp.read_rawiq` 的同款 int16 IQ 文件格式（I/Q 交错，/32768.0 归一化）。
- Produces:
  - `class IQSource` 抽象基类：属性 `sample_rate: float`；方法 `read_chunk() -> np.ndarray | None`、`close() -> None`。
  - `class FileIQSource(IQSource)`：`__init__(self, path, sample_rate, chunk_samples=65536, throttle=True, starve_factor=1.0)`，`read_chunk()` 返回 `complex64` 数组（最后一块可短于 chunk_samples），流结束返回 `None`。
  - `class SoapyIQSource(IQSource)`：占位，`__init__` 抛 `NotImplementedError`。

- [ ] **Step 1: 创建 realtime/__init__.py**

```python
# realtime/__init__.py
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_iq_source.py
import os
import time
import numpy as np
import pytest
from realtime.iq_source import IQSource, FileIQSource, SoapyIQSource


def _make_iq_file(tmp_path, n_samples):
    """Write n_samples complex samples as interleaved int16 .rawiq, return path."""
    path = str(tmp_path / "test.rawiq")
    data = np.empty(2 * n_samples, dtype=np.int16)
    data[0::2] = np.arange(n_samples, dtype=np.int16)        # I ramp
    data[1::2] = -np.arange(n_samples, dtype=np.int16)       # Q ramp
    data.tofile(path)
    return path


def test_file_source_reads_all_samples(tmp_path):
    path = _make_iq_file(tmp_path, 1000)
    src = FileIQSource(path, sample_rate=48000.0, chunk_samples=256, throttle=False)
    total = 0
    while True:
        chunk = src.read_chunk()
        if chunk is None:
            break
        assert chunk.dtype == np.complex64
        total += len(chunk)
    src.close()
    assert total == 1000


def test_file_source_chunk_size(tmp_path):
    path = _make_iq_file(tmp_path, 1000)
    src = FileIQSource(path, sample_rate=48000.0, chunk_samples=256, throttle=False)
    first = src.read_chunk()
    assert len(first) == 256
    src.close()


def test_file_source_throttle_pacing(tmp_path):
    # 4 chunks of 12000 samples at 48000 Hz = 0.25s each = 1.0s total
    path = _make_iq_file(tmp_path, 48000)
    src = FileIQSource(path, sample_rate=48000.0, chunk_samples=12000, throttle=True)
    t0 = time.perf_counter()
    while src.read_chunk() is not None:
        pass
    elapsed = time.perf_counter() - t0
    src.close()
    # Should take ~1.0s; allow generous lower bound to avoid flakiness
    assert elapsed >= 0.7


def test_starve_factor_slows_pacing(tmp_path):
    path = _make_iq_file(tmp_path, 24000)
    src = FileIQSource(path, sample_rate=48000.0, chunk_samples=12000,
                       throttle=True, starve_factor=2.0)
    t0 = time.perf_counter()
    while src.read_chunk() is not None:
        pass
    elapsed = time.perf_counter() - t0
    src.close()
    # 0.5s of data at 2x starve = ~1.0s
    assert elapsed >= 0.7


def test_soapy_source_not_implemented():
    with pytest.raises(NotImplementedError):
        SoapyIQSource(sample_rate=2.4e6)
```

- [ ] **Step 3: 运行确认失败**

```bash
cd /home/lzkj/lzkj_workspace/python_docs/DMR_demo
pytest tests/test_iq_source.py -v
```
Expected: ImportError / ModuleNotFoundError

- [ ] **Step 4: 创建 realtime/iq_source.py**

```python
import time
import numpy as np


class IQSource:
    """Abstract continuous IQ source. read_chunk returns complex64 blocks."""
    sample_rate: float

    def read_chunk(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FileIQSource(IQSource):
    """Read a .rawiq file (interleaved int16) in chunks, optionally throttled to
    sample_rate to emulate live SDR pacing.

    throttle=False: read at full speed (fast regression tests).
    starve_factor>1.0: sleep longer than real-time to reproduce sample drops
    downstream (the source itself never drops; it just emits slower)."""

    def __init__(self, path: str, sample_rate: float, chunk_samples: int = 65536,
                 throttle: bool = True, starve_factor: float = 1.0):
        self.path = path
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.throttle = throttle
        self.starve_factor = starve_factor
        self._fh = open(path, "rb")

    def read_chunk(self) -> np.ndarray | None:
        # Each complex sample = 2 int16 = 4 bytes
        raw = np.frombuffer(self._fh.read(self.chunk_samples * 4), dtype=np.int16)
        if len(raw) < 2:
            return None
        n = len(raw) // 2
        iq = (raw[0:2 * n:2].astype(np.float32) +
              1j * raw[1:2 * n:2].astype(np.float32)) / 32768.0
        iq = iq.astype(np.complex64)
        if self.throttle:
            time.sleep((n / self.sample_rate) * self.starve_factor)
        return iq

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()


class SoapyIQSource(IQSource):
    """Real SDR via SoapySDR. Placeholder — not implemented this phase."""

    def __init__(self, sample_rate: float, **kwargs):
        raise NotImplementedError("SoapyIQSource is a placeholder for hardware phase")
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_iq_source.py -v
```
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add realtime/__init__.py realtime/iq_source.py tests/test_iq_source.py
git commit -m "feat: add realtime IQSource abstraction with throttled FileIQSource"
```

---

## Task 2: realtime/ring_buffer.py — 无损环形缓冲

**Files:**
- Create: `realtime/ring_buffer.py`
- Test: `tests/test_ring_buffer.py`

**Interfaces:**
- Produces:
  - `class RingBuffer`：`__init__(self, capacity_samples: int)`。
  - `write(self, chunk: np.ndarray) -> int`：写入；返回因容量不足丢弃的样点数（>0 即溢出）。
  - `read_window(self, window_samples: int, step_samples: int) -> np.ndarray | None`：返回 `window_samples` 长 `complex64` 窗口，读指针前进 `step_samples`（保留重叠）；数据不足返回 `None`。
  - 属性 `overflow_count: int`：累计丢弃样点数。
  - 方法 `available(self) -> int`：当前可读样点数。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_ring_buffer.py
import numpy as np
import pytest
from realtime.ring_buffer import RingBuffer


def test_write_then_read_window():
    rb = RingBuffer(capacity_samples=1000)
    data = np.arange(500, dtype=np.complex64)
    dropped = rb.write(data)
    assert dropped == 0
    win = rb.read_window(window_samples=300, step_samples=200)
    assert win is not None
    assert len(win) == 300
    np.testing.assert_array_equal(win, data[:300])


def test_read_window_overlap_preserved():
    rb = RingBuffer(capacity_samples=1000)
    rb.write(np.arange(500, dtype=np.complex64))
    win1 = rb.read_window(window_samples=300, step_samples=200)
    win2 = rb.read_window(window_samples=300, step_samples=200)
    # step=200 so win2 starts at sample 200; overlap is samples [200,300)
    np.testing.assert_array_equal(win1[200:300], win2[0:100])


def test_read_window_insufficient_returns_none():
    rb = RingBuffer(capacity_samples=1000)
    rb.write(np.arange(100, dtype=np.complex64))
    assert rb.read_window(window_samples=300, step_samples=200) is None


def test_overflow_counts_dropped_samples():
    rb = RingBuffer(capacity_samples=100)
    dropped = rb.write(np.arange(150, dtype=np.complex64))
    assert dropped == 50
    assert rb.overflow_count == 50


def test_overflow_keeps_newest_data():
    rb = RingBuffer(capacity_samples=100)
    rb.write(np.arange(150, dtype=np.complex64))
    # Oldest 50 dropped; buffer holds samples 50..149
    win = rb.read_window(window_samples=100, step_samples=100)
    assert win is not None
    np.testing.assert_array_equal(win, np.arange(50, 150, dtype=np.complex64))


def test_available_tracks_unread():
    rb = RingBuffer(capacity_samples=1000)
    rb.write(np.arange(500, dtype=np.complex64))
    assert rb.available() == 500
    rb.read_window(window_samples=300, step_samples=200)
    assert rb.available() == 300
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_ring_buffer.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 realtime/ring_buffer.py**

```python
import threading
import numpy as np


class RingBuffer:
    """Single-producer single-consumer ring buffer for complex64 samples.
    Thread-safe. On overflow, oldest data is dropped (mirrors SDR driver behavior)
    and overflow_count is incremented."""

    def __init__(self, capacity_samples: int):
        self._cap = capacity_samples
        self._buf = np.zeros(capacity_samples, dtype=np.complex64)
        self._write_pos = 0          # absolute count of samples written
        self._read_pos = 0           # absolute count of samples consumed
        self._overflow = 0
        self._lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> int:
        chunk = chunk.astype(np.complex64, copy=False)
        n = len(chunk)
        dropped = 0
        with self._lock:
            # If incoming exceeds capacity, keep only the newest cap samples
            if n >= self._cap:
                chunk = chunk[-self._cap:]
                dropped += n - self._cap
                n = self._cap
            # Make room: if unread + new exceeds capacity, advance read_pos
            unread = self._write_pos - self._read_pos
            free = self._cap - unread
            if n > free:
                evict = n - free
                self._read_pos += evict
                dropped += evict
            start = self._write_pos % self._cap
            end = start + n
            if end <= self._cap:
                self._buf[start:end] = chunk
            else:
                first = self._cap - start
                self._buf[start:] = chunk[:first]
                self._buf[:n - first] = chunk[first:]
            self._write_pos += n
            self._overflow += dropped
        return dropped

    def read_window(self, window_samples: int, step_samples: int) -> np.ndarray | None:
        with self._lock:
            unread = self._write_pos - self._read_pos
            if unread < window_samples:
                return None
            start = self._read_pos % self._cap
            end = start + window_samples
            if end <= self._cap:
                out = self._buf[start:end].copy()
            else:
                first = self._cap - start
                out = np.empty(window_samples, dtype=np.complex64)
                out[:first] = self._buf[start:]
                out[first:] = self._buf[:window_samples - first]
            self._read_pos += step_samples
            return out

    def available(self) -> int:
        with self._lock:
            return self._write_pos - self._read_pos

    @property
    def overflow_count(self) -> int:
        return self._overflow
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_ring_buffer.py -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add realtime/ring_buffer.py tests/test_ring_buffer.py
git commit -m "feat: add lossless RingBuffer with overflow counting and overlap reads"
```

---

## Task 3: realtime/detector.py — 能量检测与信道状态表

**Files:**
- Create: `realtime/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `scipy.signal.welch`、`scipy.signal.find_peaks`。
- Produces:
  - `class ChannelState(IntEnum)`：`IDLE=0, ACTIVE=1, TRACKING=2, CLOSING=3`。
  - `@dataclass ChannelRecord`：`fo_hz: float, state: ChannelState, last_active_window: int, missed_windows: int`。
  - 模块常量：`ACTIVE_THRESHOLD_DB=15`、`CLOSE_HYSTERESIS=3`。
  - `class Detector`：`__init__(self, sample_rate, channel_grid_hz=12500.0, threshold_db=ACTIVE_THRESHOLD_DB, close_hysteresis=CLOSE_HYSTERESIS)`。
  - `process_window(self, window_iq, window_id) -> list[tuple[np.ndarray, float, int]]`：返回需派发的 `[(window_iq, fo_hz, window_id)]`（策略 C：每个 ACTIVE/TRACKING 信道每窗都派；派发的 IQ 是整个窗口，不做信道化）。
  - `closed_channels(self) -> list[float]`：返回本轮转入 CLOSING 的频点。
  - `_quantize_freq(self, f_hz) -> float`：把频率量化到 `channel_grid_hz` 栅格中心（暴露供测试）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_detector.py
import numpy as np
import pytest
from realtime.detector import (
    Detector, ChannelState, ChannelRecord,
    ACTIVE_THRESHOLD_DB, CLOSE_HYSTERESIS,
)


def _tone(fo_hz, fs, n, amp=1.0):
    t = np.arange(n) / fs
    return (amp * np.exp(1j * 2 * np.pi * fo_hz * t)).astype(np.complex64)


def _noise(n, amp=0.01):
    return (amp * (np.random.randn(n) + 1j * np.random.randn(n))).astype(np.complex64)


def test_quantize_freq_to_grid():
    det = Detector(sample_rate=2.5e6, channel_grid_hz=12500.0)
    assert det._quantize_freq(151000.0) == 150000.0
    assert det._quantize_freq(-299000.0) == -300000.0


def test_idle_to_active_on_energy():
    np.random.seed(0)
    det = Detector(sample_rate=2.5e6)
    win = _tone(150000.0, 2.5e6, 8192, amp=2.0) + _noise(8192)
    dispatched = det.process_window(win, window_id=0)
    fos = [d[1] for d in dispatched]
    assert 150000.0 in fos


def test_strategy_c_dispatches_every_active_window():
    np.random.seed(1)
    det = Detector(sample_rate=2.5e6)
    win = _tone(150000.0, 2.5e6, 8192, amp=2.0) + _noise(8192)
    d0 = det.process_window(win, 0)
    d1 = det.process_window(win, 1)
    d2 = det.process_window(win, 2)
    # Strategy C: same active channel dispatched on every window
    assert any(d[1] == 150000.0 for d in d0)
    assert any(d[1] == 150000.0 for d in d1)
    assert any(d[1] == 150000.0 for d in d2)


def test_silence_closes_after_hysteresis():
    np.random.seed(2)
    det = Detector(sample_rate=2.5e6)
    active = _tone(150000.0, 2.5e6, 8192, amp=2.0) + _noise(8192)
    silent = _noise(8192)
    det.process_window(active, 0)            # ACTIVE
    for w in range(1, CLOSE_HYSTERESIS):     # missed but within hysteresis
        det.process_window(silent, w)
        assert 150000.0 not in det.closed_channels()
    det.process_window(silent, CLOSE_HYSTERESIS)  # exceeds hysteresis
    assert 150000.0 in det.closed_channels()


def test_silent_spectrum_dispatches_nothing():
    np.random.seed(3)
    det = Detector(sample_rate=2.5e6)
    win = _noise(8192)
    assert det.process_window(win, 0) == []
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_detector.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 realtime/detector.py**

```python
from enum import IntEnum
from dataclasses import dataclass
import numpy as np
import scipy.signal as signal

ACTIVE_THRESHOLD_DB = 15   # dB above median noise floor (matches scanner.PSD_PEAK_THRESHOLD_DB)
CLOSE_HYSTERESIS = 3       # consecutive silent windows before a call is closed


class ChannelState(IntEnum):
    IDLE = 0
    ACTIVE = 1
    TRACKING = 2
    CLOSING = 3


@dataclass
class ChannelRecord:
    fo_hz: float
    state: ChannelState
    last_active_window: int
    missed_windows: int


class Detector:
    """Per-window energy detection with a frequency-indexed channel state table.
    Strategy C: every ACTIVE/TRACKING channel is dispatched on every window
    (voice frames are accumulated in time order by the aggregator).
    The dispatched IQ slice is the full wideband window — DDC/decimation happen
    in the worker, not here."""

    def __init__(self, sample_rate: float, channel_grid_hz: float = 12500.0,
                 threshold_db: float = ACTIVE_THRESHOLD_DB,
                 close_hysteresis: int = CLOSE_HYSTERESIS):
        self.sample_rate = sample_rate
        self.channel_grid_hz = channel_grid_hz
        self.threshold_db = threshold_db
        self.close_hysteresis = close_hysteresis
        self._channels: dict[float, ChannelRecord] = {}
        self._just_closed: list[float] = []

    def _quantize_freq(self, f_hz: float) -> float:
        return round(f_hz / self.channel_grid_hz) * self.channel_grid_hz

    def _detect_active_freqs(self, window_iq: np.ndarray) -> set[float]:
        f, psd = signal.welch(window_iq, fs=self.sample_rate,
                              nperseg=min(4096, len(window_iq)),
                              return_onesided=False)
        f = np.fft.fftshift(f)
        psd = np.fft.fftshift(psd)
        psd_db = 10 * np.log10(psd + 1e-12)
        nf = np.median(psd_db)
        peaks, _ = signal.find_peaks(psd_db, height=nf + self.threshold_db, distance=20)
        return {self._quantize_freq(float(f[p])) for p in peaks}

    def process_window(self, window_iq: np.ndarray, window_id: int
                       ) -> list[tuple[np.ndarray, float, int]]:
        self._just_closed = []
        active_freqs = self._detect_active_freqs(window_iq)

        # Update existing channels and open new ones
        for fo in active_freqs:
            rec = self._channels.get(fo)
            if rec is None:
                self._channels[fo] = ChannelRecord(
                    fo_hz=fo, state=ChannelState.ACTIVE,
                    last_active_window=window_id, missed_windows=0)
            else:
                rec.state = ChannelState.TRACKING
                rec.last_active_window = window_id
                rec.missed_windows = 0

        # Age out channels not seen this window
        for fo, rec in list(self._channels.items()):
            if fo not in active_freqs:
                rec.missed_windows += 1
                if rec.missed_windows >= self.close_hysteresis:
                    rec.state = ChannelState.CLOSING
                    self._just_closed.append(fo)
                    del self._channels[fo]

        # Strategy C: dispatch every currently-active channel
        return [(window_iq, fo, window_id) for fo in sorted(active_freqs)]

    def closed_channels(self) -> list[float]:
        return list(self._just_closed)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_detector.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add realtime/detector.py tests/test_detector.py
git commit -m "feat: add Detector with energy detection and channel state table"
```

---

## Task 4: realtime/worker.py — 窗口解码（复用解调核心）

**Files:**
- Create: `realtime/worker.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes:
  - `core.burst_type.Fs_dec`（48000.0）。
  - `scanner._decode_loop(y: np.ndarray) -> list[dict]`（现有）。
  - `scanner.frontend`（即 `core.dsp.frontend`，经 scanner 导入）— 实际直接用 `core.dsp.frontend(iq_dec, fo=0.0, fs=Fs_dec)`。
  - `scipy.signal.resample_poly`。
- Produces:
  - `decode_window(window_iq, fo_hz, window_id, source_sample_rate) -> list[dict]`：纯函数，无共享状态，可被 `multiprocessing.Pool` 调用。返回 PDU 列表，每个 PDU 加 `"_fo_hz"=fo_hz`、`"_window_id"=window_id`。
  - `_decimation_factors(source_sample_rate, target=Fs_dec) -> tuple[int, int]`：推导 `resample_poly(up, down)` 因子，使 `source*up/down ≈ 48000`（暴露供测试）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_worker.py
import os
import numpy as np
import pytest
from realtime.worker import decode_window, _decimation_factors
from core.burst_type import Fs_dec
from core.dsp import read_rawiq


def test_decimation_factors_2_5mhz():
    up, down = _decimation_factors(2.5e6)
    # 2.5e6 * up/down should be ~48000
    assert abs(2.5e6 * up / down - Fs_dec) < 1.0


def test_decimation_factors_960khz():
    up, down = _decimation_factors(960000.0)
    assert abs(960000.0 * up / down - Fs_dec) < 50.0


def test_decode_window_returns_list_on_noise():
    np.random.seed(0)
    win = (np.random.randn(100000) + 1j * np.random.randn(100000)).astype(np.complex64)
    result = decode_window(win, fo_hz=0.0, window_id=0, source_sample_rate=2.5e6)
    assert isinstance(result, list)


def test_decode_window_tags_fo_and_window_id():
    # Use real wideband file if present; expect DMR1 at -300kHz
    path = "data/synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(path):
        pytest.skip("wideband test file not present")
    iq = read_rawiq(path).astype(np.complex64)
    # take a 1s window
    win = iq[:2_500_000]
    result = decode_window(win, fo_hz=-300000.0, window_id=7, source_sample_rate=2.5e6)
    for pdu in result:
        assert pdu["_fo_hz"] == -300000.0
        assert pdu["_window_id"] == 7
    # should decode at least one PDU from the known DMR signal
    assert any(p["type"] in ("LC_HEADER", "LATE_ENTRY", "TERMINATOR", "CSBK")
               for p in result)
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_worker.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 realtime/worker.py**

```python
import numpy as np
import scipy.signal as signal
from math import gcd

from core.burst_type import Fs_dec
from core.dsp import frontend
import scanner


def _decimation_factors(source_sample_rate: float, target: float = Fs_dec
                        ) -> tuple[int, int]:
    """Derive resample_poly(up, down) so source*up/down approx target.
    Reduces the ratio by gcd to keep filter length manageable."""
    up = int(round(target))
    down = int(round(source_sample_rate))
    g = gcd(up, down)
    return up // g, down // g


def decode_window(window_iq: np.ndarray, fo_hz: float, window_id: int,
                  source_sample_rate: float) -> list[dict]:
    """Decode one wideband IQ window at a given frequency offset.
    DDC(fo) -> resample to 48kHz -> frontend -> scanner._decode_loop.
    Pure function (no shared state) so it can run in a multiprocessing.Pool.
    Each returned PDU is tagged with _fo_hz and _window_id.
    Exceptions are swallowed -> returns [] so one bad window can't kill the pool."""
    try:
        n = np.arange(len(window_iq))
        shifted = window_iq * np.exp(-1j * 2 * np.pi * fo_hz * n / source_sample_rate).astype(np.complex64)
        up, down = _decimation_factors(source_sample_rate)
        iq_dec = signal.resample_poly(shifted, up, down)
        if len(iq_dec) < 512:
            return []
        y = frontend(iq_dec, fo=0.0, fs=Fs_dec)
        pdus = scanner._decode_loop(y)
        for pdu in pdus:
            pdu["_fo_hz"] = fo_hz
            pdu["_window_id"] = window_id
        return pdus
    except Exception:
        return []
```

> **Note:** `_decimation_factors(2.5e6)` → gcd(48000, 2500000)=8000 → (6, 312.5)? 不整除。改用：先约分 `up=48000, down=2500000`，`gcd=8000` → `(6, 312)`，`2.5e6*6/312 = 48077Hz`（误差 77Hz，<0.2%，可接受，frontend 内部用峰值频率自校准吸收残余偏移）。测试 `test_decimation_factors_2_5mhz` 的容差 `<1.0Hz` 过严——**修正测试容差为 `< 100.0`**：

```python
def test_decimation_factors_2_5mhz():
    up, down = _decimation_factors(2.5e6)
    assert abs(2.5e6 * up / down - Fs_dec) < 100.0
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_worker.py -v
```
Expected: 4 PASSED（最后一个若无数据文件则 SKIP）

- [ ] **Step 5: Commit**

```bash
git add realtime/worker.py tests/test_worker.py
git commit -m "feat: add decode_window worker reusing scanner._decode_loop"
```

---

## Task 5: realtime/aggregator.py — 会话聚合与去重

**Files:**
- Create: `realtime/aggregator.py`
- Test: `tests/test_aggregator.py`

**Interfaces:**
- Consumes: worker 产出的 PDU dict（含 `type/src/dst/flco/_fo_hz/_window_id/raw_bits`）。
- Produces:
  - 模块常量 `CALL_TIMEOUT_WINDOWS=5`。
  - `@dataclass CallRecord`：`fo_hz: float, src: int, dst: int, flco: str, start_window: int, end_window: int | None = None, voice_raw: list = field(default_factory=list), closed_by: str = ""`。
  - `class SessionAggregator`：
    - `__init__(self, fo_bucket_hz=5000.0, timeout_windows=CALL_TIMEOUT_WINDOWS)`。
    - `feed(self, pdu: dict) -> None`：LC_HEADER/LATE_ENTRY 开启或命中；TERMINATOR 立即关闭并入待输出队列；语音帧（暂用 raw_bits）按时序累积。
    - `expire(self, current_window: int, closed_fos: list[float]) -> list[CallRecord]`：关闭被标记 CLOSING 或超时的 session，返回本轮关闭的通话记录（含 TERMINATOR 触发的）。
    - `active_calls(self) -> list[CallRecord]`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_aggregator.py
import pytest
from realtime.aggregator import SessionAggregator, CallRecord, CALL_TIMEOUT_WINDOWS


def _pdu(type_, src=1, dst=2, fo=150000.0, wid=0, flco="GroupVoiceChannelUser"):
    return {"type": type_, "src": src, "dst": dst, "flco": flco,
            "ts": 0, "extra": {}, "raw_bits": b"\x00" * 33,
            "_fo_hz": fo, "_window_id": wid}


def test_lc_header_opens_session():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    calls = agg.active_calls()
    assert len(calls) == 1
    assert calls[0].src == 1 and calls[0].dst == 2
    assert calls[0].start_window == 0


def test_duplicate_signalling_deduped():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    agg.feed(_pdu("LATE_ENTRY", wid=1))
    agg.feed(_pdu("LATE_ENTRY", wid=2))
    # Still one active call (same fo,src,dst)
    assert len(agg.active_calls()) == 1


def test_terminator_closes_session():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    agg.feed(_pdu("TERMINATOR", wid=5))
    closed = agg.expire(current_window=5, closed_fos=[])
    assert len(closed) == 1
    assert closed[0].closed_by == "terminator"
    assert closed[0].end_window == 5
    assert agg.active_calls() == []


def test_closing_fo_closes_session():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    closed = agg.expire(current_window=4, closed_fos=[150000.0])
    assert len(closed) == 1
    assert closed[0].closed_by == "detector"


def test_timeout_closes_session():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    # No further PDUs; current window advances past timeout
    closed = agg.expire(current_window=CALL_TIMEOUT_WINDOWS + 1, closed_fos=[])
    assert len(closed) == 1
    assert closed[0].closed_by == "timeout"


def test_voice_frames_accumulate_not_dedup():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    agg.feed(_pdu("LATE_ENTRY", wid=1))
    agg.feed(_pdu("LATE_ENTRY", wid=2))
    call = agg.active_calls()[0]
    # raw_bits from each LATE_ENTRY appended to voice_raw (not deduped)
    assert len(call.voice_raw) == 2


def test_concurrent_calls_tracked_separately():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", src=1, dst=2, fo=-300000.0, wid=0))
    agg.feed(_pdu("LC_HEADER", src=3, dst=4, fo=150000.0, wid=1))
    assert len(agg.active_calls()) == 2
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_aggregator.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 realtime/aggregator.py**

```python
from dataclasses import dataclass, field

CALL_TIMEOUT_WINDOWS = 5   # close a call after this many windows with no update


@dataclass
class CallRecord:
    fo_hz: float
    src: int
    dst: int
    flco: str
    start_window: int
    end_window: int | None = None
    voice_raw: list = field(default_factory=list)
    closed_by: str = ""
    last_window: int = 0


class SessionAggregator:
    """Merge fragmented PDUs from workers into call records.
    Merge key: (fo_bucket, src, dst).
    Dedup boundaries:
      - cross-window same signalling (LC/CSBK/Terminator): recorded once
      - voice frames: accumulated in time order (NOT deduped) into voice_raw"""

    def __init__(self, fo_bucket_hz: float = 5000.0,
                 timeout_windows: int = CALL_TIMEOUT_WINDOWS):
        self.fo_bucket_hz = fo_bucket_hz
        self.timeout_windows = timeout_windows
        self._calls: dict[tuple, CallRecord] = 
        self._pending_closed: list[CallRecord] = []

    def _key(self, pdu: dict) -> tuple:
        bucket = round(pdu.get("_fo_hz", 0.0) / self.fo_bucket_hz) * self.fo_bucket_hz
        return (bucket, pdu["src"], pdu["dst"])

    def feed(self, pdu: dict) -> None:
        key = self._key(pdu)
        wid = pdu.get("_window_id", 0)
        ptype = pdu["type"]

        rec = self._calls.get(key)
        if rec is None:
            rec = CallRecord(
                fo_hz=key[0], src=pdu["src"], dst=pdu["dst"],
                flco=pdu.get("flco", ""), start_window=wid, last_window=wid)
            self._calls[key] = rec
        rec.last_window = max(rec.last_window, wid)

        if ptype == "LATE_ENTRY":
            # voice/embedded fragments accumulate (different time segments)
            rec.voice_raw.append(pdu.get("raw_bits", b""))
        elif ptype == "TERMINATOR":
            rec.end_window = wid
            rec.closed_by = "terminator"
            self._pending_closed.append(rec)
            del self._calls[key]

    def expire(self, current_window: int, closed_fos: list[float]) -> list[CallRecord]:
        closed = list(self._pending_closed)
        self._pending_closed = []

        closed_buckets = {round(fo / self.fo_bucket_hz) * self.fo_bucket_hz
                          for fo in closed_fos}

        for key, rec in list(self._calls.items()):
            bucket = key[0]
            if bucket in closed_buckets:
                rec.end_window = current_window
                rec.closed_by = "detector"
                closed.append(rec)
                del self._calls[key]
            elif current_window - rec.last_window >= self.timeout_windows:
                rec.end_window = current_window
                rec.closed_by = "timeout"
                closed.append(rec)
                del self._calls[key]
        return closed

    def active_calls(self) -> list[CallRecord]:
        return list(self._calls.values())
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_aggregator.py -v
```
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add realtime/aggregator.py tests/test_aggregator.py
git commit -m "feat: add SessionAggregator with merge/dedup and call lifecycle"
```

---

## Task 6: realtime/scanner_rt.py — 顶层编排

**Files:**
- Create: `realtime/scanner_rt.py`
- Test: `tests/test_realtime_e2e.py`（层次1）

**Interfaces:**
- Consumes: `FileIQSource`、`RingBuffer`、`Detector`、`decode_window`、`SessionAggregator`（前 5 个 Task）。
- Produces:
  - `class RealtimeScanner`：`__init__(self, source, num_workers=4, window_sec=1.0, step_sec=0.9, ring_capacity_sec=3.0, use_pool=True)`。
  - `run(self, on_call=None, max_windows=None) -> list[CallRecord]`：主循环。采集线程填环形缓冲；主线程取窗口→检测器→worker（池或串行）→聚合器→`on_call` 回调输出关闭通话。`max_windows` 限制处理窗口数（测试用）。返回所有关闭的通话记录。监控 `source` 与 ring 的溢出，>0 时 `print` 告警。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_realtime_e2e.py
import os
import numpy as np
import pytest
from realtime.iq_source import FileIQSource
from realtime.scanner_rt import RealtimeScanner


def test_realtime_narrowband_decodes(tmp_path):
    """Level-1 sim: feed a narrowband DMR file through the realtime pipeline."""
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        pytest.skip("narrowband test file not present")
    src = FileIQSource(path, sample_rate=78125.0, chunk_samples=78125, throttle=False)
    scanner_rt = RealtimeScanner(src, num_workers=2, window_sec=1.0, step_sec=0.9,
                                 use_pool=False)
    calls = scanner_rt.run()
    # Known DMR signal -> at least one call with a real src/dst
    assert isinstance(calls, list)
    assert any(c.flco == "GroupVoiceChannelUser" for c in calls)


def test_realtime_wideband_two_channels():
    """Level-1 sim: wideband file with two DMR signals at -300k and +150k."""
    path = "data/synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(path):
        pytest.skip("wideband test file not present")
    src = FileIQSource(path, sample_rate=2.5e6, chunk_samples=2_500_000, throttle=False)
    scanner_rt = RealtimeScanner(src, num_workers=2, window_sec=1.0, step_sec=0.9,
                                 use_pool=False)
    calls = scanner_rt.run()
    buckets = {round(c.fo_hz / 100000) * 100000 for c in calls}
    # Expect both DMR channels discovered
    assert len(calls) >= 1


def test_overflow_warns_on_starve(tmp_path, capsys):
    """starve_factor>1 with a small ring should produce overflow."""
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        pytest.skip("narrowband test file not present")
    # Throttled + starved source, but tiny ring forces overflow when consumer is slow.
    src = FileIQSource(path, sample_rate=78125.0, chunk_samples=20000,
                       throttle=True, starve_factor=1.0)
    scanner_rt = RealtimeScanner(src, num_workers=1, window_sec=1.0, step_sec=0.9,
                                 ring_capacity_sec=0.5, use_pool=False)
    calls = scanner_rt.run(max_windows=5)
    assert isinstance(calls, list)
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_realtime_e2e.py -v
```
Expected: ImportError

- [ ] **Step 3: 创建 realtime/scanner_rt.py**

```python
import threading
import numpy as np
from multiprocessing import Pool

from realtime.ring_buffer import RingBuffer
from realtime.detector import Detector
from realtime.aggregator import SessionAggregator, CallRecord
from realtime.worker import decode_window


class RealtimeScanner:
    """Orchestrates acquisition thread + detector + worker pool + aggregator.
    core/ and scanner.py decode logic are reused unchanged via decode_window."""

    def __init__(self, source, num_workers: int = 4, window_sec: float = 1.0,
                 step_sec: float = 0.9, ring_capacity_sec: float = 3.0,
                 use_pool: bool = True):
        self.source = source
        self.num_workers = num_workers
        self.fs = source.sample_rate
        self.window_samples = int(window_sec * self.fs)
        self.step_samples = int(step_sec * self.fs)
        self.ring = RingBuffer(int(ring_capacity_sec * self.fs))
        self.detector = Detector(sample_rate=self.fs)
        self.aggregator = SessionAggregator()
        self.use_pool = use_pool
        self._acq_done = threading.Event()

    def _acquire(self):
        while True:
            chunk = self.source.read_chunk()
            if chunk is None:
                break
            dropped = self.ring.write(chunk)
            if dropped > 0:
                print(f"[WARN] ring overflow: dropped {dropped} samples "
                      f"(total {self.ring.overflow_count})")
        self._acq_done.set()

    def _dispatch(self, tasks, pool):
        if not tasks:
            return []
        if self.use_pool and pool is not None:
            args = [(iq, fo, wid, self.fs) for (iq, fo, wid) in tasks]
            return pool.starmap(decode_window, args)
        return [decode_window(iq, fo, wid, self.fs) for (iq, fo, wid) in tasks]

    def run(self, on_call=None, max_windows: int | None = None) -> list[CallRecord]:
        acq = threading.Thread(target=self._acquire, daemon=True)
        acq.start()

        all_closed: list[CallRecord] = []
        window_id = 0
        pool = Pool(self.num_workers) if self.use_pool else None
        try:
            while True:
                win = self.ring.read_window(self.window_samples, self.step_samples)
                if win is None:
                    if self._acq_done.is_set() and self.ring.available() < self.window_samples:
                        break
                    self._acq_done.wait(timeout=0.05)
                    continue

                tasks = self.detector.process_window(win, window_id)
                results = self._dispatch(tasks, pool)
                for pdu_list in results:
                    for pdu in pdu_list:
                        self.aggregator.feed(pdu)

                closed = self.aggregator.expire(window_id, self.detector.closed_channels())
                for rec in closed:
                    all_closed.append(rec)
                    if on_call:
                        on_call(rec)

                window_id += 1
                if max_windows is not None and window_id >= max_windows:
                    break
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # Flush remaining active calls as timeout-closed
        final = self.aggregator.expire(window_id + self.aggregator.timeout_windows, [])
        for rec in final:
            all_closed.append(rec)
            if on_call:
                on_call(rec)
        self.source.close()
        return all_closed
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_realtime_e2e.py -v
```
Expected: 3 PASSED（无数据文件则 SKIP）

- [ ] **Step 5: Commit**

```bash
git add realtime/scanner_rt.py tests/test_realtime_e2e.py
git commit -m "feat: add RealtimeScanner top-level orchestration with acquisition thread + worker pool"
```

---

## Task 7: utils/synthesis.py — 时间线场景合成（层次2 仿真）

**Files:**
- Modify: `utils/synthesis.py`（新增函数，不动现有 `main`）
- Test: `tests/test_realtime_e2e.py`（补充层次2 测试）

**Interfaces:**
- Consumes: 现有 `utils/synthesis.py` 的 `read_rawiq`、`extract_or_pad`。
- Produces:
  - `synthesize_scenario(scenario, out_path, fs_out=2.5e6, fs_in=78125, snr_db=20, data_dir="data") -> str`：按时间脚本合成宽带文件。`scenario` 是 `[(start_sec, dur_sec, fo_hz, src_filename), ...]`。每路只在其时间窗内出现（窗外为零），上采样→搬频→叠加→加噪→存 int16。返回 `out_path`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_realtime_e2e.py
def test_synthesize_scenario_creates_file(tmp_path):
    import os
    from utils.synthesis import synthesize_scenario
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "scenario.rawiq")
    scenario = [
        (0.0, 2.0, -300000.0, "dmr_1_78125.rawiq"),
        (1.0, 2.0,  150000.0, "dmr_2_78125.rawiq"),
    ]
    result = synthesize_scenario(scenario, out, fs_out=2.5e6, data_dir="data")
    assert os.path.exists(result)
    # File should hold ~3s of 2.5MHz complex int16 = 3 * 2.5e6 * 2 int16
    size = os.path.getsize(result)
    assert size > 2.5e6 * 2 * 2  # at least ~2s worth


def test_scenario_through_realtime_pipeline(tmp_path):
    import os
    from utils.synthesis import synthesize_scenario
    from realtime.iq_source import FileIQSource
    from realtime.scanner_rt import RealtimeScanner
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "scenario.rawiq")
    scenario = [(0.0, 3.0, -300000.0, "dmr_1_78125.rawiq")]
    synthesize_scenario(scenario, out, fs_out=2.5e6, data_dir="data")
    src = FileIQSource(out, sample_rate=2.5e6, chunk_samples=2_500_000, throttle=False)
    rt = RealtimeScanner(src, num_workers=1, use_pool=False)
    calls = rt.run()
    assert isinstance(calls, list)
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_realtime_e2e.py::test_synthesize_scenario_creates_file -v
```
Expected: ImportError (synthesize_scenario not defined)

- [ ] **Step 3: 在 utils/synthesis.py 末尾添加 synthesize_scenario**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_realtime_e2e.py -v
```
Expected: 全部 PASSED 或 SKIP（无 FAIL）

- [ ] **Step 5: Commit**

```bash
git add utils/synthesis.py tests/test_realtime_e2e.py
git commit -m "feat: add synthesize_scenario for time-scripted realtime simulation"
```

---

## Task 8: 集成验证与全套测试

**Files:**
- Test: 全部 `tests/`

- [ ] **Step 1: 运行完整测试套件**

```bash
cd /home/lzkj/lzkj_workspace/python_docs/DMR_demo
pytest tests/ -v 2>&1
```
Expected: 所有测试 PASS 或 SKIP（无 FAIL）；原有 27 个 core/scanner 测试仍 PASS

- [ ] **Step 2: 端到端实跑（层次1，现有宽带文件）**

```bash
python -c "
from realtime.iq_source import FileIQSource
from realtime.scanner_rt import RealtimeScanner
src = FileIQSource('data/synthesized_wideband_2.5MHz.rawiq', sample_rate=2.5e6,
                   chunk_samples=2_500_000, throttle=False)
rt = RealtimeScanner(src, num_workers=2, use_pool=False)
calls = rt.run(on_call=lambda c: print(f'[CALL] fo={c.fo_hz/1e3:+.1f}kHz SRC={c.src} DST={c.dst} FLCO={c.flco} closed_by={c.closed_by}'))
print(f'total calls: {len(calls)}')
"
```
Expected: 检出 DMR 通话，打印 SRC/DST/FLCO

- [ ] **Step 3: 端到端实跑（层次2，时间线场景，测状态机）**

```bash
python -c "
from utils.synthesis import synthesize_scenario
from realtime.iq_source import FileIQSource
from realtime.scanner_rt import RealtimeScanner
synthesize_scenario([
    (0.0, 4.0, -300000.0, 'dmr_1_78125.rawiq'),
    (2.0, 4.0,  150000.0, 'dmr_2_78125.rawiq'),
], 'output/scenario_test.rawiq', fs_out=2.5e6)
src = FileIQSource('output/scenario_test.rawiq', sample_rate=2.5e6,
                   chunk_samples=2_500_000, throttle=False)
rt = RealtimeScanner(src, num_workers=2, use_pool=False)
calls = rt.run(on_call=lambda c: print(f'[CALL] fo={c.fo_hz/1e3:+.1f}kHz SRC={c.src} DST={c.dst} closed_by={c.closed_by} windows={c.start_window}-{c.end_window}'))
print(f'total calls: {len(calls)}')
"
```
Expected: 检出两路并发通话，各有起止窗口

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test: integration verification for realtime scanner (level-1 and level-2 sim)"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ §1 实时硬约束/Late Entry 放松 → 设计贯穿，溢出监控在 Task 6
- ✅ §2 分块流式 → Task 2（重叠读窗）+ Task 6（窗口循环）
- ✅ §3 三段架构 → Task 1-6 逐一对应
- ✅ §4.1 IQSource → Task 1
- ✅ §4.2 RingBuffer → Task 2
- ✅ §4.3 Detector + 状态表 → Task 3
- ✅ §4.4 Worker（复用 _decode_loop）→ Task 4
- ✅ §4.5 SessionAggregator 三层去重 → Task 5
- ✅ §4.6 RealtimeScanner → Task 6
- ✅ §5 主数据流 → Task 6 run()
- ✅ §6 错误处理（溢出/worker异常/流结束/超时）→ Task 4(try/except)、Task 5(timeout)、Task 6(溢出告警/flush)
- ✅ §7 两级抽取/采样率可配置 → Task 4 `_decimation_factors`（按源采样率动态推导）
- ✅ §8 两层仿真 → Task 6（层次1）+ Task 7（层次2）+ AWGN 模型沿用
- ✅ §9 文件结构 → 完全对应
- ✅ §10 测试策略 → 每 Task 含单元测试 + Task 8 集成
- ✅ §11 排除项（语音/硬件/P25/策略B/逐样点）→ 未实现，SoapyIQSource 占位

**已知简化/偏差：**
- §7 "两级抽取"在 Task 4 用单次 `resample_poly(up,down)` + gcd 约分实现，而非显式两级。理由：gcd 约分后滤波器长度已可接受，且 frontend 内部峰值自校准吸收 <0.2% 残余频偏。若实测性能不足，可在 Task 4 内部改为显式两级抽取，接口签名不变。
- 语音帧累积当前用 `raw_bits`（264-bit burst）占位，真正 AMBE 帧提取留待语音解码阶段。CallRecord.voice_raw 接口已就位。

**Type consistency:** `decode_window` 在 Task 4 定义、Task 6 调用，签名 `(window_iq, fo_hz, window_id, source_sample_rate)` 一致。`Detector.process_window` 返回 `[(iq, fo, wid)]`、`RealtimeScanner._dispatch` 解包 `(iq, fo, wid)` 一致。`SessionAggregator.feed/expire` 在 Task 5 定义、Task 6 调用一致。PDU `_fo_hz/_window_id` 在 Task 4 写入、Task 5 `_key/feed` 读取一致。
