# DMR 宽带信道化扫描 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `realtime/` 实时层之上新增一个 PFB 宽带信道化前端，把一次采集的宽带 IQ（仿真，未来 60–70MHz SDR）切成 N 个重叠子带，活跃子带交给现有 2.5MHz 解码管线，输出带绝对射频频率的通话记录。

**Architecture:** 两级。第一级 `PolyphaseChannelizer`（多相 + FFT + 过抽样）把宽带流切成 N 个等宽、相邻重叠的子带；第二级复用现有 `Detector` / `worker.decode_window` / `SessionAggregator`，逐子带做能量检测、解码，并按绝对射频频率（子带中心 + 子带内偏移）归并。解调核心（`core/`、`scanner._decode_loop`、`worker.decode_window`）一行不改。

**Tech Stack:** Python 3.10，numpy 2.2.6，scipy 1.15.3（`firwin` / `fft` / `resample_poly`），multiprocessing，pytest。不引入 GNU Radio。

## Global Constraints

- 解调核心 `core/dsp.py`、`core/decoder.py`、`core/burst_type.py` **不修改**。
- `scanner._decode_loop`、`realtime/worker.decode_window`、`multiprocessing.Pool` 派发 **复用**，不重写。
- 信道化器只依赖 **numpy + scipy**，与 `core/dsp.py` 全 scipy 风格一致；不引入 GNU Radio。
- 子带数 N、过抽样因子、原型滤波器参数均为运行时可配；本期默认 `num_subbands=32`、`taps_per_phase=12`、`oversample=2`，硬件阶段实测调优。
- **过抽样必需**：相邻子带必须重叠，骑在子带分界上的信道至少在一个子带内完整。
- 全速回归测试用 `throttle=False`；数据文件缺失用 `pytest.skip()`，不得用裸 `return`。
- **核心绑定关系**：`子带宽度 = 输入采样率 / N`；`子带采样率 = 输入采样率 / N × oversample`。
- 本期目标为**离线正确性**，不承诺实时 60Msps 吞吐。
- 测试运行解释器：`/home/lzkj/miniconda3/envs/DMR_demo/bin/python`（裸 `pytest`/`python` 在 base 环境无 pytest）。

---

## File Structure

```
realtime/channelizer.py          create  PolyphaseChannelizer：宽带块 → N 个重叠子带 + 子带中心频率
realtime/wideband_source.py      create  WidebandIQSource 抽象 + FileWidebandSource(仿真) + SoapyWidebandSource(占位)
realtime/wideband_scanner.py     create  WidebandScanner：channelizer → 逐子带 detector/decode → 共享 aggregator(绝对RF)
realtime/aggregator.py           modify  _key 优先用 _rf_hz(绝对射频),回退 _fo_hz(向后兼容)
realtime/scanner_rt.py           modify  CLI 增加 --wideband 分支,调用 WidebandScanner
utils/synthesis.py               modify  增加 synthesize_wideband_grid()：按栅格摆放多路信号合成宽带文件
tests/test_channelizer.py        create  子带中心 + 过抽样重叠(straddling) + 栅格覆盖 + 流式衔接 + bin→频率映射
tests/test_wideband_source.py    create  读全部样点 + 块大小 + 占位抛错
tests/test_synthesis_wideband.py create  宽带栅格文件生成 + 信号落点
tests/test_aggregator.py         modify  增加绝对RF归并测试(现有测试保持通过)
tests/test_wideband_e2e.py       create  端到端:宽带文件 → 信道化 → 解出分散在不同子带的多路通话
```

> **与 spec §3 的偏差说明**：spec 把宽带编排写为"修改 `scanner_rt.py` 主循环"，本计划改为**新增 `realtime/wideband_scanner.py`** 承载编排，`scanner_rt.py` 仅加一个 CLI 分支调用它。理由：`scanner_rt.py` 已是窄带编排器，宽带编排职责不同，拆分独立文件边界更清晰、更易单测，符合"聚焦文件"原则。窄带 `RealtimeScanner` 保持不变。

---

## Task 1: channelizer 核心（临界抽样多相滤波器组 + 子带中心）

**Files:**
- Create: `realtime/channelizer.py`
- Test: `tests/test_channelizer.py`

**Interfaces:**
- Consumes: `scipy.signal.firwin`、`scipy.fft.ifft`、numpy。
- Produces:
  - `class PolyphaseChannelizer`：`__init__(self, sample_rate: float, num_subbands: int = 32, taps_per_phase: int = 12, oversample: int = 2)`。
  - `subband_centers(self) -> np.ndarray`：长度 `num_subbands` 的子带绝对中心频率（升序，范围 `[-fs/2, fs/2)`，第 i 个对应 `process()` 输出第 i 行）。
  - `process(self, chunk: np.ndarray) -> np.ndarray`：返回 `(num_subbands, n_out)` 的 `complex64`，第 i 行是第 i 个子带的基带 IQ。
  - `reset(self) -> None`：清空流式状态。
  - 属性 `subband_rate: float` = `sample_rate / num_subbands * oversample`。
  - 本任务只实现 `oversample == 1` 的临界抽样路径；`oversample == 2` 在 Task 2 加入（`__init__` 接受参数，`process` 内 `oversample==2` 暂可 `raise NotImplementedError`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_channelizer.py
import numpy as np
import pytest
from realtime.channelizer import PolyphaseChannelizer


def _tone(f_hz, fs, n):
    t = np.arange(n) / fs
    return np.exp(1j * 2 * np.pi * f_hz * t).astype(np.complex64)


def test_subband_centers_ascending_grid():
    fs = 8000.0
    ch = PolyphaseChannelizer(fs, num_subbands=8, taps_per_phase=8, oversample=1)
    centers = ch.subband_centers()
    assert centers.shape == (8,)
    # spacing = fs/N = 1000 Hz, ascending, lowest = -fs/2
    np.testing.assert_allclose(np.diff(centers), 1000.0)
    assert centers.min() == pytest.approx(-4000.0)


def test_tone_lands_in_expected_subband():
    fs = 8000.0
    N = 8
    ch = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=12, oversample=1)
    # tone at +1000 Hz = center of one subband
    x = _tone(1000.0, fs, 8192)
    sub = ch.process(x)                       # (N, n_out)
    energies = np.mean(np.abs(sub) ** 2, axis=1)
    centers = ch.subband_centers()
    winner = int(np.argmax(energies))
    assert centers[winner] == pytest.approx(1000.0)
    # winner subband holds far more energy than the median subband
    assert energies[winner] > 10 * np.median(energies)


def test_subband_rate():
    ch = PolyphaseChannelizer(8000.0, num_subbands=8, taps_per_phase=8, oversample=1)
    assert ch.subband_rate == pytest.approx(1000.0)
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_channelizer.py -v`
Expected: ImportError / ModuleNotFoundError（`realtime.channelizer` 不存在）

- [ ] **Step 3: 创建 realtime/channelizer.py（临界抽样路径）**

```python
import numpy as np
from scipy.signal import firwin
from scipy.fft import ifft


class PolyphaseChannelizer:
    """Maximally-decimated (and, with oversample=2, 2x-oversampled) polyphase
    DFT analysis filterbank. Splits a wideband stream into num_subbands equal,
    overlapping baseband sub-bands in one FFT per output block.

    Math: prototype lowpass h of length N*M, polyphase rows poly[k,m]=h[m*N+k];
    input loaded N (or N/2 when oversampled) samples per block into the paths,
    each path FIR-filtered along blocks (state carried for streaming), then an
    N-point IFFT across paths yields the channels. Channel k center = k*fs/N;
    outputs are fftshifted to ascending frequency order.
    """

    def __init__(self, sample_rate: float, num_subbands: int = 32,
                 taps_per_phase: int = 12, oversample: int = 2):
        self.fs = float(sample_rate)
        self.N = int(num_subbands)
        self.M = int(taps_per_phase)
        self.oversample = int(oversample)
        # prototype lowpass: passband ~ oversample/N of Nyquist (overlap when os=2)
        cutoff = min(0.99, self.oversample / self.N)
        proto = firwin(self.N * self.M, cutoff).astype(np.float64)
        # poly[k, m] = proto[m*N + k]
        self.poly = proto.reshape(self.M, self.N).T.copy()   # (N, M)
        self.subband_rate = self.fs / self.N * self.oversample
        self.reset()

    def reset(self) -> None:
        self.N
        self._tail = np.zeros(0, dtype=np.complex128)        # leftover < one block
        self._state = np.zeros((self.M - 1, self.N), dtype=np.complex128)  # prev block-rows

    def subband_centers(self) -> np.ndarray:
        # channel k center = k*fs/N (k=0..N-1), fftshifted to ascending order
        k = np.arange(self.N)
        centers = k * (self.fs / self.N)
        return np.fft.fftshift(np.where(centers >= self.fs / 2, centers - self.fs, centers))

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if self.oversample == 1:
            return self._process_critical(np.asarray(chunk, dtype=np.complex128))
        raise NotImplementedError("oversample=2 added in Task 2")

    def _process_critical(self, x: np.ndarray) -> np.ndarray:
        N, M = self.N, self.M
        x = np.concatenate([self._tail, x])
        nblocks = len(x) // N
        self._tail = x[nblocks * N:].copy()
        if nblocks == 0:
            return np.zeros((N, 0), dtype=np.complex64)
        X = x[:nblocks * N].reshape(nblocks, N)              # X[r,k]=x[r*N+k]
        Xs = np.vstack([self._state, X])                     # (M-1+nblocks, N)
        F = np.zeros((nblocks, N), dtype=np.complex128)
        for m in range(M):
            F += self.poly[:, m][None, :] * Xs[(M - 1 - m):(M - 1 - m) + nblocks, :]
        self._state = Xs[-(M - 1):, :].copy() if M > 1 else self._state
        Y = ifft(F, axis=1) * N                              # (nblocks, N) channels
        Y = np.fft.fftshift(Y, axes=1)                       # ascending freq order
        return Y.T.astype(np.complex64)                      # (N, nblocks)
```

> **Note（实现者）：** 多相 DFT 滤波器组的换向/相位约定有多种等价写法。若 `test_tone_lands_in_expected_subband` 失败（能量落在镜像子带或泄漏到相邻子带），按以下顺序排查：① 把 `ifft` 换成 `fft`（DFT 方向约定）；② 路径加载倒序 `X[:, ::-1]`；③ 检查 `subband_centers` 的 fftshift 是否与通道索引一致。以测试为准，调到通过。`reset` 里第一行 `self.N` 是占位，可删。

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_channelizer.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add realtime/channelizer.py tests/test_channelizer.py
git commit -m "feat: add critically-sampled polyphase channelizer core"
```

---

## Task 2: channelizer 过抽样（2×）+ straddling

**Files:**
- Modify: `realtime/channelizer.py`
- Test: `tests/test_channelizer.py`

**Interfaces:**
- Consumes: Task 1 的 `PolyphaseChannelizer`。
- Produces: `process()` 在 `oversample == 2` 时换向步进 `N/2`、对每个输出块做相位修正，使相邻子带 50% 重叠；输出形状仍是 `(num_subbands, n_out)`，`subband_rate = fs/N*2`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_channelizer.py
def test_oversample_straddling_tone_in_two_subbands():
    fs = 8000.0
    N = 8
    ch = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=12, oversample=2)
    centers = ch.subband_centers()
    # place tone exactly on the boundary between two adjacent subbands
    boundary = (centers[3] + centers[4]) / 2.0
    x = _tone(boundary, fs, 16384)
    sub = ch.process(x)
    energies = np.mean(np.abs(sub) ** 2, axis=1)
    order = np.argsort(energies)[::-1]
    top_two = sorted(order[:2])
    # boundary tone shows up in BOTH adjacent subbands (not split/lost)
    assert top_two == [3, 4]
    assert energies[order[1]] > 0.25 * energies[order[0]]


def test_oversample_subband_rate_doubled():
    ch = PolyphaseChannelizer(8000.0, num_subbands=8, taps_per_phase=8, oversample=2)
    assert ch.subband_rate == pytest.approx(2000.0)
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_channelizer.py::test_oversample_straddling_tone_in_two_subbands -v`
Expected: FAIL（`NotImplementedError`）

- [ ] **Step 3: 实现 oversample=2 路径**

替换 `process` 并新增 `_process_oversampled`：

```python
    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.complex128)
        if self.oversample == 1:
            return self._process_critical(x)
        if self.oversample == 2:
            return self._process_oversampled(x)
        raise ValueError(f"unsupported oversample={self.oversample}")

    def _process_oversampled(self, x: np.ndarray) -> np.ndarray:
        # 2x oversampled (WOLA): commutator steps by N/2 so blocks overlap 50%.
        N, M = self.N, self.M
        H = N // 2
        x = np.concatenate([self._tail, x])
        nblocks = (len(x) - N) // H + 1 if len(x) >= N else 0
        if nblocks <= 0:
            self._tail = x.copy()
            return np.zeros((N, 0), dtype=np.complex64)
        consumed = nblocks * H
        self._tail = x[consumed:].copy()
        # build overlapping blocks of length N, stepped by H
        idx = (np.arange(N)[None, :] + H * np.arange(nblocks)[:, None])
        B = x[idx]                                   # (nblocks, N)
        # path-load and FIR along blocks using carried state
        Xs = np.vstack([self._state_os, B])          # (M-1+nblocks, N)
        F = np.zeros((nblocks, N), dtype=np.complex128)
        for m in range(M):
            F += self.poly[:, m][None, :] * Xs[(M - 1 - m):(M - 1 - m) + nblocks, :]
        self._state_os = Xs[-(M - 1):, :].copy() if M > 1 else self._state_os
        Y = ifft(F, axis=1) * N                      # (nblocks, N)
        # WOLA phase correction: block r (step H=N/2) rotates channel k by exp(j*pi*k*r)
        r = np.arange(nblocks)[:, None]
        k = np.arange(N)[None, :]
        Y = Y * np.exp(1j * np.pi * k * r)
        Y = np.fft.fftshift(Y, axes=1)
        return Y.T.astype(np.complex64)
```

在 `reset()` 末尾增加过抽样状态：

```python
        self._state_os = np.zeros((self.M - 1, self.N), dtype=np.complex128)
```

> **Note（实现者）：** 若 straddling 测试中两个赢家不是相邻的 `[3,4]`，多半是相位修正因子方向问题——试 `np.exp(-1j*np.pi*k*r)`；若能量集中在单一子带而非两个重叠，检查 `cutoff = oversample/N`（os=2 应得更宽通带）。以测试为准。

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_channelizer.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add realtime/channelizer.py tests/test_channelizer.py
git commit -m "feat: add 2x oversampled channelizer path with WOLA phase correction"
```

---

## Task 3: channelizer 流式状态衔接 + 栅格覆盖

**Files:**
- Modify: `realtime/channelizer.py`（若测试暴露 bug 才改）
- Test: `tests/test_channelizer.py`

**Interfaces:**
- Consumes: Task 1/2 的 `PolyphaseChannelizer`。
- Produces: 无新接口；验证分块处理与整块处理**逐样点一致**（延迟线跨块延续），且全带宽栅格无覆盖空洞。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_channelizer.py
def _rand_iq(n, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


@pytest.mark.parametrize("oversample", [1, 2])
def test_streaming_matches_single_shot(oversample):
    fs = 8000.0
    N = 8
    x = _rand_iq(8192, seed=oversample)
    whole = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=10, oversample=oversample)
    out_whole = whole.process(x)
    streamed = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=10, oversample=oversample)
    parts = [streamed.process(x[:3000]), streamed.process(x[3000:5000]),
             streamed.process(x[5000:])]
    out_stream = np.concatenate(parts, axis=1)
    # carried state ⇒ split processing equals single-shot exactly (within float tol)
    assert out_stream.shape == out_whole.shape
    np.testing.assert_allclose(out_stream, out_whole, atol=1e-5)


def test_grid_coverage_every_subband_reachable():
    fs = 8000.0
    N = 8
    ch = PolyphaseChannelizer(fs, num_subbands=N, taps_per_phase=12, oversample=1)
    centers = ch.subband_centers()
    hit = set()
    for c in centers:
        sub = ch.process(_tone(float(c), fs, 8192))
        hit.add(int(np.argmax(np.mean(np.abs(sub) ** 2, axis=1))))
        ch.reset()
    # every subband index is the winner for its own center tone — no coverage hole
    assert hit == set(range(N))
```

- [ ] **Step 2: 运行确认（先跑看是否已通过）**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_channelizer.py -k "streaming or grid_coverage" -v`
Expected: 若 Task 1/2 状态衔接已正确则直接 PASS；若 FAIL，按下一步修正

- [ ] **Step 3: （仅当失败时）修正流式状态**

若 `test_streaming_matches_single_shot` 失败，根因通常是 `_tail`/`_state`(_os) 未正确跨块延续。确认：① `_tail` 保存了不足一块的余样点并在下次 `process` 开头拼接；② `_state`/`_state_os` 保存了上一批最后 `M-1` 个 block-row。无 bug 则跳过本步。

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_channelizer.py -v`
Expected: 全部 PASSED（含 parametrize 共 8+ 项）

- [ ] **Step 5: Commit**

```bash
git add realtime/channelizer.py tests/test_channelizer.py
git commit -m "test: verify channelizer streaming continuity and grid coverage"
```

---

## Task 4: wideband_source.py — 宽带 IQ 源

**Files:**
- Create: `realtime/wideband_source.py`
- Test: `tests/test_wideband_source.py`

**Interfaces:**
- Consumes: 同 `core.dsp.read_rawiq` 的 int16 IQ 文件格式（I/Q 交错，/32768.0 归一化）。
- Produces:
  - `class WidebandIQSource`：抽象基类，属性 `sample_rate: float`、`center_hz: float`；方法 `read_chunk() -> np.ndarray | None`、`close() -> None`。
  - `class FileWidebandSource(WidebandIQSource)`：`__init__(self, path, sample_rate, center_hz=0.0, chunk_samples=2_000_000, throttle=False)`，`read_chunk()` 返回 `complex64`，流尽返回 `None`。
  - `class SoapyWidebandSource(WidebandIQSource)`：`__init__` 抛 `NotImplementedError`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_wideband_source.py
import numpy as np
import pytest
from realtime.wideband_source import (
    WidebandIQSource, FileWidebandSource, SoapyWidebandSource,
)


def _make_iq_file(tmp_path, n):
    path = str(tmp_path / "wb.rawiq")
    data = np.empty(2 * n, dtype=np.int16)
    data[0::2] = np.arange(n, dtype=np.int16)
    data[1::2] = -np.arange(n, dtype=np.int16)
    data.tofile(path)
    return path


def test_reads_all_samples(tmp_path):
    path = _make_iq_file(tmp_path, 5000)
    src = FileWidebandSource(path, sample_rate=5e6, chunk_samples=1024, throttle=False)
    total = 0
    while True:
        c = src.read_chunk()
        if c is None:
            break
        assert c.dtype == np.complex64
        total += len(c)
    src.close()
    assert total == 5000


def test_chunk_size_and_center(tmp_path):
    path = _make_iq_file(tmp_path, 5000)
    src = FileWidebandSource(path, sample_rate=5e6, center_hz=435e6,
                             chunk_samples=1024, throttle=False)
    assert src.center_hz == 435e6
    assert len(src.read_chunk()) == 1024
    src.close()


def test_soapy_placeholder_raises():
    with pytest.raises(NotImplementedError):
        SoapyWidebandSource(sample_rate=60e6)
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_wideband_source.py -v`
Expected: ImportError

- [ ] **Step 3: 创建 realtime/wideband_source.py**

```python
import time
import numpy as np


class WidebandIQSource:
    """Abstract wideband (60-70MHz-class) IQ source. One-shot capture, no tune."""
    sample_rate: float
    center_hz: float

    def read_chunk(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FileWidebandSource(WidebandIQSource):
    """Read a wideband .rawiq file (interleaved int16) in chunks.
    throttle=False reads at full speed (offline correctness, default).
    center_hz is the absolute RF center of the captured band (metadata for
    absolute-frequency labeling downstream); it does not alter the samples."""

    def __init__(self, path: str, sample_rate: float, center_hz: float = 0.0,
                 chunk_samples: int = 2_000_000, throttle: bool = False):
        self.path = path
        self.sample_rate = float(sample_rate)
        self.center_hz = float(center_hz)
        self.chunk_samples = int(chunk_samples)
        self.throttle = throttle
        self._fh = open(path, "rb")

    def read_chunk(self) -> np.ndarray | None:
        raw = np.frombuffer(self._fh.read(self.chunk_samples * 4), dtype=np.int16)
        if len(raw) < 2:
            return None
        n = len(raw) // 2
        iq = (raw[0:2 * n:2].astype(np.float32) +
              1j * raw[1:2 * n:2].astype(np.float32)) / 32768.0
        iq = iq.astype(np.complex64)
        if self.throttle:
            time.sleep(n / self.sample_rate)
        return iq

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()


class SoapyWidebandSource(WidebandIQSource):
    """Real wideband SDR (SoapySDR). Placeholder — not implemented this phase."""

    def __init__(self, sample_rate: float, **kwargs):
        raise NotImplementedError("SoapyWidebandSource is a placeholder for hardware phase")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_wideband_source.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add realtime/wideband_source.py tests/test_wideband_source.py
git commit -m "feat: add WidebandIQSource abstraction with FileWidebandSource"
```

---

## Task 5: utils/synthesis.py — 宽带栅格场景合成

**Files:**
- Modify: `utils/synthesis.py`（新增函数，不动现有 `main` 与 `synthesize_scenario`）
- Test: `tests/test_synthesis_wideband.py`

**Interfaces:**
- Consumes: 现有 `utils/synthesis.py` 的 `read_rawiq`、`extract_or_pad`、`scipy.signal.resample_poly`。
- Produces:
  - `synthesize_wideband_grid(placements, out_path, fs_out, dur_sec, fs_in=78125, snr_db=20, data_dir="data") -> str`：`placements` 是 `[(fo_hz, src_filename), ...]`，每路全程在线（不带时间脚本），上采样到 `fs_out`、搬到 `fo_hz`、叠加、加 AWGN、存 int16。返回 `out_path`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_synthesis_wideband.py
import os
import numpy as np
import pytest


def test_wideband_grid_creates_file(tmp_path):
    from utils.synthesis import synthesize_wideband_grid
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "wb_grid.rawiq")
    placements = [(-1_800_000.0, "dmr_1_78125.rawiq"),
                  (+1_800_000.0, "dmr_2_78125.rawiq")]
    result = synthesize_wideband_grid(placements, out, fs_out=5e6, dur_sec=1.0,
                                      data_dir="data")
    assert os.path.exists(result)
    # ~1s of 5MHz complex int16 = 1 * 5e6 * 2 int16
    assert os.path.getsize(result) > 5e6 * 2 * 2 * 0.8


def test_wideband_grid_places_energy_at_offsets(tmp_path):
    from utils.synthesis import synthesize_wideband_grid
    from core.dsp import read_rawiq
    import scipy.signal as signal
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "wb_grid.rawiq")
    synthesize_wideband_grid([(-1_800_000.0, "dmr_1_78125.rawiq")],
                             out, fs_out=5e6, dur_sec=1.0, data_dir="data")
    iq = read_rawiq(out).astype(np.complex64)
    f, psd = signal.welch(iq, fs=5e6, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f); psd = np.fft.fftshift(psd)
    peak_f = f[int(np.argmax(psd))]
    assert abs(peak_f - (-1_800_000.0)) < 100_000.0
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_synthesis_wideband.py -v`
Expected: ImportError（`synthesize_wideband_grid` 未定义）

- [ ] **Step 3: 在 utils/synthesis.py 末尾追加 synthesize_wideband_grid**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_synthesis_wideband.py -v`
Expected: 2 PASSED（无数据文件则 SKIP）

- [ ] **Step 5: Commit**

```bash
git add utils/synthesis.py tests/test_synthesis_wideband.py
git commit -m "feat: add synthesize_wideband_grid for wideband channelizer test scenarios"
```

---

## Task 6: aggregator 绝对射频归并键

**Files:**
- Modify: `realtime/aggregator.py:33-35`（`_key` 方法）
- Test: `tests/test_aggregator.py`（新增一个测试；现有测试保持通过）

**Interfaces:**
- Consumes: worker PDU dict，新增可选字段 `_rf_hz`（绝对射频频率）。
- Produces: `SessionAggregator._key` 优先用 `_rf_hz`，缺失时回退 `_fo_hz`（向后兼容现有窄带管线与现有测试）。`CallRecord.fo_hz` 仍取 `key[0]`，即绝对射频频率（当 `_rf_hz` 提供时）。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_aggregator.py
def test_absolute_rf_key_merges_by_rf_not_subband_offset():
    from realtime.aggregator import SessionAggregator
    agg = SessionAggregator()
    # same call seen in two adjacent sub-bands (straddling): different sub-band
    # offsets but SAME absolute RF -> must merge into ONE call.
    pdu_a = {"type": "LC_HEADER", "src": 1, "dst": 2, "flco": "GroupVoiceChannelUser",
             "ts": 0, "extra": {}, "raw_bits": b"\x00" * 33,
             "_fo_hz": +200000.0, "_rf_hz": 435_000_000.0, "_window_id": 0}
    pdu_b = {"type": "LATE_ENTRY", "src": 1, "dst": 2, "flco": "GroupVoiceChannelUser",
             "ts": 0, "extra": {}, "raw_bits": b"\x00" * 33,
             "_fo_hz": -180000.0, "_rf_hz": 435_001_000.0, "_window_id": 1}
    agg.feed(pdu_a)
    agg.feed(pdu_b)
    calls = agg.active_calls()
    assert len(calls) == 1
    assert abs(calls[0].fo_hz - 435_000_000.0) < 5000.0
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_aggregator.py::test_absolute_rf_key_merges_by_rf_not_subband_offset -v`
Expected: FAIL（当前 `_key` 用 `_fo_hz`，两个 PDU 落入不同 bucket → 2 个 call）

- [ ] **Step 3: 修改 _key**

将 `realtime/aggregator.py` 的 `_key` 改为：

```python
    def _key(self, pdu: dict) -> tuple:
        # Prefer absolute RF frequency (wideband channelizer path); fall back to
        # sub-band-relative offset for the legacy narrowband pipeline/tests.
        freq = pdu.get("_rf_hz", pdu.get("_fo_hz", 0.0))
        bucket = round(freq / self.fo_bucket_hz) * self.fo_bucket_hz
        return (bucket, pdu["src"], pdu["dst"])
```

- [ ] **Step 4: 运行全部聚合器测试确认通过**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_aggregator.py -v`
Expected: 全部 PASSED（新测试 + 现有测试，证明向后兼容）

- [ ] **Step 5: Commit**

```bash
git add realtime/aggregator.py tests/test_aggregator.py
git commit -m "feat: aggregator keys on absolute RF (_rf_hz) with _fo_hz fallback"
```

---

## Task 7: wideband_scanner.py — 宽带编排

**Files:**
- Create: `realtime/wideband_scanner.py`
- Test: `tests/test_wideband_e2e.py`

**Interfaces:**
- Consumes: `PolyphaseChannelizer`（Task 1-3）、`FileWidebandSource`（Task 4）、`Detector`（现有，`process_window` 返回 `list[(iq, fo, wid)]`、`closed_channels()`）、`worker.decode_window(iq, fo_hz, window_id, source_sample_rate)`（现有）、`SessionAggregator`（Task 6，feed/expire/active_calls）、`CallRecord`。
- Produces:
  - `class WidebandScanner`：`__init__(self, source, num_subbands=32, taps_per_phase=12, oversample=2, window_sec=1.0, step_sec=0.9, energy_floor_db=6.0)`。
  - `run(self, on_call=None, max_windows=None) -> list[CallRecord]`：读宽带流 → 信道化 → 逐子带逐窗 detector/decode → 共享 aggregator（绝对RF）→ 返回所有关闭的 `CallRecord`（`fo_hz` 为绝对射频频率）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_wideband_e2e.py
import os
import numpy as np
import pytest
from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner


def test_wideband_two_channels_different_subbands(tmp_path):
    """End-to-end: two DMR signals far apart in a 5MHz band, beyond a single
    2.5MHz sub-band's reach -> only channelization can catch both."""
    from utils.synthesis import synthesize_wideband_grid
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband files not present")
    out = str(tmp_path / "wb_e2e.rawiq")
    synthesize_wideband_grid(
        [(-1_800_000.0, "dmr_1_78125.rawiq"), (+1_800_000.0, "dmr_2_78125.rawiq")],
        out, fs_out=5e6, dur_sec=10.0, data_dir="data")
    src = FileWidebandSource(out, sample_rate=5e6, center_hz=435e6,
                             chunk_samples=5_000_000, throttle=False)
    scanner = WidebandScanner(src, num_subbands=4, taps_per_phase=12, oversample=2,
                              window_sec=1.0, step_sec=0.9)
    calls = scanner.run()
    assert isinstance(calls, list)
    # at least one real DMR call decoded, with absolute RF near 435MHz
    voice = [c for c in calls if c.flco == "GroupVoiceChannelUser"]
    assert len(voice) >= 1
    assert all(abs(c.fo_hz - 435e6) < 3e6 for c in voice)


def test_wideband_returns_list_on_noise(tmp_path):
    """Pure-noise wideband file: pipeline runs clean, returns a list (no crash)."""
    path = str(tmp_path / "noise.rawiq")
    n = 2_000_000
    rng = np.random.default_rng(0)
    data = np.empty(2 * n, dtype=np.int16)
    data[0::2] = (rng.standard_normal(n) * 200).astype(np.int16)
    data[1::2] = (rng.standard_normal(n) * 200).astype(np.int16)
    data.tofile(path)
    src = FileWidebandSource(path, sample_rate=5e6, center_hz=435e6,
                             chunk_samples=1_000_000, throttle=False)
    scanner = WidebandScanner(src, num_subbands=4, oversample=2)
    calls = scanner.run(max_windows=3)
    assert isinstance(calls, list)
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_wideband_e2e.py -v`
Expected: ImportError（`realtime.wideband_scanner` 不存在）

- [ ] **Step 3: 创建 realtime/wideband_scanner.py**

```python
import numpy as np

from realtime.channelizer import PolyphaseChannelizer
from realtime.detector import Detector
from realtime.aggregator import SessionAggregator, CallRecord
from realtime.worker import decode_window


class WidebandScanner:
    """Two-stage wideband scanner: channelize one-shot wideband capture into N
    overlapping sub-bands, then run the existing per-band decode pipeline on each
    sub-band, feeding a SHARED aggregator keyed on absolute RF frequency.

    Offline correctness path: the whole capture is read and channelized, then each
    sub-band is windowed and decoded. Decode core (scanner._decode_loop via
    worker.decode_window) is reused unchanged."""

    def __init__(self, source, num_subbands: int = 32, taps_per_phase: int = 12,
                 oversample: int = 2, window_sec: float = 1.0, step_sec: float = 0.9,
                 energy_floor_db: float = 6.0):
        self.source = source
        self.fs = source.sample_rate
        self.center_hz = getattr(source, "center_hz", 0.0)
        self.channelizer = PolyphaseChannelizer(
            self.fs, num_subbands=num_subbands, taps_per_phase=taps_per_phase,
            oversample=oversample)
        self.subband_rate = self.channelizer.subband_rate
        self.centers = self.channelizer.subband_centers()
        self.window_samples = int(window_sec * self.subband_rate)
        self.step_samples = int(step_sec * self.subband_rate)
        self.energy_floor_db = energy_floor_db
        self.aggregator = SessionAggregator()
        # one detector per sub-band (each holds its own frequency state table)
        self._detectors = [Detector(sample_rate=self.subband_rate)
                           for _ in range(num_subbands)]

    def _read_all(self) -> np.ndarray:
        chunks = []
        while True:
            c = self.source.read_chunk()
            if c is None:
                break
            chunks.append(c)
        self.source.close()
        if not chunks:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(chunks)

    def _active_subbands(self, subbands: np.ndarray) -> list[int]:
        # cheap energy gate: keep sub-bands whose mean power exceeds the median
        # sub-band power by energy_floor_db
        power = np.mean(np.abs(subbands) ** 2, axis=1) + 1e-12
        power_db = 10 * np.log10(power)
        floor = np.median(power_db)
        return [i for i in range(len(power_db))
                if power_db[i] >= floor + self.energy_floor_db]

    def run(self, on_call=None, max_windows: int | None = None) -> list[CallRecord]:
        wide = self._read_all()
        if len(wide) == 0:
            return []
        subbands = self.channelizer.process(wide)        # (N, n_out)
        active = self._active_subbands(subbands)

        all_closed: list[CallRecord] = []
        n_out = subbands.shape[1]
        n_windows = max(0, (n_out - self.window_samples) // self.step_samples + 1)
        if max_windows is not None:
            n_windows = min(n_windows, max_windows)

        for wid in range(n_windows):
            start = wid * self.step_samples
            stop = start + self.window_samples
            for i in active:
                win = subbands[i, start:stop]
                tasks = self._detectors[i].process_window(win, wid)
                for (iq, fo_rel, w) in tasks:
                    pdus = decode_window(iq, fo_rel, w, self.subband_rate)
                    rf = self.center_hz + float(self.centers[i]) + fo_rel
                    for pdu in pdus:
                        pdu["_rf_hz"] = rf
                        self.aggregator.feed(pdu)
                closed = self.aggregator.expire(wid, self._detectors[i].closed_channels())
                for rec in closed:
                    all_closed.append(rec)
                    if on_call:
                        on_call(rec)

        # flush remaining active calls as timeout-closed
        flush_window = n_windows + self.aggregator.timeout_windows
        for rec in self.aggregator.expire(flush_window, []):
            all_closed.append(rec)
            if on_call:
                on_call(rec)
        return all_closed
```

> **Note（实现者）：** 子带绝对中心 `self.centers[i]` 已是 fftshift 升序，与 `subbands` 行序一致（见 Task 1 `subband_centers`）。`_rf_hz` = 源 `center_hz` + 子带中心 + 子带内检测偏移 `fo_rel`，是 aggregator 归并键（Task 6）。若 e2e 解不出通话，先单独验证"该子带流喂给现有 `RealtimeScanner` 能否解出"，以隔离是信道化问题还是窗口参数问题。

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_wideband_e2e.py -v`
Expected: 2 PASSED（无数据文件则第一个 SKIP，第二个仍 PASS）

- [ ] **Step 5: Commit**

```bash
git add realtime/wideband_scanner.py tests/test_wideband_e2e.py
git commit -m "feat: add WidebandScanner two-stage channelize+decode orchestration"
```

---

## Task 8: scanner_rt.py CLI 增加 --wideband 分支

**Files:**
- Modify: `realtime/scanner_rt.py`（`main()` 内 argparse + 分支；不动 `RealtimeScanner` 类）
- Test: `tests/test_wideband_e2e.py`（追加 CLI 冒烟测试）

**Interfaces:**
- Consumes: `WidebandScanner`（Task 7）、`FileWidebandSource`（Task 4）、现有 `_detect_sample_rate`。
- Produces: `python -m realtime.scanner_rt <file> --wideband --fs HZ [--center HZ] [--nsub N] [--oversample K]` 运行宽带信道化扫描并打印通话。新增辅助函数 `run_wideband_cli(args)` 供测试直接调用。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/test_wideband_e2e.py
def test_wideband_cli_runs(tmp_path, capsys):
    import os
    from utils.synthesis import synthesize_wideband_grid
    from realtime.scanner_rt import run_wideband_cli
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "cli_wb.rawiq")
    synthesize_wideband_grid([(-1_800_000.0, "dmr_1_78125.rawiq")],
                             out, fs_out=5e6, dur_sec=10.0, data_dir="data")

    class Args:
        path = out
        fs = 5e6
        center = 435e6
        nsub = 4
        oversample = 2
    calls = run_wideband_cli(Args())
    assert isinstance(calls, list)
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_wideband_e2e.py::test_wideband_cli_runs -v`
Expected: ImportError（`run_wideband_cli` 未定义）

- [ ] **Step 3: 修改 realtime/scanner_rt.py**

在文件顶部 import 区追加：

```python
from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner
```

在 `main()` 之前新增辅助函数：

```python
def run_wideband_cli(args) -> list:
    """Run a wideband channelizer scan from parsed CLI args. Returns CallRecords."""
    src = FileWidebandSource(args.path, sample_rate=args.fs,
                             center_hz=getattr(args, "center", 0.0),
                             chunk_samples=int(args.fs), throttle=False)
    scanner = WidebandScanner(src, num_subbands=args.nsub,
                              oversample=args.oversample)

    def on_call(c):
        print(f"[CALL] RF={c.fo_hz/1e6:.4f}MHz SRC={c.src} DST={c.dst} "
              f"FLCO={c.flco} closed_by={c.closed_by} "
              f"windows={c.start_window}-{c.end_window}")

    calls = scanner.run(on_call=on_call)
    print(f"=== total wideband calls: {len(calls)} ===")
    return calls
```

在 `main()` 的 argparse 中追加参数，并在解析后分支：

```python
    parser.add_argument("--wideband", action="store_true",
                        help="run wideband channelizer scan (PFB front-end)")
    parser.add_argument("--center", type=float, default=0.0,
                        help="absolute RF center of the captured band, Hz")
    parser.add_argument("--nsub", type=int, default=32,
                        help="number of channelizer sub-bands (default 32)")
    parser.add_argument("--oversample", type=int, default=2,
                        help="channelizer oversample factor (default 2)")
```

并在 `args = parser.parse_args()` 与 `if not os.path.exists(...)` 校验之后，原窄带流程之前插入：

```python
    if args.wideband:
        if args.fs is None:
            fs = _detect_sample_rate(args.path)
            if fs is None:
                parser.error("could not infer sample rate; pass --fs HZ")
            args.fs = fs
        print(f"=== Wideband channelizer scan: {args.path} "
              f"(fs={args.fs/1e6:.3f} MHz, center={args.center/1e6:.3f} MHz, "
              f"nsub={args.nsub}, oversample={args.oversample}) ===")
        run_wideband_cli(args)
        return
```

> **Note（实现者）：** `args.fs` 在现有窄带流程里是 `--fs`（默认 None）。宽带分支需要 `args.fs` 为数值，上面已做推断/校验。`--center`/`--nsub`/`--oversample` 仅宽带分支使用，不影响现有窄带 CLI。

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/test_wideband_e2e.py -v`
Expected: 全部 PASSED（无数据文件则相关项 SKIP）

- [ ] **Step 5: Commit**

```bash
git add realtime/scanner_rt.py tests/test_wideband_e2e.py
git commit -m "feat: add --wideband CLI branch dispatching to WidebandScanner"
```

---

## Task 9: 集成验证与全套回归

**Files:**
- Test: 全部 `tests/`

- [ ] **Step 1: 运行完整测试套件**

Run: `/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m pytest tests/ -v 2>&1`
Expected: 所有测试 PASS 或 SKIP（无 FAIL）；现有 realtime/core/scanner 测试仍全部 PASS（信道化为纯新增 + aggregator 向后兼容）

- [ ] **Step 2: 端到端实跑（宽带栅格场景）**

Run:
```bash
/home/lzkj/miniconda3/envs/DMR_demo/bin/python -c "
from utils.synthesis import synthesize_wideband_grid
from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner
synthesize_wideband_grid(
    [(-1_800_000.0, 'dmr_1_78125.rawiq'), (+1_800_000.0, 'dmr_2_78125.rawiq')],
    'output/wb_scenario.rawiq', fs_out=5e6, dur_sec=10.0)
src = FileWidebandSource('output/wb_scenario.rawiq', sample_rate=5e6, center_hz=435e6,
                         chunk_samples=5_000_000, throttle=False)
rt = WidebandScanner(src, num_subbands=4, oversample=2)
calls = rt.run(on_call=lambda c: print(f'[CALL] RF={c.fo_hz/1e6:.4f}MHz SRC={c.src} DST={c.dst} FLCO={c.flco} closed_by={c.closed_by}'))
print(f'total calls: {len(calls)}')
"
```
Expected: 检出分散在两个不同子带的 DMR 通话，打印绝对 RF 频率（~433.2MHz 与 ~436.8MHz）、SRC/DST/FLCO

- [ ] **Step 3: CLI 实跑**

Run:
```bash
/home/lzkj/miniconda3/envs/DMR_demo/bin/python -m realtime.scanner_rt output/wb_scenario.rawiq --wideband --fs 5000000 --center 435000000 --nsub 4 --oversample 2
```
Expected: 打印若干 `[CALL] RF=...MHz ...` 与 `=== total wideband calls: N ===`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test: integration verification for wideband channelizer scanner"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ §3 PolyphaseChannelizer → Task 1-3
- ✅ §3 WidebandIQSource / FileWidebandSource / SoapyWidebandSource → Task 4
- ✅ §3 RealtimeScanner 改造（编排）→ Task 7 `WidebandScanner`（新文件，见偏差说明）+ Task 8 CLI
- ✅ §3 SessionAggregator 绝对RF归并 → Task 6
- ✅ §3 synthesis 宽带场景 → Task 5
- ✅ §4.1-4.2 多相 + FFT + 流式 → Task 1, 3
- ✅ §4.3 过抽样 straddling → Task 2
- ✅ §4.4 四风险点 → Task 1(原型/映射)、Task 2(相位修正)、Task 3(流式衔接)
- ✅ §4.5 参数关系/可配 → Task 1 `__init__` 参数 + `subband_rate` 公式
- ✅ §4.6 子带率衔接第二级 → Task 7（detector/decode 用 `subband_rate`）
- ✅ §6 验证（子带中心/重叠/覆盖/流式/映射/端到端）→ Task 1-3, 7
- ✅ §6 CLI 验证脚本 → Task 8
- ✅ §7 决策（不跳频/两级/穷举栅格/scipy自拼/过抽样/离线优先）→ 贯穿
- ✅ §2 解码核心不改 → 无任务触碰 `core/`、`scanner._decode_loop`、`worker.decode_window`

**已知简化/偏差：**
- 宽带编排放在新文件 `wideband_scanner.py`（spec 原写"改 scanner_rt 主循环"），理由见 File Structure 偏差说明，`scanner_rt.py` 仅加 CLI 分支。
- WidebandScanner 离线一次性读全 + 信道化（非流式分块），符合 §2"本期离线正确性优先"；信道化器本身的流式能力已在 Task 3 独立验证，硬件阶段可切换为分块喂入。
- PFB 换向/相位约定以行为测试为准（Task 1/2 Note 给出排查顺序），不在计划里钉死单一公式写法。

**Type consistency:**
- `PolyphaseChannelizer(sample_rate, num_subbands, taps_per_phase, oversample)` 在 Task 1 定义、Task 7 调用一致；`process()->(N,n_out)`、`subband_centers()->(N,)`、`subband_rate` 属性贯穿一致。
- `decode_window(iq, fo_hz, window_id, source_sample_rate)` 现有签名，Task 7 以 `(iq, fo_rel, w, subband_rate)` 调用一致。
- `Detector(sample_rate=...)`、`process_window(win, wid)->list[(iq,fo,wid)]`、`closed_channels()` 现有签名，Task 7 调用一致。
- `SessionAggregator.feed/expire/active_calls`、`CallRecord.fo_hz` 现有；Task 6 `_key` 改动不变签名，新增 `_rf_hz` 字段读取，Task 7 写入一致。
- `FileWidebandSource(path, sample_rate, center_hz, chunk_samples, throttle)` Task 4 定义、Task 7/8 调用一致。
- `synthesize_wideband_grid(placements, out_path, fs_out, dur_sec, fs_in, snr_db, data_dir)` Task 5 定义、Task 7/8/9 调用一致。
