# DMR 实时 SDR 盲扫与信令解析系统设计

**日期：** 2026-06-17
**范围：** 将现有离线固定文件扫描系统改造为支持连续 IQ 流（仿真 + 未来 SDR 硬件）的实时检索、识别、解析系统
**不在范围内：** 语音解码（AMBE+2，预留 `Session.voice_raw` 接口）、P25/NXDN 解析、SDR 硬件实测（先用仿真验证，硬件接入仅替换 IQSource）

---

## 1. 设计目标与核心约束

### 1.1 目标

单台 SDR 设备在一段可配置带宽的频谱上**持续监视**，自动发现活跃的 DMR 通话，实时解析其信令（LC Header / CSBK / Terminator / Late Entry），输出结构化通话记录。先用文件仿真验证整条链路，硬件接入时仅替换数据源。

### 1.2 唯一的实时硬约束：采集不能丢样点

SDR 以固定采样率连续产出样点，写入容量有限的缓冲区。这是经典生产者-消费者模型：

- **生产者**（采集）：固定速率，不可调
- **消费者**（处理）：速率取决于实现
- 消费者持续慢于生产者 → 缓冲区溢出 → **样点永久丢失**（overflow）

丢样点不可逆：轻则符号定时错位、单个 burst 解不出；重则整段语音帧丢失、连 Late Entry 都救不回通话。

**核心设计原则：采集（前端）丢样点致命且不可逆；解码（后端）滞后只是延迟、可补救。** 因此架构把二者彻底隔离——采集前端只做最轻的搬样点工作，永远跑在实时线以上；解码后端可滞后，只要平均吞吐跟得上。

### 1.3 实时线定义

处理 1 秒数据必须在 1 秒内完成（≥1× 实时）。当前实测单信道块处理 = 233ms（4.3× 实时），其中重采样占 161ms 为主要优化点（见 §7）。

### 1.4 关键放松：Late Entry 消除"与脉冲赛跑"

DMR 通话一旦开始会连续发语音超帧（每 60ms 一帧）持续整个通话（典型十几秒）。即使后端忙、错过通话开头的 LC Header，也能从后续语音帧的嵌入信令重建 LC（SRC/DST/FLCO）。这把"必须实时抓住每个通话起点"放松为"通话持续期内注意到即可"。

---

## 2. 处理范式：分块流式（chunked streaming）

| 范式 | 数据单位 | filtfilt 可用 | 接无限流 | core/ 改动 | 延迟 |
|------|---------|--------------|---------|-----------|------|
| 逐样点流式 | 单样点 | ❌ | ✅ | 全部重写 | 极低 |
| 块处理（现状） | 整个文件 | ✅ | ❌ | — | 文件长度 |
| **分块流式（采用）** | **重叠窗口** | **✅** | **✅** | **几乎不改** | **一个窗口** |

采用**分块流式**：把连续流切成带重叠的窗口，每个窗口当作"一个小文件"喂给现有块处理逻辑。

- 窗口内部仍是块处理 → `filtfilt`、`resample_poly`、`find_sync_positions` **全部复用，不改**
- 窗口之间**重叠**覆盖跨边界 burst
- 避免逐样点流式所需的有状态算子重写（否则 `core/dsp.py` 推倒重来）

**窗口参数：**
- 窗口长 `WINDOW_SEC = 1.0s`
- 步进 `STEP_SEC = 0.9s`（读指针每次前进 0.9s）
- 重叠 = 0.1s = 100ms ＞ 一个 burst 长度（264 符号 × 10 SPS ÷ 48kHz ≈ 55ms），保证任何跨边界 burst 至少在一个窗口内完整出现

---

## 3. 整体架构

```
┌─ 采集前端（实时防线，必须 >1× 实时）──────────────┐
│  IQSource.read_chunk()  →  环形缓冲区（无损）       │
│   ├ FileIQSource：按配置采样率节流读 .rawiq（仿真） │
│   └ SoapyIQSource：真实 SDR（后期，接口同构）       │
└────────────────────────┬──────────────────────────┘
                         │ 连续 IQ 流
              ┌──────────▼──────────┐
              │  检测器/调度器        │  廉价能量检测(Welch/FFT)
              │  - 找活跃子带         │  维护"信道状态表"(快开慢关滞回)
              │  - 切重叠窗口         │  策略C: 持续派发每个活跃窗口
              │  - 按 fo 打标签       │
              └──────────┬──────────┘
                         │ 队列: (宽带IQ切片, fo, window_id)
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐     ┌─────────┐      ┌─────────┐
   │worker 1 │     │worker 2 │ ...  │worker N │  多进程池(可滞后)
   │DDC+抽取  │     │复用现有  │      │复用现有  │  →frontend→_decode_loop
   └────┬────┘     └────┬────┘      └────┬────┘
        └────────────────┼────────────────┘
                         │ PDU 结果(带 fo, window_id)
              ┌──────────▼──────────┐
              │  会话聚合器           │  Session 状态机
              │  - 跨窗口信令去重      │  LC/LateEntry→开,Terminator→关
              │  - 语音帧按时序累积    │  超时兜底关闭
              │  - 通话生命周期输出    │
              └─────────────────────┘
```

三条职责线：
- **采集前端**：唯一实时硬约束，只采集+缓冲，越轻越不丢样点
- **检测+调度**：廉价侦察，决定哪段频谱派活，维护信道状态表
- **worker 池**：可滞后的重活，复用现有解码核心，多进程绕开 GIL

**核心思路：实时层只做"切片调度"，解调核心（`core/`）完全复用、一行不改。**

---

## 4. 模块职责与接口

### 4.1 IQSource（数据源抽象）

仿真与硬件的唯一差异点。下游所有代码对 IQSource 类型无感知。

```python
class IQSource:
    """连续 IQ 源抽象。read_chunk 返回固定大小的 complex64 块。"""
    sample_rate: float
    def read_chunk(self) -> np.ndarray | None: ...  # None 表示流结束
    def close(self) -> None: ...

class FileIQSource(IQSource):
    """按 sample_rate 节流读 .rawiq 文件，模拟 SDR 实时节奏。
    chunk_samples 个样点为一块，每块之间 sleep(chunk_samples/sample_rate)。
    throttle=False 时全速读（用于快速回归测试，不模拟实时节奏）。
    starve_factor: 人为放慢吐出速度以复现丢样点（>1.0 表示慢于实时）。"""
    def __init__(self, path: str, sample_rate: float,
                 chunk_samples: int = 65536, throttle: bool = True,
                 starve_factor: float = 1.0): ...

class SoapyIQSource(IQSource):
    """真实 SDR（SoapySDR）。后期实现，签名同构。占位，本期不实现。"""
```

### 4.2 RingBuffer（环形缓冲区）

采集线程写、检测器读的无损缓冲。容量有限，溢出时计数并丢弃最旧数据（模拟硬件行为）。

```python
class RingBuffer:
    """单生产者单消费者环形缓冲。线程安全。
    容量 capacity_samples 个 complex64。"""
    def __init__(self, capacity_samples: int): ...
    def write(self, chunk: np.ndarray) -> int:
        """写入；返回因容量不足丢弃的样点数（>0 即发生溢出）。"""
    def read_window(self, window_samples: int, step_samples: int) -> np.ndarray | None:
        """读出 window_samples 长的窗口，读指针前进 step_samples（保留重叠）。
        数据不足一个窗口时返回 None。"""
    @property
    def overflow_count(self) -> int: ...  # 累计丢弃样点数（监控指标）
```

### 4.3 Detector（检测器 + 信道状态表）

```python
ACTIVE_THRESHOLD_DB = 15   # 高于噪底判活跃（沿用现有 PSD_PEAK_THRESHOLD_DB）
CLOSE_HYSTERESIS = 3       # 能量连续消失 N 窗才判通话结束（快开慢关）

class ChannelState(IntEnum):
    IDLE = 0; ACTIVE = 1; TRACKING = 2; CLOSING = 3

@dataclass
class ChannelRecord:
    fo_hz: float
    state: ChannelState
    last_active_window: int    # 上次检测到能量的窗口号
    missed_windows: int        # 连续未检测到能量的窗口数

class Detector:
    """对每个窗口做能量检测，维护按频点(信道)索引的状态表，
    决定哪些频点该派发解码任务。策略 C：每个活跃窗口都派发。"""
    def __init__(self, sample_rate: float,
                 channel_grid_hz: float = 12500.0): ...
    def process_window(self, window_iq: np.ndarray, window_id: int
                      ) -> list[tuple[np.ndarray, float, int]]:
        """对窗口做 Welch PSD → 找活跃频点 → 更新状态表 →
        返回需派发的 [(宽带IQ切片, fo_hz, window_id)]。
        策略 C：所有 ACTIVE/TRACKING 信道每窗都派发（语音帧靠聚合器按时序累积）。
        IDLE→ACTIVE 视为新通话；能量消失 CLOSE_HYSTERESIS 窗后→CLOSING。"""
    def closed_channels(self) -> list[float]:
        """返回本轮转入 CLOSING 的频点，供聚合器关闭对应 session。"""
```

> **频点归一化：** 按 `channel_grid_hz`（默认 12.5kHz DMR 信道栅格）把检测到的峰值频率量化到信道中心，避免同一信道因频率微抖被当成多个。

> **派发的 IQ 切片是整个宽带窗口（不在检测器做信道化）：** 检测器只输出"哪个 `fo` 该解"，把同一份宽带窗口 IQ 连同 `fo` 标签交给 worker；混频(DDC)与抽取都在 worker 内完成（见 §4.4）。这把全速率混频成本从实时防线（检测器/采集）移到可滞后的 worker，符合 §1.2 原则。代价是队列传的是宽带切片（数据量大）——本期求稳接受；未来若带宽/并发增大成为瓶颈，可退回"检测器做信道化、只传窄带流"（即处理模型策略 B 的变体），接口不变。

### 4.4 Worker（解码工作单元）

```python
def decode_window(window_iq: np.ndarray, fo_hz: float, window_id: int,
                  source_sample_rate: float) -> list[dict]:
    """对一个宽带 IQ 切片在指定频偏上解码，返回 PDU 列表(每个带 _fo_hz/_window_id)。
    内部：DDC(fo) → 两级抽取到 48kHz → frontend → _decode_loop。
    复用 core/ 与 scanner._decode_loop，不重写解调逻辑。
    设计为可被 multiprocessing.Pool 调用的纯函数（无共享状态）。"""
```

多进程池（`multiprocessing.Pool`）承载，绕开 GIL。worker 是纯函数、无状态，IQ 切片通过队列传入。

### 4.5 SessionAggregator（会话聚合器，策略 C 去重核心）

```python
CALL_TIMEOUT_WINDOWS = 5   # 无 Terminator 时，超时兜底关闭

@dataclass
class CallRecord:
    fo_hz: float
    src: int
    dst: int
    flco: str
    start_window: int
    end_window: int | None
    voice_raw: list = field(default_factory=list)  # 按时序累积，不去重
    closed_by: str = ""   # "terminator" | "timeout"

class SessionAggregator:
    """把 worker 流回的碎片化 PDU 按通话归并。
    归并键：(fo_bucket, src, dst)。
    三层去重边界：
      ① 窗口内：worker 的 seen_bursts(已有)
      ② 跨窗口同信令：相同 LC/CSBK/Terminator 只记一次
      ③ 语音帧：不去重，按时序 append 到 voice_raw(策略C的价值)"""
    def feed(self, pdu: dict) -> None:
        """LC_HEADER/LATE_ENTRY → 开启或命中 session；
        TERMINATOR → 立即关闭并输出；其余更新 last_seen。"""
    def expire(self, current_window: int, closed_fos: list[float]) -> list[CallRecord]:
        """关闭：被 Detector 标记 CLOSING 的频点，或超时 CALL_TIMEOUT_WINDOWS 窗。
        返回本轮关闭的通话记录。"""
    def active_calls(self) -> list[CallRecord]: ...
```

### 4.6 RealtimeScanner（顶层编排）

```python
class RealtimeScanner:
    """组装采集线程 + 检测器 + worker 池 + 聚合器，运行主循环。"""
    def __init__(self, source: IQSource, num_workers: int = 4,
                 window_sec: float = 1.0, step_sec: float = 0.9,
                 ring_capacity_sec: float = 3.0): ...
    def run(self, on_call: Callable[[CallRecord], None] | None = None) -> None:
        """主循环：采集线程填环形缓冲 → 主线程取窗口 → 检测器派发 →
        worker 池解码 → 聚合器归并 → on_call 回调输出关闭的通话。
        监控 source/ring 的 overflow_count，>0 时告警(丢样点)。"""
```

---

## 5. 主数据流（时间线）

```
采集线程(独立)         主循环                    worker池          聚合器
─────────────────────────────────────────────────────────────────────
read_chunk()──────►ring.write()
(节流,固定速率)        │(返回丢弃数→监控)
                     │
                     ▼ 每 step_sec
              ring.read_window()──►detector.process_window()
                                        │找活跃子带+更新状态表
                                        │策略C:每活跃窗都派
                                        ▼
                                  queue.put((切片,fo,wid))──►pool.apply
                                                              │DDC+抽取
                                                              │frontend
                                                              │_decode_loop
                                                              ▼
                                  aggregator.feed(pdu)◄──── PDU结果
                                        │归并/去重/累积语音
                                        ▼
                                  aggregator.expire()──►on_call(CallRecord)
                                  detector.closed_channels()  通话记录输出
```

---

## 6. 错误处理与监控

| 场景 | 处理 |
|------|------|
| 缓冲区溢出（丢样点） | `RingBuffer.overflow_count`/`IQSource` 计数递增，主循环检测到 >0 时 `log.warning`，持续溢出则提示降采样率/加 worker |
| worker 解码异常 | worker 内 try/except，单个窗口失败返回空列表，不拖垮池；记录失败 window_id |
| 流结束（文件读完） | `read_chunk()` 返回 None → 采集线程退出 → 排空队列 → 聚合器 expire 所有未关闭 session（标记 timeout）→ 主循环退出 |
| Terminator 丢失 | 超时兜底（`CALL_TIMEOUT_WINDOWS`）关闭 session，`closed_by="timeout"` |
| 频率微抖动 | 检测器按 `channel_grid_hz` 量化频点；聚合器按 `fo_bucket` 归并 |
| 能量瞬时闪断 | `CLOSE_HYSTERESIS` 快开慢关，避免一通话被切成多段 |

---

## 7. 性能优化：两级抽取

当前 `resample_poly(12, 625)` 单信道 161ms（分母 625 → 多相滤波器极长），是主瓶颈。

**优化：两级抽取。** 例如 2.5MHz 先廉价整数抽取到中间速率，再细调到 48kHz；每级滤波器短得多。目标把重采样从 161ms 降到 ~30ms，单信道回到 ~100ms（10× 实时），单核扛 8+ 信道。

> **采样率可配置**（用户选择）：设计不绑定具体采样率，把"瞬时带宽 vs 数据量"权衡留到运行时配置。`decode_window` 接受 `source_sample_rate`，两级抽取的因子按源采样率动态推导（优先选能整除到 48kHz 倍数的路径）。仿真在多个档位（240k / 960k / 2.4M）验证性能。

**容量公式：** 所需 worker 数 ≈ 峰值同时活跃信道数 ÷ 单核实时倍数 ≈ 5 ÷ 10 ≈ 1~2 核（优化后）。

---

## 8. 仿真源策略（两层）

仿真与实测走**完全相同的下游代码**，只换 IQSource 实现。仿真数据本质是已有的**真实对讲机窄带录音**（`data/dmr_*_78125.rawiq`），按场景重新合成。

### 层次 1：复用现有文件跑通管线
用现有 `data/synthesized_wideband_2.5MHz.rawiq`（2.5MHz，3 路信号全程在线，20dB AWGN）经 `FileIQSource` 节流喂入，验证"采集→检测→worker池→聚合器"整条链路连通、能解出两路 DMR。零新数据。

### 层次 2：升级合成器，造带时间线的场景（测状态机）
扩展 `utils/synthesis.py`，支持按时间脚本摆放通话：

```python
scenario = [
    # (起始秒, 时长秒, 频偏Hz, 源窄带文件)
    (0.0, 15.0, -300e3, "dmr_1_78125.rawiq"),  # 0s 开始,持续15s
    (5.0, 10.0, +150e3, "dmr_2_78125.rawiq"),  # 5s 并发插入,10s
    (8.0,  3.0, +600e3, "p25_1_78125.rawiq"),  # 8s 干扰(非DMR),3s
]
```

每路信号只在其时间窗内出现（窗外填零），其余照旧（上采样→搬频→加噪）。验证：
- 0s：检测 -300k 活跃 → 新建 session
- 5s：+150k 并发 → 第二 session
- 8s：+600k 出现但解不出 DMR（P25）→ 误报抑制
- 15/15/11s：能量消失 → session 超时/CLOSING 关闭

### 噪声模型（沿用现有合成器）
SNR 对整个宽带定义：`P_noise = P_signal / 10^(SNR/10)`，`P_signal = mean(|wideband_base|²)`（加噪前）。复数噪声实虚部各分 `P_noise/2`。**单信道滤窄到 12.5kHz 后带内 SNR 远高于标称宽带 SNR**，故 20dB 下 LC Header（强 FEC）可稳定解出。

---

## 9. 文件结构

```
realtime/__init__.py          create  (empty)
realtime/iq_source.py         create  IQSource / FileIQSource / SoapyIQSource(占位)
realtime/ring_buffer.py       create  RingBuffer
realtime/detector.py          create  Detector + ChannelState + ChannelRecord
realtime/worker.py            create  decode_window (复用 core/ 与 scanner._decode_loop)
realtime/aggregator.py        create  SessionAggregator + CallRecord
realtime/scanner_rt.py        create  RealtimeScanner 顶层编排
utils/synthesis.py            modify  增加 scenario 时间线合成函数(层次2)
tests/test_iq_source.py       create  节流节奏 + starve_factor 丢样点
tests/test_ring_buffer.py     create  溢出计数 + 重叠读窗
tests/test_detector.py        create  状态机转移 + 滞回 + 频点量化
tests/test_aggregator.py      create  三层去重 + 生命周期 + 超时
tests/test_realtime_e2e.py    create  层次1/层次2 端到端仿真
```

`core/`、`scanner.py` 现有解码逻辑**不改**（`scanner._decode_loop` 被 worker 复用）。

---

## 10. 测试策略

| 层级 | 测什么 | 怎么测 |
|------|--------|--------|
| 单元 | RingBuffer 溢出/重叠 | 写超容量验证 overflow_count；读窗验证重叠样点一致 |
| 单元 | Detector 状态机 | 构造能量出现/消失序列，断言 IDLE→ACTIVE→TRACKING→CLOSING 与滞回 |
| 单元 | Aggregator 去重 | 喂重复 LC（去重）、多段语音（累积）、Terminator（关闭）、无终止（超时） |
| 集成 | 节流节奏 | FileIQSource throttle=True，断言吐块间隔 ≈ chunk/fs |
| 集成 | 丢样点复现 | starve_factor>1 → 断言 overflow_count>0 且 Late Entry 仍救回通话 |
| 端到端 | 层次1 | 现有 2.5MHz 文件 → 断言解出两路 DMR 的 LC/Late Entry |
| 端到端 | 层次2 | 时间线场景 → 断言 session 起止时间、并发、P25 误报抑制 |

全速回归用 `throttle=False`（不 sleep）避免测试慢；实时节奏单独一个测试验证。

---

## 11. 不实现的内容（明确排除）

- 语音解码（AMBE+2）——预留 `CallRecord.voice_raw` 累积接口
- SDR 硬件实测——`SoapyIQSource` 仅占位，本期不实现
- P25/NXDN 解析——P25 仅作仿真干扰源验证误报抑制
- 策略 B（只在通话起止派发）——当前用策略 C 求稳，接口上可退回（检测器少派窗即可）
- 逐样点流式重写 `core/dsp.py`——分块流式已满足需求
