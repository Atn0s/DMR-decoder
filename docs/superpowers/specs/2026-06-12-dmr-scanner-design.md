# DMR 离线盲扫 + 完整信令解析系统设计

**日期：** 2026-06-12  
**范围：** 离线 IQ 文件扫描、DMR 信令全解析（LC Header / CSBK / Terminator / Late Entry）  
**不在范围内：** 实时 SDR 流式前端（架构预留接入点）、语音解码（后期扩展）、P25/NXDN 协议（后期扩展）

---

## 1. 目录结构

```
DMR_demo/
├── core/
│   ├── dsp.py          ← DDC、鉴频、NCC、符号恢复、相位标定
│   ├── decoder.py      ← LC Header / CSBK / Terminator / LateEntryCollector
│   └── burst_type.py   ← Slot Type 枚举 + Sync 模板常量
├── scanner.py          ← 多频点调度、会话跟踪、结果聚合
├── dmr_pipeline_v2.py  ← 保留为入口脚本，实现迁移到 core/
├── late_entry.py       ← 保留为入口脚本，实现迁移到 core/decoder.py
├── data/
├── output/
└── docs/
```

---

## 2. 模块职责边界

| 模块 | 输入 | 输出 | 不负责 |
|------|------|------|--------|
| `core/dsp.py` | numpy IQ 数组 + 频偏 fo | 符号数组 + burst 位置列表 | 任何协议解析 |
| `core/decoder.py` | 264-bit burst + sync_type | 结构化 PDU dict 或 None | 任何信号处理 |
| `core/burst_type.py` | — | 枚举常量 | — |
| `scanner.py` | IQ 文件路径 / 频点列表 | List[PDU dict] | DSP 细节、协议细节 |

`core/dsp.py` 只接受 numpy 数组，不知道数据来源——这是后续接入实时 SDR 的唯一接缝。

---

## 3. 主数据流

```
IQ 文件
    │
    ▼
scanner.scan_file(path)
    │
    │  宽带：Welch PSD → 峰值检测 → 候选频偏列表 [fo1, fo2, ...]
    │  窄带：直接 fo=0
    │
    ▼ 对每个 fo：
core/dsp.frontend(iq, fo)          → y（48kHz FM 解调基带）
core/dsp.find_sync_positions(y)    → [(pos, polarity, sync_type), ...]
core/dsp.recover_burst(y, pos)     → symbols[132]
    │
    ▼
core/decoder.decode_burst(symbols, sync_type) → PDU dict 或 None
    │  Voice Burst → LateEntryCollector.feed() → PDU 或 None
    │
    ▼
scanner：会话跟踪 + 去重
    │
    ▼
List[PDU]，打印 / JSON 输出
```

---

## 4. 关键接口签名

```python
# core/dsp.py
def frontend(iq: np.ndarray, fo: float, fs: float = 2_500_000) -> np.ndarray:
    """DDC + 鉴频 + 基带滤波，返回 48kHz 实数符号流"""

def find_sync_positions(y: np.ndarray) -> list[tuple[int, float, str]]:
    """NCC 扫描，返回 (center_sample, polarity, sync_type)
    sync_type ∈ {'MS_VOICE', 'BS_VOICE', 'DATA', 'RESERVED'}"""

def recover_burst(y, center, polarity, sync_type) -> np.ndarray | None:
    """相位标定 + 自适应判决，返回 132 符号数组"""

# core/decoder.py
def decode_burst(symbols: np.ndarray, sync_type: str) -> dict | None:
    """Data Sync burst 解码分发：LC_HEADER / CSBK / TERMINATOR。
    Voice Sync burst 不经此函数，由 scanner 直接路由给 LateEntryCollector。"""

class LateEntryCollector:
    def feed(self, ba264: bitarray) -> dict | None:
        """喂入一个 Voice Sync burst（264 bit），集齐 4 片时返回 PDU，否则返回 None。
        只接受 Voice Sync burst，Data Sync burst 不经过此路径。"""

# scanner.py
def scan_file(path: str, freq_list: list[float] | None = None) -> list[dict]:
    """扫描离线 IQ 文件，freq_list=None 时做盲搜（Welch PSD 峰值检测）"""
```

---

## 5. PDU 统一结构

所有解码结果使用统一 dict schema：

```python
{
  "type":     "LC_HEADER" | "CSBK" | "TERMINATOR" | "LATE_ENTRY",
  "src":      int,
  "dst":      int,
  "ts":       int,       # 时隙 0 或 1
  "flco":     str,       # e.g. "GROUP_VOICE_CHANNEL_USER"
  "extra":    dict,      # type-specific 字段
  "raw_bits": bytes,     # 原始 264 bit，留给后续 voice frame 提取
}
```

---

## 6. 各 PDU 类型解析链路

### LC Header & Terminator with LC

两者结构完全相同，仅 Slot Type 字段不同，复用同一 FEC 路径：

```
ba[0:98] + ba[166:264]  →  BPTC(196,96)  →  FLC 72 bit
                                          →  RS(12,9,4) 校验（mask=0x969696）
                                          →  FullLinkControl.from_bits()
```

Terminator 收到时：关闭对应 Session，计算通话时长。

### CSBK

```
ba[0:98] + ba[166:264]  →  BPTC(196,96)  →  96 bit CSBK PDU
                                          →  CRC-CCITT(16) 校验 → 80 bit 有效数据

CSBK PDU 布局（80 bit）：
  [0:8]   CSBKO  (Opcode，决定消息类型)
  [8]     Last
  [9]     P
  [10:16] FID
  [16:80] 内容（按 CSBKO 分）
```

使用 `okdmr` 库的 `CSBK.from_bits()` 解析，不手写字段切分。

常见 MS 互通场景 CSBKO：

| CSBKO | 名称 |
|-------|------|
| 0x01  | Unit to Unit Answer Request |
| 0x04  | Call Alert |
| 0x24  | Random Access |

### Late Entry（EMB 碎片重组）

```
Voice Burst[108:156]  →  16-bit EMB + 32-bit 信令碎片
                      →  EmbeddedSignalling.from_bits()  →  LCSS 字段
                      →  状态机：First → Cont → Cont → Last
                      →  4×32=128 bit
                      →  VBPTC(128,72) 解交织纠错
                      →  72-bit LC + 5-bit CS5
                      →  CS5 校验（严格模式）
                      →  FullLinkControl.from_bits()
```

`LateEntryCollector` 是有状态对象，由 scanner 持有，每个 Voice Burst 调一次 `feed()`。

---

## 7. 会话跟踪

scanner 内部维护会话表，把同一通话的帧组织在一起：

```python
class Session:
    src: int
    dst: int
    start_burst_type: str      # "LC_HEADER" 或 "CSBK"
    start_pdu: dict
    voice_raw: list[bytes]     # 原始 voice frame bits，留给后期 AMBE 解码
    terminator: dict | None
    late_entry_lc: dict | None
    duration_s: float | None
```

会话生命周期：
```
CSBK（可选）→ LC_HEADER → Voice×N → TERMINATOR
                               ↑
                     LateEntryCollector.feed()
```

去重键：`(src, dst, flco, ncc_peak_sample_in_raw_iq)`，其中 `ncc_peak_sample_in_raw_iq` 是在原始宽带 IQ（未经 DDC）中的 NCC 峰值位置，换算自 DDC 后的位置。同一 burst 被多个频偏候选命中时，保留 NCC 值最大的一次，其余丢弃。

---

## 8. 盲搜策略

**宽带文件（fs > 200 kHz）：**
1. Welch PSD，nperseg=4096
2. 峰值检测：高于噪底 15 dB，间距 > 20 bins
3. 对每个峰值频偏做 DDC → frontend → 解码

**窄带文件（fs ≤ 200 kHz）：**
1. 跳过 PSD 检测，fo=0
2. 直接 resample_poly → frontend → 解码

**输入文件采样率推断：**
- 优先从文件名中的数字提取（如 `dmr_2_78125.rawiq` → 78125 Hz）
- 无法推断时要求用户通过参数指定

---

## 9. 输出格式

**终端打印：**
```
[LC_HEADER  ] SRC=1234567 DST=9876543 FLCO=GROUP_VOICE ts=0 (fo=+12.3kHz)
[CSBK       ] CSBKO=CALL_ALERT SRC=1234567 DST=9876543
[LATE_ENTRY ] SRC=1234567 DST=9876543 CS5=OK
[TERMINATOR ] SRC=1234567 DST=9876543 duration=4.2s
```

**JSON 输出（可选）：**
```python
scanner.scan_file("data/xxx.rawiq", output_json="output/result.json")
```

---

## 10. 后期扩展接入点

| 扩展方向 | 接入点 |
|---------|--------|
| 实时 SDR | 替换 `scan_file()` 的 IQ 来源，`core/dsp.py` 不变 |
| 语音解码（AMBE） | 消费 `Session.voice_raw`，独立模块 `audio/ambe.py` |
| P25 / NXDN | 新增 `core/decoder_p25.py`，在 `scanner.py` 加协议分类前置步骤 |
| 频点数据库 | `scanner.scan_freqs(freq_db)` 替换盲搜，接口已预留 |

---

## 11. 不实现的内容（明确排除）

- AMBE+2 / AMBE2000 语音解码（当前阶段）
- 加密语音（RC4 / ARC4 / Basic Privacy）破解
- 基站（BS Sourced）特有信令（CAI / MFID 扩展）
- 数据信道（Data Header + Rate 块 + TPDU 重组）——列为第二期
- P25 / NXDN / dPMR / 模拟 FM——列为第三期
