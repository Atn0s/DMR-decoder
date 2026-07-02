# DMR 协议帧结构说明（草案）

> 版本：v0.1  
> 适用范围：当前工程中的 DMR 离线解码链路，以及后续完善实时/宽带链路时的协议结构参考。  
> 主要代码参考：`dmr/constants.py`、`dmr/dsp.py`、`dmr/decoder.py`、`dmr/offline.py`。  
> 标准参考方向：ETSI TS 102 361 系列。本文先按当前工程已实现和已验证的结构展开，未完整实现的业务类型会标注为“待补全”。

## 1. 基本参数

### 1.1 空口与物理层参数

| 项目 | 数值 / 说明 |
|------|-------------|
| 协议名称 | DMR（Digital Mobile Radio） |
| 频点 / 频段 | 协议本身不绑定固定 RF 频点；工程中的 `.rawiq` 是复基带文件，只保留相对中心频偏 |
| 典型信道带宽 | 12.5 kHz |
| 多址方式 | 2-slot TDMA |
| 调制方式 | 4FSK / 4-level FSK |
| 符号率 | 4800 symbols/s |
| 每符号承载 | 2 bit / symbol |
| 原始比特率 | 9600 bit/s |
| 当前工程解码采样率 | 48 kHz |
| 当前工程每符号采样点 | 10 samples/symbol |
| 当前工程前端低通 | 默认约 9.5 kHz |
| 当前工程 nominal deviation | 约 1944 Hz，用于 4FSK 鉴频幅度归一化 |
| 基本突发时长 | 30 ms |
| 基本突发长度 | 132 symbols = 264 raw bits |
| TDMA 帧时长 | 60 ms，包含 slot 1 和 slot 2 各 30 ms |
| 语音超帧时长 | 360 ms，6 个语音突发 A-F |
| 当前工程已实现同步 | BS/MS Voice Sync、BS/MS Data Sync |

### 1.2 当前工程使用的符号映射

当前工程把 4FSK 判决后的符号映射为 dibit：

| Dibit | 判决电平 |
|-------|----------|
| `01` | `+3` |
| `00` | `+1` |
| `10` | `-1` |
| `11` | `-3` |

接收端流程：

```text
IQ
  -> FM discriminator
  -> DC removal / low-pass
  -> sync-based polarity and phase recovery
  -> adaptive four-level slicing
  -> 264-bit burst
```

### 1.3 同步码

当前工程定义了四类 DMR 同步模板：

| 同步 | 长度 | 当前工程用途 |
|------|------|--------------|
| `BS_VOICE` | 24 symbols = 48 bits | 基站侧语音同步，Late Entry 语音超帧锚点 |
| `MS_VOICE` | 24 symbols = 48 bits | 终端侧语音同步，Late Entry 语音超帧锚点 |
| `DATA_BS` | 24 symbols = 48 bits | 基站侧数据/控制突发同步，LC Header/Terminator/CSBK |
| `DATA_MS` | 24 symbols = 48 bits | 终端侧数据/控制突发同步，LC Header/Terminator/CSBK |

同步码位于 132-symbol 突发的中心区域：

```text
burst symbol index: 0 ........................................ 131
sync symbol index:                      54 ........ 77
sync bit index:                         108 ....... 155
```

## 2. 帧结构

DMR 的一次语音呼叫可以按如下序列理解：

```text
Call / Transmission
    │
    ├─ Voice LC Header / Control Header
    │     建立呼叫，携带 Full Link Control
    │
    ├─ Voice Superframe × N
    │     每个 superframe = 同一 slot 上的 6 个语音突发，跨度 360 ms
    │     语音突发中夹带 Embedded Signalling，可用于 Late Entry
    │
    └─ Terminator with LC
          结束呼叫，并可再次携带 Full Link Control
```

当前工程的主要输出 PDU：

| PDU | 来源 | 当前含义 |
|-----|------|----------|
| `LC_HEADER` | Data Sync burst | 呼叫建立 LC |
| `TERMINATOR` | Data Sync burst | 呼叫结束 LC |
| `CSBK` | Data Sync burst | Control Signalling Block |
| `LATE_ENTRY` | Voice Sync / Embedded Signalling | 从语音超帧中重组出的 LC |

### 2.1 DMR 基本突发

DMR 基本突发为 132 symbols：

```text
DMR Burst = 132 symbols = 264 raw bits = 30 ms

┌────────────────────┬──────────────┬────────────────────┐
│ Info 1             │ Sync / EMB   │ Info 2             │
│ 54 symbols         │ 24 symbols   │ 54 symbols         │
│ 108 bits           │ 48 bits      │ 108 bits           │
└────────────────────┴──────────────┴────────────────────┘
```

其中中心 48 bit 的含义取决于突发类型：

| 突发类型 | 中心 48 bit |
|----------|-------------|
| Data Sync burst | 48 bit data sync |
| Voice Sync burst | 48 bit voice sync |
| 普通 voice burst | Embedded Signalling：16 bit EMB header + 32 bit signalling fragment |

### 2.2 Slot Type

数据/控制突发中，Slot Type 是最外层的类型字段：

```text
slot_type = burst_bits[98:108] + burst_bits[156:166]
slot_type length = 20 bits

┌────────────┬──────────────┬────────────────────┐
│ Color Code │ Data Type    │ Golay parity/check │
│ 4 bits     │ 4 bits       │ 12 bits            │
└────────────┴──────────────┴────────────────────┘
```

当前工程处理：

```text
slot_type
  -> Golay(20,8,7) check
  -> color_code = first 4 bits
  -> data_type = next 4 bits
```

当前工程识别的 `SlotDataType`：

| Data Type | 名称 | 当前工程处理 |
|-----------|------|--------------|
| 0 | `PI_HEADER` | 待补全 |
| 1 | `VOICE_LC_HEADER` | 解码为 `LC_HEADER` |
| 2 | `TERMINATOR_WITH_LC` | 解码为 `TERMINATOR` |
| 3 | `CSBK` | 解码为 `CSBK` |
| 4 | `MBC_HEADER` | 待补全 |
| 5 | `MBCC` | 待补全 |
| 6 | `DATA_HEADER` | 待补全 |
| 7 | `RATE_HALF` | 待补全 |
| 8 | `RATE_34` | 待补全 |
| 9 | `IDLE` | 待补全 |
| 10 | `RATE_1` | 待补全 |

### 2.3 Voice LC Header

Voice LC Header 是当前 DMR 元数据解码的主路径之一。结构如下：

```text
264-bit Data Sync burst
  ├─ Slot Type: 20 bits
  │    ├─ Color Code: 4 bits
  │    ├─ Data Type: 4 bits = VOICE_LC_HEADER
  │    └─ Golay parity/check: 12 bits
  │
  └─ Info field: 196 bits
       ├─ burst_bits[0:98]
       └─ burst_bits[166:264]
```

接收端解码链路：

```text
196-bit info field
  -> BPTC(196,96) deinterleave / repair
  -> 96 bits
  -> 12 bytes
  -> Reed-Solomon(12,9,4) check, mask = 0x969696
  -> Full Link Control
```

FLC 当前输出字段：

| 字段 | 当前工程字段名 | 说明 |
|------|----------------|------|
| Source Address | `src` | 源 ID |
| Group / Target Address | `dst` | 组呼 TGID 或个呼目标 ID |
| Full Link Control Opcode | `flco` | 例如 Group Voice Channel User |
| Feature Set ID | `fid` | 厂商/功能集 |
| Color Code | `extra.color_code` | 来自 Slot Type |
| Raw Burst | `raw_bits` | 原始 264-bit burst 打包 |

当前 PDU：

```text
type      = LC_HEADER
protocol  = DMR
src/dst   = Full Link Control 地址
flco/fid  = FLC 解析结果
extra     = color_code
raw_bits  = 264-bit burst
```

### 2.4 Voice Superframe / Late Entry

DMR 语音超帧由同一 TDMA slot 上的 6 个语音突发组成：

```text
Voice Superframe = 6 same-slot bursts = 360 ms

单个 burst 自身占 30 ms；
但同一 slot 的相邻语音突发每 60 ms 出现一次：
same-slot burst stride = 60 ms
当前工程样点间距 = 60 ms × 48 kHz = 2880 samples
```

在工程的 Late Entry 路径中，先用 Voice Sync 找到超帧锚点，再沿同一 slot 每 2880 个样点恢复后续语音突发：

```text
Voice Sync anchor
  -> lock voice phase
  -> recover burst j = 0..5
  -> adaptive_slice_bits()
  -> LateEntryCollector.feed()
```

普通语音突发中心 48 bit 的 Embedded Signalling 被拆为：

```text
center = burst_bits[108:156]

┌──────────────┬────────────────────────┬──────────────┐
│ EMB part A   │ Signalling fragment    │ EMB part B   │
│ 8 bits       │ 32 bits                │ 8 bits       │
└──────────────┴────────────────────────┴──────────────┘

emb_bits   = center[0:8] + center[40:48]   # 16 bits
signalling = center[8:40]                  # 32 bits
```

EMB header 中当前工程关心 `LCSS`：

| LCSS 状态 | 当前工程动作 |
|-----------|--------------|
| `FirstFragmentLC` | 开始收集 32-bit LC fragment |
| `ContinuationFragmentLCorCSBK` | 收集中间 fragment |
| `LastFragmentLCorCSBK` | 收集最后 fragment 并触发 VBPTC 解码 |

Late Entry LC 的解码链路：

```text
4 × 32-bit signalling fragment
  -> 128 bits
  -> VBPTC(128,72), include CS5
  -> 72-bit Full Link Control + 5-bit CS5
  -> FiveBitChecksum verify
  -> FullLinkControl.from_bits()
```

当前 PDU：

```text
type      = LATE_ENTRY
protocol  = DMR
src/dst   = Full Link Control 地址
flco/fid  = FLC 解析结果
extra     = cs5_ok
raw_bits  = 最后一个参与解码的 264-bit burst
```

### 2.5 Terminator with LC

Terminator with LC 的处理链路与 Voice LC Header 基本一致：

```text
Data Sync burst
  -> Slot Type: Data Type = TERMINATOR_WITH_LC
  -> Info field 196 bits
  -> BPTC(196,96)
  -> RS(12,9,4)
  -> Full Link Control
```

当前 PDU：

```text
type      = TERMINATOR
protocol  = DMR
src/dst   = Full Link Control 地址
flco/fid  = FLC 解析结果
extra     = color_code
raw_bits  = 264-bit burst
```

实时聚合路径可以使用 `TERMINATOR` 作为呼叫关闭条件。

### 2.6 CSBK

CSBK 使用 Data Sync burst：

```text
Data Sync burst
  -> Slot Type: Data Type = CSBK
  -> Info field 196 bits
  -> BPTC(196,96)
  -> CSBK.from_bits(decoded[0:96])
```

当前 PDU：

```text
type      = CSBK
protocol  = DMR
src/dst   = CSBK 地址字段
flco      = csbko.name
fid       = feature_set.name
extra     = color_code, last_block
raw_bits  = 264-bit burst
```

## 3. 数据组织过程：从语音 bit 到空口信号

这一部分按“发送端生成空口”的方向描述。接收端解码基本是反向过程。

### 3.1 总体流程

```text
业务输入
  ├─ 语音：PCM / 麦克风音频
  └─ 控制：源地址、目标地址、色码、FLCO、FID、呼叫控制

语音路径
  PCM
    -> AMBE / voice vocoder
    -> voice bits
    -> voice FEC / interleave
    -> 语音突发 payload

控制路径
  Full Link Control / CSBK
    -> FEC
    -> interleave
    -> Slot Type
    -> Data Sync burst

组帧
  Voice LC Header
  Voice Superframe × N
  Terminator with LC

调制
  dibits
    -> 4FSK symbol levels
    -> pulse shaping / FM deviation
    -> 12.5 kHz RF channel
```

### 3.2 LC Header 生成过程

Voice LC Header 的发送端组织顺序：

```text
1. 组织 72-bit Full Link Control

   flco / fid / service options / dst / src ...

2. 加 Reed-Solomon parity

   72-bit LC + 24-bit RS parity = 96 bits
   RS(12,9,4), Voice LC Header mask = 0x969696

3. BPTC(196,96)

   96 bits -> 196 bits

4. 组织 Slot Type

   color_code 4 bits
   data_type  4 bits = VOICE_LC_HEADER
   Golay(20,8,7) protection

5. 拼成 264-bit burst

   info[0:98] + slot_type[0:10] + sync[48] + slot_type[10:20] + info[98:196]

6. 每 2 bit 映射成 1 个 4FSK symbol

   264 bits -> 132 symbols -> 30 ms
```

接收端对应函数：

| 阶段 | 函数 |
|------|------|
| 同步检测 | `dmr.dsp.find_sync_positions()` |
| 突发恢复 | `dmr.dsp.recover_burst()` |
| 四电平判决 | `dmr.dsp.adaptive_slice_bits()` |
| Slot Type / Golay | `dmr.decoder.decode_burst()` |
| BPTC | `BPTC19696.deinterleave_data_bits()` |
| RS | `ReedSolomon1294.check()` |
| FLC | `FullLinkControl.from_bits()` |

### 3.3 Voice Superframe 生成过程

语音超帧可以理解为：

```text
Voice Burst A
Voice Burst B
Voice Burst C
Voice Burst D
Voice Burst E
Voice Burst F
```

每个 burst：

```text
voice payload + center sync/EMB + voice payload
  -> 264 bits
  -> 132 symbols
  -> 30 ms
```

Late Entry 所需的 LC 片段由多个普通语音突发中的 Embedded Signalling 拼出：

```text
EMB header
  -> LCSS: First / Continuation / Continuation / Last

signalling fragment
  -> 32 bits each

4 fragments
  -> 128 bits
  -> VBPTC(128,72) + CS5
  -> Full Link Control
```

### 3.4 接收端离线解码数据流

```text
rawiq file
  -> common.io.read_rawiq()
  -> scanner/dmr.offline 重采样到 48 kHz
  -> dmr.dsp.frontend()
       FM 鉴频
       低通滤波
       nominal deviation 归一化
  -> dmr.offline._decode_dmr_loop()
       find_sync_positions()
       Data Sync -> decode_burst()
       Voice Sync -> LateEntryCollector
  -> protocols.deduplicate_pdus()
  -> scanner 输出文本或 JSON
```

主要中间产物：

| 中间产物 | 类型 | 含义 |
|----------|------|------|
| `iq` | `np.ndarray[complex]` | 原始复基带 IQ |
| `iq_dec` | `np.ndarray[complex]` | 重采样到 48 kHz 的窄带 IQ |
| `y` | `np.ndarray[float]` | DMR 4FSK 鉴频输出 |
| sync position | `(center, polarity, sync_type)` | 同步中心、极性、同步类型 |
| recovered burst | `np.ndarray` | 132-symbol 校准后突发 |
| `ba264` | `bitarray` | 264-bit 突发 |
| `PDU dict` | `dict` | scanner 输出的统一协议结果 |

## 4. 当前工程覆盖范围与后续补全

### 4.1 已实现

| 能力 | 状态 |
|------|------|
| DMR 窄带前端 | 已实现 |
| Data/Voice Sync 检测 | 已实现 |
| 反极性检测 | 已实现 |
| 132-symbol 突发恢复 | 已实现 |
| Slot Type + Golay 校验 | 已实现 |
| Voice LC Header 解码 | 已实现 |
| Terminator with LC 解码 | 已实现 |
| CSBK 基础解码 | 已实现 |
| Late Entry LC 重组 | 已实现 |
| VBPTC(128,72) + CS5 校验 | 已实现 |
| 离线主入口和 DMR 单独入口 | 已实现 |

### 4.2 待补全

| 内容 | 说明 |
|------|------|
| AMBE 语音还原 | 当前只输出元数据，不解码音频 |
| PI Header / Data Header / Rate Data | `SlotDataType` 已列出，但未完整解析 |
| 双 slot 会话建模 | 当前 PDU 中 `ts` 固定为 0，需要后续按 TDMA slot 推断 |
| 更完整的呼叫聚合 | realtime 有 `CallRecord`，离线 scanner 仍以 PDU 列表为主 |
| 加密/隐私相关字段 | PI Header、语音加密参数等需要后续补齐 |
| 标准章节逐项索引 | 后续应把每个字段补上 ETSI 条款编号 |
