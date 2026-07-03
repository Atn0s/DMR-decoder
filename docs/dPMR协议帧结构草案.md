# dPMR 协议帧结构说明（草案）

> 版本：v0.1  
> 适用范围：当前工程中的 dPMR 离线解码链路，以及后续编写 DMR/P25 协议文档时的格式模板。  
> 主要代码参考：`dpmr/constants.py`、`dpmr/dsp.py`、`dpmr/cch.py`、`dpmr/decoder.py`、`dpmr/session.py`。  
> 标准参考方向：ETSI TS 102 658 / TS 102 490。本文先按当前工程已实现和已验证的结构展开，未在工程中完整解码的业务帧会标注为“待补全”。

## 1. 基本参数

### 1.1 空口与物理层参数

| 项目 | 数值 / 说明 |
|------|-------------|
| 协议名称 | dPMR（digital Private Mobile Radio） |
| 频点 / 频段 | 协议本身不绑定固定 RF 频点；工程中的 `.rawiq` 是复基带文件，只保留相对中心频偏。实际系统可运行在授权 PMR 频段；dPMR446 属于 446 MHz 附近的免执照应用形态 |
| 典型信道带宽 | 6.25 kHz |
| 多址方式 | FDMA，单载波单信道，不使用 DMR 那种 2-slot TDMA |
| 调制方式 | 4FSK / 4-level FSK |
| 符号率 | 2400 symbols/s |
| 每符号承载 | 2 bit / symbol |
| 原始比特率 | 4800 bit/s |
| 当前工程解码采样率 | 48 kHz |
| 当前工程每符号采样点 | 20 samples/symbol |
| 当前工程前端低通 | 约 3.5 kHz |
| 当前工程 nominal deviation | 约 1050 Hz，用于把 dPMR 外层 `±3` 频偏归一化到符号电平 |
| 基本帧时长 | 160 ms |
| 基本帧长度 | 384 symbols = 768 raw bits |
| 色码范围 | 0-63，当前工程从 24 bit Channel Code 中恢复 |
| 当前工程已实现同步 | FS1、FS2 同步检测；FS3、FS4 常量已定义但业务语义未完全下沉 |

### 1.2 当前工程使用的符号映射

当前工程把空口 4FSK 判决后的符号表示为 `0/1/2/3` 四个 dibit，并映射成两个 bit：

| 符号值 | Dibit | 当前工程判决电平 |
|--------|-------|------------------|
| 0 | `00` | `+1` |
| 1 | `01` | `+3` |
| 2 | `10` | `-1` |
| 3 | `11` | `-3` |

接收端处理时先做 FM 鉴频，再通过同步码做仿射校准，把连续幅度拉回到 `+1/+3/-1/-3` 四个电平附近，最后判决为 dibit。

### 1.3 同步码

当前工程定义了四类 dPMR Frame Sync，并同时支持反极性同步检测：

| 同步 | 长度 | 当前工程用途 | 说明 |
|------|------|--------------|------|
| FS1 | 24 symbols = 48 bits | Header frame 定位 | `DPMR_HEADER` 解码入口 |
| FS2 | 12 symbols = 24 bits | Voice frame 定位 | `DPMR_VOICE` 解码入口 |
| FS3 | 12 symbols = 24 bits | 待补全 | 常量已定义，业务帧未完整解析 |
| FS4 | 24 symbols = 48 bits | 待补全 | 常量已定义，可能用于其他控制/结束类帧，需要按标准和样本补齐 |

当前解码器用归一化互相关定位同步码，并记录：

```text
sync_type            FS1 / FS2
polarity_inverted    是否反极性
sync_ncc             互相关峰值
fs_start             同步码起始样点
```

## 2. 帧结构

dPMR 的基本可重复空口单元可以按“呼叫序列”理解：

```text
Call / Transmission
    │
    ├─ Header Frame
    │     建立或声明一次业务传输，携带 CCH、色码、链路信息等控制内容
    │
    ├─ Voice / Traffic Superframe  × N
    │     周期性承载语音 TCH，同时夹带 CCH 慢速控制信息
    │
    └─ End / Terminator Frame
          结束业务传输；当前工程尚未完整解析
```

当前工程最稳定的可观测结构是：

```text
FS1 Header Frame      384 symbols / 160 ms
FS2 Voice Frame       384 symbols / 160 ms
FS2 Voice Frame       384 symbols / 160 ms
...
```

其中 CCH 的 `frame_number` 为 2 bit，取值 `0/1/2/3`。工程把 `0+1` 组合为目标地址片段，把 `2+3` 组合为源地址片段。因此在文档和调试中，可以把 `frame_number=0..3` 的一次循环视作当前解码器能观察到的 dPMR 控制 superframe。

### 2.1 Header Frame（FS1）

Header Frame 以 FS1 开始，总长度为 384 symbols：

```text
Header Frame = 384 symbols = 768 raw bits = 160 ms

┌────────────────────────┬─────────────────────────────────────────────┐
│ FS1                    │ Header Payload                               │
│ 24 symbols / 48 bits   │ 360 symbols / 720 bits                       │
└────────────────────────┴─────────────────────────────────────────────┘
```

当前工程的 Header 解析策略：

```text
FS1 定位
  -> 恢复整帧 384 symbols
  -> 去掉 FS1，得到 360-symbol payload
  -> 以 36-symbol 窗口搜索 CCH
  -> 以 12-symbol 窗口搜索 Color Code
  -> 根据 CCH CRC/Hamming 和 Color Code 给候选打分
```

Header Payload 中当前工程关注的结构单元：

| 单元 | 长度 | 原始比特 | 当前工程处理 |
|------|------|----------|--------------|
| CCH | 36 symbols | 72 bits | `decode_cch()` 解扰、解交织、Hamming、CRC7 |
| Color Code / Channel Code | 12 symbols | 24 bits | `get_color_code()` 从 24 bit 通道码映射出 0-63 色码 |
| 其他 Header 业务字段 | 剩余 payload | 待补全 | 当前工程未逐字段解析 |

Header 的当前 PDU 输出：

```text
type      = DPMR_HEADER
flco      = HEADER
src/dst   = 仅在 CCH 高置信度时暴露
extra     = color_code, sync_type, polarity_inverted, cch, frame_numbers, quality...
raw_bits  = Header payload 的原始 bit 打包结果
```

### 2.2 Voice Frame（FS2）

当前工程对 FS2 Voice Frame 的结构切分是明确的：

```text
Voice Frame = 384 symbols = 768 raw bits = 160 ms

┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│ FS2        │ CCH0       │ TCH0       │ Color Code │ CCH1       │ TCH1       │
│ 12 sym     │ 36 sym     │ 144 sym    │ 12 sym     │ 36 sym     │ 144 sym    │
│ 24 bits    │ 72 bits    │ 288 bits   │ 24 bits    │ 72 bits    │ 288 bits   │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

长度校验：

```text
12 + 36 + 144 + 12 + 36 + 144 = 384 symbols
24 + 72 + 288 + 24 + 72 + 288 = 768 bits
```

当前工程只输出语音帧中的元数据，不做 AMBE/声码器还原：

| 单元 | 长度 | 内容 | 当前工程处理 |
|------|------|------|--------------|
| FS2 | 12 symbols / 24 bits | 语音帧同步 | NCC 定位、极性判断、样点相位估计 |
| CCH0 | 36 symbols / 72 bits | 慢速控制信道片段 | 解码为 `CCHRecord` |
| TCH0 | 144 symbols / 288 bits | 语音业务信道 | 当前暂不解析，仅在结构上跳过 |
| Color Code | 12 symbols / 24 bits | 信道色码编码 | 映射为 0-63 色码 |
| CCH1 | 36 symbols / 72 bits | 慢速控制信道片段 | 解码为 `CCHRecord` |
| TCH1 | 144 symbols / 288 bits | 语音业务信道 | 当前暂不解析，仅在结构上跳过 |

Voice Frame 的当前 PDU 输出：

```text
type      = DPMR_VOICE
flco      = VOICE
src/dst   = 仅在 CCH 高置信度且拼出完整地址时暴露
extra     = color_code, cch[0..1], frame_numbers, superframe_part, quality...
raw_bits  = 当前只保存 Color Code 区域的原始 bit
```

### 2.3 Control Channel Header（CCH）

CCH 是当前工程中 dPMR 元数据解码的核心结构。空口中每个 CCH 为：

```text
CCH air block = 36 symbols = 72 raw bits
```

接收端解码流程：

```text
72 raw bits
  -> descramble，9-bit LFSR 初值 0x1FF
  -> 6 x 12 deinterleave
  -> 6 个 Hamming(12,8) codeword
  -> 48 data bits
  -> CRC7 校验前 41 bits
```

#### 2.3.1 CCH 空口保护结构

```text
72 raw bits
┌──────────────────────────────────────────────────────────────┐
│ scramble + 6x12 interleave 后的 6 个 Hamming(12,8) codeword   │
└──────────────────────────────────────────────────────────────┘

deinterleave 后：

┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ H(12,8) #0   │ H(12,8) #1   │ H(12,8) #2   │ H(12,8) #3   │ H(12,8) #4   │ H(12,8) #5   │
│ 12 bits      │ 12 bits      │ 12 bits      │ 12 bits      │ 12 bits      │ 12 bits      │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
        │              │              │              │              │              │
        └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
                         Hamming 解码后得到 6 x 8 = 48 data bits
```

#### 2.3.2 CCH 信息字段

当前工程解析出的 48 bit CCH data：

```text
┌──────────────┬──────────────┬────────────────────┬──────────┬──────────────┬────────────┬──────────┬──────────────┬──────────┐
│ Frame Number │ ID Half      │ Communication Mode │ Version  │ Comms Format │ Emergency  │ Reserved │ Slow Data    │ CRC7     │
│ 2 bits       │ 12 bits      │ 3 bits             │ 2 bits   │ 2 bits       │ 1 bit      │ 1 bit    │ 18 bits      │ 7 bits   │
└──────────────┴──────────────┴────────────────────┴──────────┴──────────────┴────────────┴──────────┴──────────────┴──────────┘
bit index:
  0..1          2..13         14..16              17..18    19..20        21          22        23..40       41..47
```

字段说明：

| 字段 | bit 宽度 | 当前工程字段名 | 含义 |
|------|---------|----------------|------|
| Frame Number | 2 | `frame_number` | CCH 片段编号，当前用于重组地址：`0/1` -> dst，`2/3` -> src |
| ID Half | 12 | `id_half` | 24 bit Air Interface ID 的半段 |
| Communication Mode | 3 | `communication_mode` | 通信模式，当前只透出数值 |
| Version | 2 | `version` | 协议/业务版本，当前只透出数值 |
| Comms Format | 2 | `comms_format` | 通信格式，当前只透出数值 |
| Emergency Priority | 1 | `emergency_priority` | 紧急/优先级标记 |
| Reserved | 1 | `reserved` | 保留位 |
| Slow Data | 18 | `slow_data` | 慢速数据，当前只透出数值 |
| CRC7 | 7 | `crc_value` / `crc_computed` | 校验前 41 bit 的 CRC7 |

地址重组：

```text
dst_ai_id = CCH(frame_number=0).id_half << 12
          | CCH(frame_number=1).id_half

src_ai_id = CCH(frame_number=2).id_half << 12
          | CCH(frame_number=3).id_half
```

工程输出时会把 24 bit AI ID 转为 7 位字符形式。转换逻辑在 `air_interface_id_to_str()` 中，通过 dPMR AI ID 权重表逐位展开，数字 `10` 显示为 `*`：

```text
24 bit AI ID -> digit/digit/digit/digit/digit/digit/digit
digit 0..9   -> "0".."9"
digit 10     -> "*"
```

### 2.4 Color Code / Channel Code

Color Code 区域长度为：

```text
12 symbols = 24 raw bits
```

当前工程处理方式：

```text
24 raw bits
  -> bits_to_int()
  -> OR 0x555555
  -> 查 COLOR_CODE_BY_CHANNEL_CODE
  -> color_code: 0..63，失败为 -1
```

Color Code 的作用类似 DMR 的 Color Code / P25 的 NAC：它用于区分同频或邻近系统，使接收机只接受匹配系统的帧。当前 scanner 还会用重复出现的 dPMR Color Code 做稳定性过滤：

```text
filter_stable_pdus()
  -> 统计 dPMR PDU 中重复出现的 color_code
  -> 优先保留重复且质量高的色码
  -> 降低误同步和旁瓣候选造成的假 PDU
```

### 2.5 Traffic Channel（TCH）

每个 FS2 Voice Frame 中有两个 TCH：

```text
TCH0 = 144 symbols = 288 raw bits
TCH1 = 144 symbols = 288 raw bits
```

当前工程状态：

| 项目 | 状态 |
|------|------|
| TCH 位置切分 | 已明确 |
| TCH 原始 bit 提取 | 可从符号流得到，但当前未作为 PDU 输出 |
| 声码器帧解析 | 待补全 |
| AMBE/语音还原 | 待补全 |
| 与 CCH 慢速信令的跨帧同步 | 当前只做 CCH 地址片段重组 |

后续如果要实现语音，还需要补齐：

```text
TCH symbols
  -> dibit stream
  -> TCH 内部去交织 / FEC / voice frame 拆分
  -> AMBE voice bits
  -> AMBE decoder
  -> PCM audio
```

### 2.6 End / Terminator Frame（待补全）

当前工程尚未把 dPMR End / Terminator 作为独立 PDU 输出。因此本文只给出目标文档形态：

```text
End Frame
┌──────────────┬────────────────────────────────────────────┐
│ Frame Sync   │ End / Release / Control Payload             │
│ FS?          │ 业务释放、终止原因、可能的 CCH 或校验字段      │
└──────────────┴────────────────────────────────────────────┘
```

后续需要补齐的内容：

| 待补项 | 说明 |
|--------|------|
| End 使用的同步类型 | 需要按 ETSI 标准和真实样本确认 FS3/FS4 的业务含义 |
| End payload 字段 | 需要确认是否复用 CCH、是否有专门终止原因字段 |
| FEC / CRC | 需要确认 End 控制字段的保护方式 |
| Session 关闭条件 | 后续可让 `dpmr.session` 在收到 End 后输出完整 `DPMR_CALL` |

## 3. 数据组织过程：从语音 bit 到空口信号

这一部分按“发送端生成空口”的方向描述。接收端解码基本是反向过程。

### 3.1 总体流程

```text
业务输入
  ├─ 语音：PCM / 麦克风音频
  └─ 控制：源地址、目标地址、通信模式、色码、慢速数据、结束标记

语音路径
  PCM
    -> 声码器编码
    -> TCH voice bits
    -> TCH FEC / 交织
    -> TCH dibits

控制路径
  CCH fields
    -> CRC7
    -> 6 x Hamming(12,8)
    -> 6x12 interleave
    -> LFSR scramble
    -> CCH dibits

组帧
  FS1 Header frame
  FS2 Voice frame: FS2 + CCH0 + TCH0 + Color Code + CCH1 + TCH1
  End frame

调制
  dibits
    -> 4FSK symbol levels
    -> pulse shaping / FM deviation
    -> 6.25 kHz RF channel
```

### 3.2 CCH 生成过程

CCH 的发送端组织顺序：

```text
1. 组织 41 bit CCH payload

   frame_number       2 bits
   id_half           12 bits
   communication      3 bits
   version            2 bits
   comms_format       2 bits
   emergency          1 bit
   reserved           1 bit
   slow_data         18 bits

2. 计算 CRC7

   crc7(CCH payload[0:41]) -> 7 bits

3. 拼成 48 bit CCH data

   41 payload bits + 7 crc bits = 48 bits

4. 分成 6 个 8-bit block

   block0..block5

5. 每个 block 做 Hamming(12,8)

   6 x 8 bits -> 6 x 12 bits = 72 bits

6. 6x12 interleave

   把连续错误扩散到多个 Hamming codeword

7. LFSR scramble

   9-bit LFSR 初值 0x1FF

8. 每 2 bit 映射成 1 个 4FSK symbol

   72 bits -> 36 symbols
```

接收端对应函数：

| 阶段 | 函数 |
|------|------|
| 解扰 | `dpmr.cch.descramble()` |
| 解交织 | `dpmr.cch.deinterleave_6x12()` |
| Hamming 解码 | `dpmr.cch.hamming_12_8_decode()` |
| CRC7 | `dpmr.cch.crc7()` |
| 字段解析 | `dpmr.cch.decode_cch()` |

### 3.3 Color Code 生成过程

当前工程只实现接收端反查，文档可按反向理解发送端：

```text
color_code: 0..63
  -> 标准定义的 24 bit channel code
  -> 12 symbols
```

接收端：

```text
12 symbols
  -> 24 bits
  -> channel_code = bits_to_int(bits) | 0x555555
  -> 查表得到 color_code
```

### 3.4 Voice Frame 组装过程

一个当前工程可识别的 FS2 Voice Frame：

```text
CCH0 fields
  -> CCH0 coded bits
  -> 36 symbols

TCH0 voice bits
  -> voice FEC/interleave
  -> 144 symbols

Color Code
  -> 12 symbols

CCH1 fields
  -> CCH1 coded bits
  -> 36 symbols

TCH1 voice bits
  -> voice FEC/interleave
  -> 144 symbols

FS2 + CCH0 + TCH0 + Color Code + CCH1 + TCH1
  -> 384 symbols / 160 ms
```

CCH 片段跨帧循环：

```text
frame_number=0  -> dst high 12 bits
frame_number=1  -> dst low 12 bits
frame_number=2  -> src high 12 bits
frame_number=3  -> src low 12 bits
```

因此接收端可以在多个 Voice Frame 中逐步积累 CCH：

```text
FS2 frame #k
  CCH0/CCH1 -> maybe frame_number 0/1
  -> 拼出 dst

FS2 frame #k+1
  CCH0/CCH1 -> maybe frame_number 2/3
  -> 拼出 src
```

当前工程对应：

```text
dpmr.decoder.decode()
  -> find_fs2_sync()
  -> recover_voice_fs2_symbol_candidates()
  -> split_voice_fs2()
  -> decode_cch(cch0_bits)
  -> get_color_code(cc_bits)
  -> decode_cch(cch1_bits)
  -> DPMRSessionAssembler.feed(cch0, cch1)
```

### 3.5 接收端离线解码数据流

从 IQ 文件到 dPMR PDU 的当前工程数据流：

```text
rawiq file
  -> common.io.read_rawiq()
  -> scanner 重采样到 48 kHz
  -> dpmr.dsp.frontend_dpmr()
       FM 鉴频
       低通滤波
       nominal deviation 归一化
  -> dpmr.decoder.decode()
       FS1/FS2 同步搜索
       符号相位 / SPS 搜索
       4FSK 四电平判决
       Header / Voice frame 切分
       CCH 解码
       Color Code 解码
       src/dst 片段重组
  -> filter_stable_pdus()
  -> scanner 输出文本或 JSON
```

主要中间产物：

| 中间产物 | 类型 | 含义 |
|----------|------|------|
| `iq` | `np.ndarray[complex]` | 原始复基带 IQ |
| `iq_dec` | `np.ndarray[complex]` | 重采样到 48 kHz 的窄带 IQ |
| `y_dpmr` | `np.ndarray[float]` | dPMR 4FSK 鉴频输出 |
| `DPMRSyncCandidate` | dataclass | 同步位置、同步类型、极性、NCC |
| `DPMRSymbolCandidate` | dataclass | 384-symbol 候选、SPS、相位、残差、判决质量 |
| `CCHRecord` | dataclass | 48 bit CCH 字段、CRC/Hamming 状态、纠错位数 |
| `DPMRSessionAssembler` | class | 累积 `frame_number=0..3` 的 CCH，拼出 src/dst |
| `PDU dict` | `dict` | scanner 输出的统一协议结果 |

## 4. 当前工程覆盖范围与后续补全

### 4.1 已实现

| 能力 | 状态 |
|------|------|
| dPMR 窄带前端 | 已实现 |
| FS1/FS2 同步检测 | 已实现 |
| 反极性同步检测 | 已实现 |
| 384-symbol 帧恢复 | 已实现 |
| FS1 Header CCH/Color Code 搜索 | 已实现 |
| FS2 Voice CCH/Color Code 切分 | 已实现 |
| CCH 解扰、解交织、Hamming、CRC7 | 已实现 |
| Color Code 0-63 恢复 | 已实现 |
| src/dst AI ID 片段重组 | 已实现 |
| dPMR 稳定色码过滤 | 已实现 |

### 4.2 待补全

| 内容 | 说明 |
|------|------|
| FS3/FS4 的业务帧语义 | 常量已定义，但未形成独立 PDU |
| End / Terminator 帧 | 尚未解析，无法基于 dPMR 自身终止帧关闭 session |
| Header payload 完整字段 | 当前只搜索 CCH 和 Color Code，未逐字段解析剩余控制字段 |
| TCH 语音内容 | 当前只定位并跳过，未输出 voice raw bits |
| 声码器 / AMBE | 尚未实现 |
| 完整 DPMR_CALL 聚合 | 当前输出 Header/Voice 元数据，未聚合成完整呼叫记录 |
| 标准章节逐项索引 | 后续应把每个字段补上 ETSI 条款编号 |

## 5. 建议的后续文档格式

后续写 DMR、P25 时可以复用同一结构：

```text
1. 基本参数
   - 带宽、调制、多址、符号率、比特率、帧时长、同步类型

2. 帧结构
   - 完整呼叫序列
   - Header
   - Superframe / Voice frame
   - End / Terminator
   - 每个模块的 bit 宽度、FEC、CRC、交织、加扰

3. 数据组织过程
   - 语音 bit / 控制字段
   - FEC
   - 交织
   - 加扰
   - 同步插入
   - 调制成空口信号

4. 工程覆盖范围
   - 已实现
   - 待补全
   - 对应代码入口
```
