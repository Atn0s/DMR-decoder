# P25 Phase 1 元数据解码 — 设计文档

- 日期:2026-06-24
- 状态:已通过设计评审,待写实现计划
- 范围:在现有 DMR 解调项目基础上扩展 P25 Phase 1(C4FM/FDMA)信号识别与控制层元数据解码

## 1. 背景与前提

现有项目是一条干净的 DMR 解调链路:

```
read_rawiq → DDC → resample→48k → frontend(FM鉴频→4电平)
  → find_sync_positions(NCC相关) → recover_burst(相位扫描)
  → decode_burst / LateEntryCollector(FEC: BPTC/Golay/RS/VBPTC, okdmr 库)
```

关键事实:**DMR 用 4FSK,P25 Phase 1 用 C4FM——在 FM 鉴频器输出端两者几乎等价**(同为 4800 符号/秒、±1/±3 四电平、12.5kHz 信道、符号偏移 ±1800/±600Hz ≈ DMR ±1944/±648Hz)。因此现有 FM 鉴频前端可复用。

前提验证:用现有 `core.dsp.frontend` 跑 `data/p25_1_78125.rawiq`(78125Hz,19.5s),确认信号存在(载波偏移 ~858Hz)、鉴频输出为多电平 FSK 形态。干净 4 电平需在同步锚定后按符号率采样才显现(与 DMR 一致)。前端复用成立。

P25 Phase 1 无 `okdmr` 等价的成熟纯 Python 库(事实标准为 C/C++ 的 OP25、DSD-FME),故 FEC 采用纯 Python 自实现,与项目现有 `okdmr` 纯 Python 风格一致、无重依赖。

## 2. 范围

### 目标(控制层元数据)
- 每帧解出 NAC(网络接入码)+ DUID(帧类型)——即"识别"层。
- LDU1 的 Link Control:源单元 ID、目标 TGID、LCO/MFID。
- TSBK 中继信令:opcode + 参数(如信道授权)。

### 已定决策(不做)
- **不做 IMBE 语音解码 / 音频输出。**
- **不解密**;LDU2 加密同步(MI/ALGID/KID)按 YAGNI **暂不解析**,后续如需再加。
- 不做 P25 Phase 2(TDMA/CQPSK)——FM 前端不适用,另立 spec。

## 3. 架构:并列 `p25/` 包 + 协议分派

对现有 DMR 链路零侵入。新代码隔离在 `p25/`,复用 DSP 与实时框架。

### 复用(提为协议无关工具,DMR 行为不变)
- `core.dsp`:`read_rawiq`、DDC 下变频、`resample_poly`→48k、`frontend` FM 鉴频、`adaptive_slice` 思路、NCC 相关机制(必要时抽出通用 `ncc_find(y, template, threshold)`)。
- `realtime/`:`detector`、`channelizer`、`aggregator`、`ring_buffer`、`wideband_*`——调制无关,全部复用。
- `scanner`:能量检测、PSD 盲搜、`_print_results`/`_write_json` 输出。

### 新建 `p25/` 包
| 文件 | 职责 |
|------|------|
| `p25/sync.py` | P25 帧同步字 `0x5575F5FF77FF`(48 bit / 24 符号)NCC 检测;帧头定位 |
| `p25/dsp.py` | 同步锚定的**连续符号恢复**(P25 帧为连续 bit 流,非 DMR 132 符号 burst)+ 状态符号去交织 |
| `p25/fec.py` | 纯 Python FEC:BCH(63,16,23)、Golay(24,12,8)、Hamming(10,6,3)/(15,11,3)、Reed-Solomon(24,12,13)/(24,16,9)/(36,20,17) over GF(2⁶)、1/2-rate 网格(Viterbi)、CRC-16 |
| `p25/framing.py` | NID 解析(NAC+DUID)、HDU/LDU1/LDU2/TDU/TDULC/TSBK 帧状态机 |
| `p25/decoder.py` | 输出统一 PDU dict,字段形状对齐 DMR(`{type, src, dst, flco, fid, extra, ...}`),复用 scanner 打印/JSON |

## 4. 数据流

```
candidate IQ → DDC → resample 48k → frontend(鉴频) → y
  → 协议分派: ncc(y, DMR模板)  vs  ncc(y, P25_FS模板)
       命中 P25 → p25.sync 定位帧头(FS+NID)
                → p25.dsp 连续符号恢复 + 去交织
                → p25.framing 按 DUID 解帧
                → p25.fec 纠错(BCH/RS/Hamming/Golay/trellis/CRC)
                → p25.decoder 出 PDU(NAC / DUID / LDU1-LC / TSBK)
```

## 5. P25 Phase 1 帧与 FEC 对照

| 帧 (DUID) | 内容 | FEC |
|-----------|------|-----|
| FS | 48-bit 帧同步 `0x5575F5FF77FF` | — |
| NID | 12-bit NAC + 4-bit DUID | BCH(63,16,23) + parity |
| HDU (0x0) | 头(含 TGID/MI/ALGID) | Golay(18,6,8) + RS(36,20,17) |
| LDU1 (0x5) | 9×IMBE(忽略) + Link Control | RS(24,12,13) + Hamming(10,6,3) |
| LDU2 (0xA) | 9×IMBE(忽略) + 加密同步(暂不解析) | RS(24,16,9) + Hamming(10,6,3) |
| TDU (0x3) / TDULC (0xF) | 终止符 | (TDULC: RS(24,12,13)) |
| TSBK (0x7) | 中继信令块 | 1/2-rate 网格 + CRC-16 |

DUID 编码:HDU=0x0, TDU=0x3, LDU1=0x5, TSBK=0x7, LDU2=0xA, PDU=0xC, TDULC=0xF。

## 6. scanner / realtime 集成

- `scanner._decode_loop`:在生成 `y` 后增加协议探测分支——同一段 `y` 上分别跑 DMR 同步 NCC 与 P25 帧同步 NCC,命中者分派到对应解码器。**DMR 路径代码完全不动。**
- `realtime/worker.decode_window`:`frontend` 之后同样做双协议尝试。
- 输出复用现有 `_print_results` / `_write_json`,P25 PDU 填同样字段。

## 7. 会话组装

DMR 有 `LateEntryCollector` 把语音 burst 串起来。P25 类似地需要一个会话组装器把 `HDU → LDU1/LDU2 …(交替)→ TDU` 串成一次"通话",汇报一次 src/TGID/时长。放在 `p25/decoder.py`(离线)与 `realtime/aggregator`(实时)。

## 8. 关键差异 / 风险

1. **符号恢复机制不同**:DMR 是固定 132 符号 burst + 固定 stride 步进;P25 是同步锚定后**连续符号时钟跟踪**整帧。判决思路(sync 区 lstsq 定标 + 最近电平判决)可借鉴,帧布局逻辑需重写。
2. **验证基准(最大风险)**:`p25_1_78125.rawiq` 无已知真值。采用**自洽性验证**:NAC 跨帧一致、DUID 序列合法(HDU→LDU 交替→TDU)、RS/CRC 校验通过率、LC 字段取值合理;尽量用 DSD-FME / OP25 输出作旁证 oracle。
3. **FEC 正确性**:每个码用 TIA-102 标准测试向量做单元测试。GF(2⁶) Reed-Solomon 与 1/2-rate 网格译码是自实现中最易出错处,优先用向量锁定。

## 9. 测试策略(TDD)

镜像现有 `tests/` 结构,全程 TDD——先写测试与已知向量,再实现:
- `tests/test_p25_fec.py`:BCH/Golay/Hamming/RS/trellis/CRC,用标准测试向量。
- `tests/test_p25_sync.py`:合成 + 样本上的帧同步检出与误检。
- `tests/test_p25_framing.py`:NID 解析、DUID 分派、LC/TSBK 字段。
- `tests/test_p25_e2e.py`:跑 `data/p25_1_78125.rawiq`,断言稳定 NAC + 合法 DUID 序列 + 至少一条可解 LC/TSBK。

## 10. 交付里程碑(供实现计划细化)

1. `p25/fec.py` + 单元测试全绿(标准向量)。
2. `p25/sync.py` 帧同步检出 + `p25/dsp.py` 符号恢复。
3. `p25/framing.py` + `p25/decoder.py`:NID/LDU1-LC/TSBK 解码,样本 E2E 自洽。
4. scanner / realtime 协议分派集成,回归确认 DMR 不受影响。
