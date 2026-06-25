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

---

## 第二阶段:完整 LC 解码(达成 DMR 级会话效果)

- 日期:2026-06-25
- 状态:设计已通过评审,待写实现计划
- 前置:NID 级解码已完成(`p25/` 包 20 测试绿,`protocols.decode_all` 已集成 DMR+P25 双路径)。

### A. 第一阶段交付现状与缺口

真实样本 `data/p25_1_78125.rawiq`(19.5s)诊断:

- 帧同步检出 112 个候选(thr=0.62),DUID 分布 **LDU2=45 / LDU1=42 / HDU=4 / TDU=2** —— 合法 P25 通话序列。
- NAC=0x293 出现 84 次,跨帧强一致 —— 信号真实。
- `p25.decoder.decode()` 输出 112 个 NID PDU,但 `scanner.scan_file` 去重 key `(src,dst,type,fo_bucket)` 把它们**全压成 1 个**(P25 PDU 的 src/dst/type 全相同)。

缺口:① src/dst 恒为 0(未解 LC);② NID 无 BCH 纠错(`valid_bch=None`,弱帧无过滤);③ scanner 去重把 P25 多帧误删;④ 无会话组装(DMR 有 `LateEntryCollector`)。

### B. 已定决策(本阶段)

1. **目标深度:完整 LC** —— 解 LDU1 Link Control 拿真实 SrcID+TGID,并做会话组装(HDU→LDU→TDU 串成一次通话)。
2. **符号时钟:分段重同步** —— 每个 FS 锚点独立恢复本帧 1728 符号(360ms),不做跨帧 PLL(样本每帧都有 FS 可锚定)。
3. **验证基准:自洽性为主 + TIA-102 标准向量锁 FEC** —— 不强依赖外部 C/C++ 工具;DSD-FME/OP25 仅作一次性人工旁证。
4. **集成方式:P25 自己的会话层 + 协议感知去重** —— `p25/session.py` 组装会话;scanner 去重按 protocol 分派,P25 用 `(nac, type, fs_start/帧长)`,DMR 路径不变。
5. **FEC 策略:纯 Python 自实现** —— BCH/Hamming/RS/CRC 自实现,与 `okdmr` 风格一致、零新依赖,标准向量锁定。

### C. LDU1 LC 编码层级(TIA-102.BAAA)

72-bit LC → RS(24,12,13)/GF(2⁶) → 144 bit → Hamming(10,6,3) ×24 → 240 编码 bit → 交织散布在 1728 符号 LDU 帧中(与 9×IMBE 语音交错)。

- 字段:LCO(8) + MFID(8) + SrcID(24) + TGID/dst(16) + 其它。
- 精确交织表与 RS/Hamming 生成多项式:TIA-102.BAAA 为权威(非公开),实现对照 OP25 / DSD-FME / sdrtrunk 源码。

### D. 新增/改动清单

```
新增:
  p25/fec.py          BCH(63,16,23)、Hamming(10,6,3)、RS(24,12,13)/GF(2⁶)、CRC-16
  p25/link_control.py LDU1 LC 字段解析(LCO/MFID/SrcID/TGID/dst)
  p25/session.py      会话组装器(HDU→LDU→TDU → 一次通话 PDU)
改动:
  p25/constants.py    LDU 帧布局常量(1728符号、LC 交织表、各段偏移)
  p25/dsp.py          整帧 1728 符号恢复 + 去交织抽 LC
  p25/nid.py          接入 BCH(63,16) 真纠错,valid_bch 不再为 None
  p25/decoder.py      LDU1 走 LC 解码填 src/dst;接入 session
  scanner.py          去重改协议感知(P25 按 nac+fs_start),DMR 不变
```

### E. 数据流

```
y → find_frame_sync → recover_full_frame(1728符号)
  → decode_nid + BCH(63,16)         NAC+DUID 真纠错(质量闸门,过滤假帧)
     DUID==LDU1(0x5)?
       是 → deinterleave_lc(240 bit) → Hamming×24(144) → RS(72)
          → parse_link_control(LCO/MFID/SrcID/TGID) → session.feed
       否 → 只出 NID PDU(HDU/LDU2/TDU/TSBK)
  → session 收齐 HDU→…→TDU → 汇报一次通话(src/dst/duration/nac)
```

### F. 模块接口契约

```python
# p25/fec.py — 无状态纯函数,每码一份标准向量测试
bch_63_16_decode(bits64)   -> (bitarray|None, ok: bool)
hamming_10_6_3_decode(b10) -> (hexbit6: bitarray, corrected: bool)
rs_24_12_13_decode(hx24)   -> (bytes|None, ok: bool)   # GF(2⁶)
crc16_check(bits)          -> bool

# p25/link_control.py
@dataclass(frozen=True)
class LinkControl: lco; mfid; src; dst; tgid; is_group; raw
parse_link_control(lc72) -> LinkControl | None

# p25/session.py — 对齐 DMR LateEntryCollector.feed 风格
class P25SessionAssembler:
    feed(frame_info, link_control|None) -> dict | None   # TDU 时返回会话 PDU
    reset()
```

### G. 输出 PDU 形状(对齐 DMR)

```python
# LDU1 单帧(填真实 src/dst)
{"protocol":"P25","type":"P25_LDU1","src":<SrcID>,"dst":<TGID>,"ts":0,
 "flco":"GROUP","fid":"STANDARD",
 "extra":{"nac":0x293,"duid":0x5,"tgid":<TGID>,"rs_ok":True,"fs_start":...},
 "raw_bits":b"..."}
# 会话级(TDU 时汇报一次)
{"protocol":"P25","type":"P25_CALL","src":<SrcID>,"dst":<TGID>,
 "extra":{"nac":0x293,"duration_s":<秒>,"ldu_count":<帧数>}}
```
HDU/LDU2/TDU/TSBK 仍出 NID 级 PDU(src/dst=0)。

### H. scanner 协议感知去重(修复 112→1)

- DMR:保持原 key `(src,dst,type,fo_bucket)` 不变。
- P25:key 改为 `(nac, type, round(fs_start/帧长))` —— 按物理帧位置去重,不再因 src/dst 相同误删。

### I. 测试策略(TDD)

```
tests/test_p25_fec.py          BCH/Hamming/RS/CRC,TIA-102 标准向量(里程碑①闸门)
tests/test_p25_dsp.py          扩展:整帧 1728 恢复 + 去交织(合成向量)
tests/test_p25_link_control.py LC 字段解析(已知 72-bit LC → src/dst)
tests/test_p25_session.py      HDU→LDU→TDU 序列 → 一次会话 PDU
tests/test_p25_e2e.py          样本:NAC 一致、合法 DUID 序列、至少一条可解 LDU1 LC
```

### J. 交付里程碑(本阶段)

1. `p25/fec.py` + 单元测试全绿(TIA-102 标准向量锁定)。
2. 整帧符号恢复 + 去交织 + LC 抽取(内部先无纠错检错验证交织表,再接全 FEC)。
3. `link_control.py` + `session.py` + decoder/scanner 集成,样本 E2E 自洽。

### K. 风险

1. **交织表正确性(最高)** —— 1728 符号里 LC 的精确位置。缓解:里程碑②先用无纠错检错在干净帧上验证抽取的 240 bit 过 Hamming/RS syndrome 为 0。
2. **RS over GF(2⁶) 自实现** —— 生成多项式/根/移位易错。缓解:TIA-102 标准向量单测锁定,参考 `okdmr` RS 骨架。
3. **分段重同步漂移** —— 1728 符号末端定时误差。缓解:lstsq 用 FS 区定标,末端若超界则该帧 RS 失败自然丢弃。
