# TETRA 制式扩展实现方案

> 目标：在当前 DMR / P25 / dPMR 离线多制式框架下新增 TETRA 支持，优先识别和解码 MS-MS 语音通信期间同步产生的控制与数据内容；不解码语音内容，不做语音还原。

## 1. 当前前提

### 1.1 已确认可用的样本

当前本地已有两类 TETRA IQ 样本：

| 文件 | 判断 | 用途 |
|------|------|------|
| `data/TETRA IQ.wav` | 连续载波、四时隙功率接近，行为更像 TMO 基站下行 | 用于 TMO 下行同步、时隙定位、MCCH/业务下行验证 |
| `data/tetra_dmo_20240413_430050000_baseband.wav` | 间歇突发、空闲功率低、时隙能量有明显突发结构，来源标注为 DMO/simplex | 用于 DMO / MS-MS 方向的主要开发样本 |

因此 TETRA 扩展不再卡在样本不可用问题上。第一阶段以 DMO 样本为主，TMO 样本作为同步和模式识别的补充。

### 1.2 项目接入边界

当前离线主链路为：

```text
scanner.py
  -> common.io.read_rawiq()
  -> radio.pipeline.scan_iq()
  -> radio.registry.decode_iq_enabled()
  -> protocol SPEC.frontend()
  -> protocol SPEC.decode()
  -> protocol SPEC.postprocess()
  -> radio.registry.deduplicate_pdus()
```

新增 TETRA 时，应和现有三种制式一样，通过 `ProtocolSpec` 接入，不在 `scanner.py` 或 `radio.pipeline.py` 中写 TETRA 专用分支。

## 2. 范围定义

### 2.1 本阶段目标

TETRA 本阶段只输出元数据与伴随数据，目标 PDU 包括：

| PDU 类型 | 含义 |
|----------|------|
| `TETRA_DETECT` | 识别到 TETRA 载波或突发，输出模式候选、频偏、符号率质量 |
| `TETRA_SYNC` | 成功完成 π/4-DQPSK 符号同步和 slot/frame 定位 |
| `TETRA_BURST` | 已恢复一个 TETRA burst 的 dibit/raw bit，并标注 slot、frame、burst 类型 |
| `TETRA_DMO_CALL` | DMO/MS-MS 语音通信会话级摘要，包含可恢复的主叫/被叫/组呼/时隙/加密标志等 |
| `TETRA_DATA` | 随语音或独立出现的 SDS、GPS、状态、短数据等业务数据；无法完整解析时保留 raw payload |

输出仍使用 `radio.pdu.PDU` 的标准字段：

```text
protocol = "TETRA"
type     = TETRA_*
src      = 主叫 SSI / MS ID，未知则为空
dst      = 被叫 SSI / group ID，未知则为空
ts       = slot number，未知则为空
flco     = CALL / DATA / SYNC / CONTROL 等归类
extra    = TETRA 专用字段
raw_bits = 原始逻辑比特或 hex
```

### 2.2 非目标

本阶段明确不做：

1. ACELP / TETRA speech codec 解码。
2. 语音音频输出。
3. 加密内容解密。
4. 完整 TMO 集群跟踪器。
5. 完整 telive/osmo-tetra 级别的所有上层业务解析。

如果业务或短数据被加密，只输出加密标志、可见头字段和 raw payload。

## 3. 推荐文件结构

新增目录：

```text
tetra/
  __init__.py
  __main__.py
  cli.py
  config.py
  constants.py
  dsp.py
  sync.py
  burst.py
  mac.py
  session.py
  decode_flow.py
  plugin.py
```

职责划分：

| 文件 | 职责 |
|------|------|
| `config.py` | TETRA 参数、同步阈值、采样率、后处理门限 |
| `constants.py` | 符号率、slot/frame 长度、训练序列、burst 类型、逻辑信道常量 |
| `dsp.py` | π/4-DQPSK 前端、RRC 匹配滤波、频偏校正、定时恢复、dibit 判决 |
| `sync.py` | TMO/DMO 同步搜索、slot/frame 锚点、突发能量门控 |
| `burst.py` | burst 切片、训练序列校验、去扰/解交织/FEC 调用边界 |
| `mac.py` | MAC/LLC/SDS/GPS 等可见控制与数据字段解析 |
| `session.py` | 把 burst 级 PDU 聚合成 DMO call / data session |
| `decode_flow.py` | 单文件解码编排，返回 PDU list |
| `plugin.py` | `ProtocolSpec`、formatter、dedup key、postprocess |

## 4. 插件接入设计

`tetra/plugin.py` 按现有协议格式声明：

```python
SPEC = ProtocolSpec(
    "TETRA",
    ("tetra",),
    DEFAULT_TETRA_CONFIG,
    "tetra_pi4dqpsk",
    frontend,
    decode,
    postprocess,
    dedup_key,
    format_pdu,
)
```

然后在 `radio/registry.py` 中导入并加入：

```python
from tetra import plugin as tetra_plugin

PROTOCOL_REGISTRY = (
    dmr_plugin.SPEC,
    p25_plugin.SPEC,
    dpmr_plugin.SPEC,
    tetra_plugin.SPEC,
)
```

`scanner.py` 的 `--protocol` choices 需要增加 `tetra`。其它主流程不应增加 TETRA 专用逻辑。

## 5. 采样率策略

当前全局离线 pipeline 会把候选信道重采样到 48 kHz。TETRA 符号率为 18 ksym/s，48 kHz 对应 2.666 samples/symbol，不利于简单整数 SPS 处理。

为减少对现有协议的影响，第一版建议：

1. 保持 `radio.pipeline` 全局 48 kHz 行为不变。
2. TETRA 插件内部把输入 IQ 再重采样到 `72 kHz`，得到 `4 samples/symbol`。
3. 后续如果 TETRA 稳定后，再考虑把 `ProtocolSpec` 扩展出协议级 `target_sample_rate_hz`。

TETRA 默认参数建议：

| 参数 | 初值 |
|------|------|
| `symbol_rate_hz` | 18000 |
| `frontend_sample_rate_hz` | 72000 |
| `samples_per_symbol` | 4 |
| `slot_symbols` | 255 |
| `frame_symbols` | 1020 |
| `slot_duration_ms` | 14.1667 |
| `frame_duration_ms` | 56.6667 |
| `rrc_alpha` | 0.35 |
| `channel_cutoff_hz` | 12500 左右，结合样本调参 |

## 6. 分阶段实现

### Phase 0：样本与工具固定

目标：把可用样本和基本信号指标固定下来，避免后续算法调参没有基准。

实现内容：

1. 在测试辅助代码中读取 `data/TETRA IQ.wav` 和 DMO 样本。
2. 记录采样率、时长、峰值、均方幅度、频谱占用。
3. 增加一个轻量测试，确认 WAV IQ 读取稳定。
4. 建立 `tests/fixtures` 风格的短片段裁剪策略，避免完整样本拖慢单测。

验收：

```text
DMO 样本可读入，能检测到突发功率结构。
TMO 样本可读入，能检测到连续 TETRA 带宽和四时隙周期。
```

### Phase 1：TETRA 检测与模式判别

目标：先不解协议，只判断输入是不是 TETRA，以及更像 TMO 还是 DMO。

实现内容：

1. `tetra/dsp.py` 做中心频偏估计和粗频偏校正。
2. 基于功率包络检测：
   - 连续载波 + 4 slot 周期：倾向 TMO。
   - bursty + 空闲低功率：倾向 DMO。
3. 基于 18 ksym/s 的符号率特征做粗评分。
4. 输出 `TETRA_DETECT` PDU。

验收：

```text
python scanner.py "data/TETRA IQ.wav" --protocol tetra
  输出 TETRA_DETECT mode=TMO-like

python scanner.py data/tetra_dmo_20240413_430050000_baseband.wav --protocol tetra
  输出 TETRA_DETECT mode=DMO-like
```

### Phase 2：π/4-DQPSK 符号恢复

目标：从 IQ 恢复稳定 dibit 流。

实现内容：

1. RRC 匹配滤波。
2. 粗频偏和相位旋转补偿。
3. Gardner 或 Mueller-Muller 定时恢复。
4. π/4-DQPSK 差分解调，输出 dibit `0..3`。
5. 用已知训练序列或同步序列校准 dibit 极性/相位。

验收：

```text
对 DMO 样本可产生非空 dibit 流。
对 TMO 样本可产生连续 dibit 流。
同步相关峰值显著高于噪声段。
```

### Phase 3：slot / burst 定位

目标：按 TETRA slot 结构切出 burst，并确定 mode、slot、frame。

实现内容：

1. `tetra/sync.py` 搜索 DMO/TMO 同步序列或训练序列。
2. 根据 `255 symbols/slot` 建立 slot 锚点。
3. 在 TMO 连续样本中验证 `1020 symbols/frame` 周期。
4. 在 DMO 突发样本中用能量门限和同步相关共同切片。
5. 输出 `TETRA_SYNC` 和 `TETRA_BURST`。

验收：

```text
DMO 样本能稳定输出多个 TETRA_BURST。
TMO 样本能稳定输出 slot/frame 序列。
burst extra 中包含 slot、frame_start、sync_score、mode。
```

### Phase 4：控制与伴随数据解码

目标：只解析语音通信伴随的控制和数据，不解语音内容。

优先解析内容：

1. DMO call setup / call continuation / release 相关字段。
2. 主叫 SSI、被叫 SSI、组呼 ID、呼叫类型。
3. 时隙、帧号、加密/保密指示。
4. 与语音关联出现的短数据、状态、SDS、GPS/LIP 相关 payload。
5. 无法完整解释的 MAC/LLC payload 以 hex/raw bits 保留。

实现策略：

```text
TETRA_BURST
  -> burst 类型识别
  -> 控制区块提取
  -> 去扰 / 解交织 / FEC / CRC
  -> MAC/LLC/SDS 字段解析
  -> TETRA_DATA 或 TETRA_DMO_CALL
```

验收：

```text
不输出语音内容。
能输出至少一种 DMO call/session 级 PDU。
能在存在短数据时输出 TETRA_DATA；不存在时保留 burst raw metadata。
```

### Phase 5：会话聚合与去重

目标：和 DMR/P25/dPMR 一样，输出稳定、少重复、便于 JSON 消费的结果。

实现内容：

1. `tetra/session.py` 聚合同一呼叫的多个 burst。
2. `dedup_key()` 使用：
   - mode
   - src/dst
   - slot
   - call id 或 frame bucket
   - 频偏 bucket
3. `postprocess()` 过滤孤立低置信度 burst。
4. `format_pdu()` 输出人类可读摘要。

验收：

```text
同一 DMO 呼叫不会刷屏输出大量重复行。
JSON 中保留 burst 级 evidence 和 session 级摘要。
```

## 7. 输出示例

预期文本输出形态：

```text
[TETRA_DETECT] PROTO=TETRA MODE=DMO SLOTS=BURSTY SCORE=0.91
[TETRA_SYNC  ] PROTO=TETRA MODE=DMO SLOT=1 FRAME=42 SYNC=0.86
[TETRA_DMO_CALL] PROTO=TETRA CALL=GROUP SRC=123456 DST=789 SLOT=1 ENC=0 BURSTS=8
[TETRA_DATA  ] PROTO=TETRA KIND=SDS/GPS SRC=123456 DST=789 LEN=18 CRC=OK PAYLOAD=...
```

字段名初版可以保守：

```python
extra = {
    "mode": "DMO",
    "slot": 1,
    "frame": 42,
    "burst_type": "unknown/control/traffic",
    "sync_score": 0.86,
    "crc_ok": True,
    "encrypted": False,
    "payload_kind": "sds/gps/status/unknown",
}
```

## 8. 参考实现使用策略

本机已确认 `dsd-fme`、`dsd-neo` 对 TETRA 支持有限，不适合作为主要 TETRA 参考。

主要参考顺序：

1. ETSI TETRA Air Interface PDF：用于字段、FEC、逻辑信道定义校验。
2. osmo-tetra / telive：用于物理层、MAC 层和字段解析流程参考。
3. 当前项目 DMR/P25/dPMR：用于模块组织、PDU schema、去重、测试风格。

实现时需要注意开源许可边界。osmo-tetra 为 AGPL 系列代码时，不能直接复制大段实现；更适合用作协议行为参考，然后在本项目中独立实现必要算法。

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 48 kHz 全局重采样不适合 TETRA | 同步不稳 | TETRA 插件内部二次重采样到 72 kHz |
| DMO 样本数量少 | 字段解析泛化差 | 第一阶段只承诺 burst/sync 和 raw payload，后续继续补样本 |
| TETRA 上层字段复杂 | 解析周期长 | 先解析 call/data 最小字段，未知 payload 保留 raw bits |
| 语音与控制复用复杂 | 容易误解析语音块 | 明确跳过 speech payload，只处理可识别控制/数据区块 |
| 加密或扰码导致 payload 不可见 | 无法还原业务内容 | 输出 encrypted/unknown/raw，不做解密 |

## 10. 推荐实施顺序

1. 增加 `tetra/config.py`、`tetra/plugin.py`、CLI 注册和空 decoder，确认 `--protocol tetra` 可运行。
2. 实现 `TETRA_DETECT`，用两个现有样本建立 mode 判别测试。
3. 实现 π/4-DQPSK 前端和 dibit 恢复，先用合成信号单测，再跑真实样本。
4. 实现 slot/burst 同步，输出 `TETRA_SYNC` / `TETRA_BURST`。
5. 针对 DMO 样本做最小 MAC/伴随数据解析，输出 `TETRA_DMO_CALL` / `TETRA_DATA`。
6. 完善 session 聚合、去重和 JSON 输出。

这个顺序保证每一步都有可验证产物，不需要等完整 TETRA 协议栈实现完才看到结果。
