# DMR 跳频扫描（全频段覆盖）设计

**日期：** 2026-06-18
**范围：** 在现有 `realtime/` 实时层之上新增**调谐层**，使单台接收机能在一段超过其瞬时带宽的总频段内跳频扫描，发现并解析所有 DMR 通话。本期仍用仿真验证，硬件接入仅替换数据源。
**不在范围内：** SDR 硬件实测（`SoapyTunableSource` 占位）；M-of-N 开门确认与 P25 负缓存（接口预留，留待硬件阶段填充）；语音解码（沿用 `CallRecord.voice_raw` 占位接口）。

---

## 1. Context（为什么做这个）

### 1.1 现状的根本限制

现有 `RealtimeScanner` 锁定在**单一固定带宽**上，从流开始跑到流结束，全程只盯着一个频段，没有任何换频/扫频逻辑（grep 验证：无 retune/sweep 代码）。瞬时带宽 = 复采样率（Nyquist），一旦真实 DMR 频段宽度超过单次可承载的瞬时带宽，固定带宽就**物理上看不全**。

### 1.2 目标

单台接收机在一段可配置的总频段（可宽于瞬时带宽）内**持续巡视**，自动发现活跃 DMR 通话、解析信令（LC Header / CSBK / Terminator / Late Entry）、输出带**绝对射频频率**的结构化通话记录。先用仿真验证整条链路，硬件接入时仅替换 `TunableIQSource` 实现。

### 1.3 跳频引入的两个新难题

1. **时间与频率耦合**：单台接收机不能同时身处两个子带。调谐在子带 B 时，子带 A 在那段时间是**盲区**——这段时间永久错过，不可回取。仿真必须忠实复现这一点，否则就是"作弊"（接收机看到了它物理上不该看到的东西）。
2. **"没看到"有两种原因**：信道没能量，可能是"信号真结束"，也可能是"接收机当时没调谐过去（盲区）"。现有 `CLOSE_HYSTERESIS` 假设"每窗都在看"，跳频下会误杀盲区里的活跃通话。

### 1.4 缓解跳频代价的既有杠杆（复用，不重建）

- **Late Entry**：DMR 通话连续发数个超帧（每 360ms 一个），盲区错过的超帧可在下次驻守回来时从后续语音帧重建 LC。目标从"抓住通话起点"放松为"通话存活期内任意一次驻守命中即可"。
- **超时兜底** `CALL_TIMEOUT_WINDOWS`：盲区里通话其实已结束、却永远挂着——超时关闭兜底。

---

## 2. 全局约束

- 解调核心 `core/dsp.py`、`core/decoder.py`、`core/burst_type.py` **不修改**。
- `scanner._decode_loop`、`realtime/worker.decode_window`、`multiprocessing.Pool` 派发**复用**，不重写。
- 采样率/带宽不绑定具体值——总频段与瞬时带宽均为运行时可配参数。
- **物理下限（诚实声明）**：一个完整 DMR 超帧 = 360ms（`BURST_STRIDE=2880` 样点 @48kHz × 6 帧）。Voice Sync 只在 Burst A 出现，因此**保证抓到一次 Voice Sync 的最短驻守 = 360ms**；为给 Late Entry 4 片段收集留余量，CAMP 默认驻守 720ms（两个超帧）。这是协议决定的，软件不可绕过。
- **真正救不回的场景（诚实声明）**：多路真信号分散在不同子带、且每路都极短，服务一圈 M×720ms 扛不住。这是带宽/硬件问题（更宽瞬时带宽或多接收机），调度层不承诺解决。前提假设"同时活跃信号不多"（与策略 C 选型一致）。
- 全速回归测试用 `throttle=False`；数据文件缺失用 `pytest.skip()`，不得用裸 `return`。

---

## 3. 架构与组件边界

在现有 `realtime/` 之上新增**调谐层**。核心不变量：**解调核心一行不改，新增的全是"频率维度的调度与坐标"。**

```
┌─ 调谐层（本期新增）────────────────────────────────────────┐
│  SweepController        调度大脑：频点列表 + 两挡速度 + 活跃优先 │
│      │ next_dwell() → (center_hz, dwell_windows, mode)        │
│      ▼                                                        │
│  TunableIQSource.tune(center) + read_chunk()                  │
│      ├ FileTunableSource：从大文件 DDC 提取子带（仿真）        │
│      └ SoapyTunableSource：真实 SDR 调谐命令（占位）           │
└────────────────────────┬─────────────────────────────────────┘
                         │ 当前子带 IQ 流
              ┌──────────▼──────────┐
              │  RealtimeScanner     │  改造：主循环内嵌 tune→驻留→跳频
              │  （复用实时层主体）   │  盲区冻结的协调点
              └──────────┬──────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Detector(小改)   worker 池(复用)   SessionAggregator(小改)
   冻结非驻留信道    decode_window     按绝对射频频率归并
```

### 文件清单

```
realtime/tunable_source.py   create  TunableIQSource + FileTunableSource(仿真) + SoapyTunableSource(占位)
realtime/sweep_controller.py create  SweepController(两挡调度 + 频点列表 + 活跃优先)
realtime/scanner_rt.py       modify  主循环内嵌 tune→驻留→跳频;盲区冻结协调
realtime/detector.py         modify  支持 in_view 冻结(非驻留信道不累计 missed_windows)
realtime/aggregator.py       modify  归并键改为绝对射频频率(子带中心 + 检测偏移)
utils/synthesis.py           reuse   复用 synthesize_scenario;本期主用现有 2.5MHz 文件
tests/test_tunable_source.py create  调谐子带正确 + 时间指针只进不退(盲区语义)
tests/test_sweep_controller.py create 轮询顺序 + SURVEY↔CAMP 转移 + 活跃优先
tests/test_blind_spot.py     create  in_view 冻结 vs 老化
tests/test_sweep_e2e.py      create  端到端跳频:固定带宽放不下两路 → 跳频两路都抓到
```

### 各单元职责

- **TunableIQSource**：唯一的频率维度抽象。`tune(center_hz)` 命令调谐；`read_chunk()` 返回**当前调谐子带、当前时间指针起**的一小段瞬时带宽 IQ（`complex64`）。下游对仿真/硬件无感知。
- **SweepController**：调度大脑。维护频点列表（总频段按瞬时带宽切分的子带中心）、每子带状态、两挡速度策略。每次输出 `(center_hz, dwell_windows, mode)`。调度决策独立成可替换策略点——M-of-N / 负缓存是这里的**增强插点**，本期不实现。
- **RealtimeScanner（改造）**：主循环从"固定带宽跑到结束"变为"问调度器 → tune → 取窗口处理 → 喂聚合器 → 问调度器是否跳频"。盲区冻结在此协调（tune 后告知 detector 当前子带覆盖哪些频点）。
- **Detector（小改）**：增加 `in_view` 概念——只对当前驻留子带内的信道做老化，其余冻结。
- **SessionAggregator（小改）**：归并键从子带内偏移 `_fo_hz` 改为**绝对射频频率** = 子带中心 + 检测偏移。

---

## 4. 数据流与时间-频率耦合

### 4.1 仿真源的取数语义（防作弊核心）

```
大文件 = 整个总频段 × 全部时长（空中真实存在的一切）

FileTunableSource 内部状态：
  _time_ptr   全局时间指针（样点偏移），只进不退，与 _center 无关
  _center     当前调谐中心频率

tune(center):    只改 _center，不动 _time_ptr
read_chunk():    从 _time_ptr 起取一段全带宽 IQ
                 → DDC 到 _center → 低通 → 抽取到瞬时带宽
                 → _time_ptr 前进 chunk 时长（消耗了时间）
                 → 返回该子带 IQ；文件耗尽返回 None
```

**关键不变量：`_time_ptr` 全局、只进不退、与频率无关。** 由此强制：

- 调谐到子带 B 处理那段时间 → `_time_ptr` 推进 → 子带 A 的那段时间被**永久跳过**（= 接收机当时没看 A）。
- tune 切回 A 时 `_time_ptr` 已在后面，**取不回**错过的 A——与真实接收机一致。

### 4.2 主循环数据流

```
SweepController.next_dwell() → (center, dwell_windows, mode)
        │  mode = SURVEY(快扫,极短) | CAMP(驻守,≥720ms)
        ▼
source.tune(center)
        │
        ▼ 取 dwell_windows 个窗口：read_chunk → ring → read_window
        │
   SURVEY: 只跑 detector 能量检测 → 上报"该子带有无能量"给 controller
   CAMP  : 完整 detector → worker → aggregator 解码
        │
        ▼
   controller 据本次 dwell 结果更新子带状态(IDLE/ACTIVE/CAMPING)
        │
        ▼ 决定下一次 tune 哪个子带、什么模式（活跃优先）
   回到顶部
```

**两挡速度协作：** SURVEY 极短地轮询所有子带找能量；某子带见能量 → 升级 CAMP 去驻守解码；CAMP 期间该子带优先，解完/超时回到 SURVEY 轮询。

### 4.3 绝对频率坐标

worker 解出的 PDU 带子带内偏移 `_fo_hz`；进聚合器前由 scanner 加上当前 `_center` 得到**绝对射频频率**作为归并键。子带 A 不同次驻守的同一通话据此正确接续，不因换子带而裂成两条记录。

### 4.4 带内多信号 = 免费搭车（复用策略 C）

一份子带 IQ 物理上同时含该子带内所有信号。现有 `Detector` 一次 Welch 找多个活跃频点 + 策略 C 并行派发 + worker 池并行解 + 聚合器分别归并——**这是现有实时层已验证的能力，跳频不为此做特殊处理**。唯一要求：CAMP 驻守够久（≥720ms）让每路凑齐超帧，worker 数够分。

**设计含义：瞬时带宽越宽 → 越多信号落在同一子带被一次并发解掉 → retune 次数越少。** 这是"瞬时带宽 vs 数据量"权衡的有利面。

---

## 5. 盲区感知状态机（本期核心难点）

```
信道没检测到能量：
   ├ 原因A: 接收机当时没调谐到该子带 → 盲区，非信号消失
   └ 原因B: 接收机正看着它但没能量    → 信号真结束
```

**解决规则：信道状态机只在"正在看"时推进，盲区里冻结。**

```
每个信道（按绝对射频频点索引）：
   state          IDLE / ACTIVE / CLOSING
   in_view        当前驻留子带是否覆盖此频点（由 SweepController 经 scanner 告知 detector）

老化规则：
   if in_view:    正常逻辑 — 有能量→刷新; 无能量→missed_windows++; 超阈值→CLOSING
   else:          冻结 — missed_windows 与 state 不变（盲区无从判断）
```

即 `missed_windows` **只统计"在看且无能量"的窗口**。只有接收机确实驻守在某信道所在子带、却连续数窗收不到能量，才判定真结束；盲区时间不计入。

**两个兜底（均复用现有实现）：**

1. **超时兜底** `CALL_TIMEOUT_WINDOWS`：通话太久无任何新 PDU（无论盲区或真静默）→ 超时关闭，`closed_by="timeout"`。防盲区里早结束的通话永久挂起。
2. **Late Entry 补救**：盲区错过的超帧，下次驻守回来时从后续语音帧重建 LC，摊薄盲区代价。

**两层状态衔接：** SweepController 按**子带**索引（决定 tune 哪、是否继续 CAMP——只要子带内还有任一活跃信道就继续）；Detector/Aggregator 按**绝对射频频点**索引（每通话独立生命周期）。scanner 每次 tune 后把"当前子带覆盖哪些频点"传给 detector，detector 据此置各信道 `in_view`。

**本期范围：** "冻结 vs 老化"规则建进骨架；阈值（几窗判结束、超时窗数）为可配参数留待硬件调。M-of-N 开门确认、负缓存为本状态机的增强，接口预留、本期不实现。

---

## 6. 错误处理与监控

| 场景 | 处理 |
|------|------|
| 文件/流耗尽 | `read_chunk()` 返回 None → 主循环排空 → 聚合器 flush 未关闭 session（timeout）→ 退出 |
| worker 解码异常 | 复用 `decode_window` 内 try/except，单窗失败返回 []，不拖垮池 |
| 盲区误判通话结束 | §5 in_view 冻结 + 超时兜底 + Late Entry 三重保护 |
| Terminator 丢失 | 超时兜底关闭，`closed_by="timeout"` |
| 频率微抖 | detector 按 `channel_grid_hz` 量化；aggregator 按 `fo_bucket` 归并（均复用） |
| 跳频太慢漏信号 | 诚实声明：M×720ms 下限受协议+硬件约束；SURVEY 快扫压低空子带成本；Late Entry 给多次机会 |

---

## 7. 测试与验证策略

沿用 `throttle=False` 全速回归 + 数据缺失 `pytest.skip()`。

**单元层：**
- `FileTunableSource`：tune 不同中心 → 返回子带频谱峰值在对应位置；**时间指针只进不退**——tune 到 B 再切回 A，取到 A 的"后段"而非错过的那段（盲区语义核心断言）。
- `SweepController`：固定频点列表轮询顺序；SURVEY 见能量 → 升级 CAMP；CAMP 结束 → 回 SURVEY；活跃优先。
- 盲区状态机：构造"在看/盲区"序列，断言 `missed_windows` 只在 `in_view` 时累加，盲区冻结不误关。

**集成层：**
- 绝对频率归并：同一信道跨两次驻守 → 归并成**一个** CallRecord，不裂成两个。
- 带内多信号：一个子带放两路 → 一次驻守并行解出两个 CallRecord。

**端到端：**
- 用现有 `data/synthesized_wideband_2.5MHz.rawiq`（DMR1@-300kHz、DMR2@+150kHz），瞬时带宽设为放不下两路的宽度（如 500kHz）→ 强制跳频 → 断言两路都被发现且各带绝对射频频率。**这是"跳频确实在工作"的铁证：固定带宽放不下，唯有跳频能两个都抓到。**

**验证脚本：** 扩展 `scanner_rt.py` CLI，加 `--sweep --span HZ --inst-bw HZ`，可命令行直接跑一次跳频扫描看输出。

---

## 8. 关键决策记录

1. **架构**：可调谐源 + 调度器内嵌 RealtimeScanner（非外层 mini-run）。理由：跳频是持续、有记忆的过程，同一实例承载状态最自然，跨子带状态无需外层缝合；硬件接入只换 `TunableIQSource`。
2. **机制范围**：快扫/驻守两挡 + 活跃优先建进骨架；M-of-N、负缓存预留接口。
3. **仿真语义**：单调谐指针 + 时间只进不退，最贴近真实接收机、防作弊。
4. **频段模型**：总频段 = 文件跨度，瞬时带宽可配；本期用现有 2.5MHz 文件验证。
5. **多信号**：带内并发交给现有策略 C（驻守够久即可）；跳频只管带间调度。两层状态：子带级（调度）/ 信道级（通话生命周期）。
