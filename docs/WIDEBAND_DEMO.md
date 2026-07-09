# 宽带 DMR 信道化器 — 使用与演示指南

一次采集 60–70MHz 宽带 IQ → PFB 多相信道化切成 N 个重叠子带 → 逐子带能量检测 →
解调 → 按绝对射频频率归并通话。下面是怎么跑、怎么演示。

所有命令在**项目根目录** `DMR_demo/` 下，用专用解释器运行：

```bash
PY=/home/lzkj/miniconda3/envs/DMR_demo/bin/python
```

---

## 0. 项目结构

```
DMR_demo/
├── bvsp_decode.py          # 最终解码脚本（repo 根目录，git 跟踪）
├── utils/                  # 可视化脚本（git 跟踪）
│   ├── wideband_live.py
│   ├── wideband_viz.py
│   ├── wideband_anim.py
│   └── late_entry_viz.py
├── debug/                  # 验证 / 诊断工具（git 跟踪）
│   └── verify_src_dst.py
├── output/                 # 仅存放生成产物（PNG / GIF / scene，gitignore）
└── docs/                   # 文档
```

---

## 1. 真实数据：解析 BVSP 采集文件

`DMR_signal/*.bvsp` 是 USRP 采集（112 字节头 + 交错 int16 IQ，61.44Msps，中心 431MHz，
每个 1 秒；见 `DMR_signal/README.txt`）。无图、纯解码，最快看结果：

```bash
$PY bvsp_decode.py 1        # 解码 1.bvsp
$PY bvsp_decode.py all      # 解码 1..5.bvsp 并汇总
```

每个文件约 30 秒（信道化 61.44M 样点是主要开销）。当前 5 个文件的实测结果：

| 文件 | 解出通话 |
|------|---------|
| 1–4  | 432.2300 MHz (SRC=1 DST=1, GroupVoiceChannelUser) |
| 5    | 429.7700 + 432.2300 MHz（两路语音） |

> 跨 5 个独立文件、同频点解出一致的 SRC/DST，基本排除误判——管线在真实空口数据上工作正常。

---

## 2. 实时弹窗演示（推荐给别人看）

像 `dmr_pipeline_v2` 一样弹出窗口，**边扫描边刷新**：高亮框在子带间移动、扫到信号变绿、
解出的通话实时弹到 RF 轴上。

```bash
$PY utils/wideband_live.py                          # 默认 5.bvsp（两路，最好看）
$PY utils/wideband_live.py --file DMR_signal/1.bvsp # 指定文件
$PY utils/wideband_live.py --pause 0.5              # 放慢播放，讲解用
$PY utils/wideband_live.py --synth                  # 不用采集文件，合成 2 路信号场景
```

窗口三层：
- **上**：全带 PSD + 移动的扫描高亮框（橙=该区间静默，绿=发现信号→解调）
- **中**：当前子带基带谱 + 归属区阴影（绿线=区内信号被解码，灰线=区外混叠被跳过）
- **下**：绝对 RF 轴，随时间累计点亮解出的通话

前 ~30 秒是“读取 + 信道化”（终端打印 `[stage 1] ...`），之后进入实时扫描动画。
关闭窗口即退出。

**无显示器 / SSH 无 X 转发**时改为导出 GIF（同一套渲染，无需窗口）：

```bash
$PY utils/wideband_live.py --file DMR_signal/5.bvsp --save out.gif
```

---

## 3. 静态分级图（讲原理 / 放文档）

一张四阶段图：宽带输入(时/频) → 子带能量门 → 各活跃子带归属区 → RF 轴解码结果。

```bash
$PY utils/wideband_viz.py                 # 默认合成场景 → output/wideband_viz.png
```

离线动画 GIF（同样基于合成场景，便于嵌 PPT/文档）：

```bash
$PY utils/wideband_anim.py --fps 2        # → output/wideband_anim.gif
```

---

## 4. 常用参数

`wideband_live` / `bvsp_decode` 关键参数（信道化按 `子带率 = fs/N × oversample`）：

| 参数 | 含义 | BVSP 推荐 |
|------|------|----------|
| `--nsub N` | 子带数 | 48（→ 子带率 2.56MHz，贴近解码 2.5MHz 设计点） |
| `--oversample K` | 过抽样（相邻子带重叠，防边界漏检） | 2 |
| `--window-sec` / `--step-sec` | 解码窗长 / 步进 | 0.5 / 0.25 |
| `--center` `--fs` `--header` | 带中心 / 采样率 / 文件头字节 | 431e6 / 61.44e6 / 112（.bvsp 自动） |

换一批不同中心频率/采样率的采集时，改 `--center --fs` 即可；`.bvsp` 后缀会自动套用
112 字节头和 61.44MHz 默认值。

---

## 5. 文件清单

| 文件 | 作用 |
|------|------|
| `bvsp_decode.py`          | 真实 BVSP 文件批量解码（无图，最快） |
| `utils/wideband_live.py`  | **实时弹窗演示**（核心演示脚本） |
| `utils/wideband_viz.py`   | 静态四阶段分级图 |
| `utils/wideband_anim.py`  | 离线动画 GIF |
| `utils/late_entry_viz.py` | 晚入帧可视化 |
| `debug/verify_src_dst.py` | 验证 / 诊断：手工 ETSI 比特提取 vs 原生解析器（含 FEC 状态） |
| `realtime/wideband_source.py` | `FileWidebandSource`（新增 `header_bytes` 支持 BVSP 头） |
| `realtime/wideband_scanner.py` | `WidebandScanner` 两级编排（信道化 + 逐子带解码） |

> 以上演示脚本全部**复用真实生产组件**（`PolyphaseChannelizer` / `WidebandScanner` /
> `Detector` / `decode_window`），不另写任何 DSP——屏幕上看到的就是真实管线的行为。
