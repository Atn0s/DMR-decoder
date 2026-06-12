#!/usr/bin/env python3
"""直观展示 find_data_sync_positions 和 recover_burst_symbols 的效果。

用法:
    python visualize_stage6.py [iq_file]

默认使用 synthesized_wideband_2.5MHz.rawiq（宽带合成文件）。
也支持窄带文件（如 dmr_1_78125.rawiq 或 dmr_2_78125.rawiq）。

输出 4 张图:
  图1: 鉴频波形 + NCC 曲线 + 检测到的数据同步峰
  图2: 校准前 vs 校准后的 132 符号波形对比（最佳相位）
  图3: 校准前 vs 校准后的符号星座图（散点 + 直方图）
  图4: 33 个相位扫描的残差曲线 + 各相位的星座对比
"""

import os
import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import dmr_pipeline_v2 as P

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True

# ── 配置 ─────────────────────────────────────────────────────
TARGET_FILE = "data/synthesized_wideband_2.5MHz.rawiq"
CANDIDATE_IDX = 0          # 默认处理第一个候选（通常是 MS Sourced）
PEAK_IDX = 0               # 用第几个 NCC 峰做 recover 演示


def plot_stage6a_front_end_and_ncc(y_lc, ncc, sync_positions, name, cf_hz):
    """图1: 鉴频波形 + NCC 曲线 + 检测到的数据同步峰。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1]})
    fig.suptitle(f"find_data_sync_positions — {name} 数据同步码 NCC 检测\n"
                 f"(残余载波 {cf_hz/1e3:+.2f} kHz)",
                 fontsize=13, fontweight='bold')

    # ── 上: 鉴频波形 ──
    t_ms = np.arange(len(y_lc)) / P.Fs_dec * 1000.0
    ax1.plot(t_ms, y_lc, color='steelblue', alpha=0.7, linewidth=0.5,
             label='鉴频输出 (lc_front_end)')
    ax1.axhline(+3, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.axhline(+1, color='orange', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.axhline(-1, color='orange', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.axhline(-3, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.set_ylabel('标称符号电平')
    ax1.set_ylim(-6, 6)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 下: NCC ──
    ax2.plot(t_ms, ncc, color='darkgreen', alpha=0.85, linewidth=0.7,
             label=f'NCC (数据同步 {name})')
    ax2.axhline(+P.NCC_THRESHOLD, color='red', linestyle='-.', alpha=0.6,
                label=f'阈值 ±{P.NCC_THRESHOLD}')
    ax2.axhline(-P.NCC_THRESHOLD, color='red', linestyle='-.', alpha=0.6)

    # 标记检测到的峰
    colors_pos = {'+': 'darkorange', '-': 'darkred'}
    for sc, sgn in sync_positions:
        t_peak = sc / P.Fs_dec * 1000.0
        c = colors_pos['+' if sgn > 0 else '-']
        marker = 'v' if sgn > 0 else '^'
        ax2.plot(t_peak, sgn * P.NCC_THRESHOLD, marker, color=c,
                 ms=10, markeredgewidth=0.5, markeredgecolor='black',
                 zorder=5)

    ax2.set_xlabel('时间 (ms)')
    ax2.set_ylabel('NCC')
    ax2.set_ylim(-1.05, 1.05)
    ax2.legend(loc='lower left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _run_phase_sweep(y_lc, sync_center, sgn, name, phases):
    """内部: 跑一次相位扫描，返回 all_phases 列表和 best 元组。"""
    ref = P.data_sync_sym[name]
    levels = np.array([-3, -1, 1, 3])
    all_phases = []
    best = (1e18, None, None, None)

    for ph in phases:
        start = sync_center - (54 + 12) * P.SPS + ph
        pos = start + np.arange(132) * P.SPS
        if pos[0] < 0 or pos[-1] >= len(y_lc) - 1:
            continue
        seg = sgn * P._interp(y_lc, pos)
        sy = seg[54:78]
        A = np.vstack([sy, np.ones(24)]).T
        a, b = np.linalg.lstsq(A, ref, rcond=None)[0]
        segc = a * seg + b
        near = levels[np.argmin(np.abs(segc[:, None] - levels[None, :]), axis=1)]
        resid = np.mean((segc - near) ** 2)
        all_phases.append({'ph': ph, 'a': a, 'b': b, 'resid': resid,
                           'seg_raw': seg.copy(), 'seg_cal': segc.copy()})
        if resid < best[0]:
            best = (resid, segc, ph, seg)
    return all_phases, best


def plot_stage6b_recover_comparison(y_lc, sync_center, sgn, name):
    """图2+3: recover_burst_symbols 前后对比。同时跑窄/宽两种范围。"""
    # 窄范围（原版）
    phases_narrow = np.linspace(-4, 4, 33)
    all_narrow, best_narrow = _run_phase_sweep(y_lc, sync_center, sgn, name, phases_narrow)

    # 宽范围（实验对比）
    phases_wide = np.linspace(-8, 8, 65)
    all_wide, best_wide = _run_phase_sweep(y_lc, sync_center, sgn, name, phases_wide)

    if best_narrow[0] >= 1e17:
        print("  [warn] 所有相位越界，无法生成对比图")
        return None, None

    resid_opt, segc_opt, ph_opt, seg_raw_opt = best_narrow
    ph_opt_wide = best_wide[2]
    resid_opt_wide = best_wide[0]

    # ── 图2: 132 符号波形对比 ──
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [1, 1]})
    fig2.suptitle(
        f"recover_burst_symbols 校准前后对比 — {name} "
        f"(同步中心={sync_center}, sgn={sgn:+.0f}, 最优相位={ph_opt:+.2f})",
        fontsize=13, fontweight='bold')

    sym_idx = np.arange(132)

    # 上: 校准前
    ax1.plot(sym_idx, seg_raw_opt, 'o-', color='steelblue', ms=4, linewidth=0.8,
             alpha=0.8, label='原始插值符号')
    ax1.axvspan(54, 77, color='gold', alpha=0.2, label='同步区 [54,78)')
    for lv, col in [(-3, 'red'), (-1, 'orange'), (1, 'orange'), (3, 'red')]:
        ax1.axhline(lv, color=col, linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.set_ylabel('原始电平 (校准前)')
    ax1.set_ylim(-6, 6)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 下: 校准后
    ax2.plot(sym_idx, segc_opt, 'o-', color='darkgreen', ms=4, linewidth=0.8,
             alpha=0.8, label='校准后符号')
    ax2.axvspan(54, 77, color='gold', alpha=0.2, label='同步区 [54,78)')
    for lv, col in [(-3, 'red'), (-1, 'orange'), (1, 'orange'), (3, 'red')]:
        ax2.axhline(lv, color=col, linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.set_xlabel('符号索引 (0-131)')
    ax2.set_ylabel('校准后电平 (gain/offset 标定)')
    ax2.set_ylim(-5, 5)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig2, (seg_raw_opt, segc_opt, all_narrow, all_wide,
                   ph_opt, resid_opt, ph_opt_wide, resid_opt_wide)


def plot_stage6c_constellation(seg_raw, seg_cal, name, ph_opt, resid_opt):
    """图3: 星座图（散点 + 直方图）校准前后对比。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"星座对比 — {name} 最优相位={ph_opt:+.2f} 残差={resid_opt:.4f}",
        fontsize=13, fontweight='bold')

    # (0,0) 校准前散点
    axes[0, 0].scatter(np.arange(132), seg_raw, c='steelblue', s=8, alpha=0.6)
    for lv, col in [(-3, 'red'), (-1, 'orange'), (1, 'orange'), (3, 'red')]:
        axes[0, 0].axhline(lv, color=col, linestyle='--', alpha=0.5, linewidth=0.8)
    axes[0, 0].set_title('校准前 — 符号散点')
    axes[0, 0].set_ylabel('电平')
    axes[0, 0].set_ylim(-6, 6)
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) 校准后散点
    axes[0, 1].scatter(np.arange(132), seg_cal, c='darkgreen', s=8, alpha=0.6)
    for lv, col in [(-3, 'red'), (-1, 'orange'), (1, 'orange'), (3, 'red')]:
        axes[0, 1].axhline(lv, color=col, linestyle='--', alpha=0.5, linewidth=0.8)
    axes[0, 1].set_title('校准后 — 符号散点')
    axes[0, 1].set_ylim(-5, 5)
    axes[0, 1].grid(True, alpha=0.3)

    # (1,0) 校准前直方图
    axes[1, 0].hist(seg_raw, bins=80, color='steelblue', alpha=0.75, edgecolor='white')
    for lv, col in [(-3, 'red'), (-1, 'orange'), (1, 'orange'), (3, 'red')]:
        axes[1, 0].axvline(lv, color=col, linestyle='--', alpha=0.6, linewidth=1.2)
    axes[1, 0].set_title('校准前 — 电平直方图')
    axes[1, 0].set_xlabel('电平')
    axes[1, 0].set_ylabel('计数')
    axes[1, 0].set_xlim(-6, 6)

    # (1,1) 校准后直方图
    axes[1, 1].hist(seg_cal, bins=80, color='darkgreen', alpha=0.75, edgecolor='white')
    for lv, col in [(-3, 'red'), (-1, 'orange'), (1, 'orange'), (3, 'red')]:
        axes[1, 1].axvline(lv, color=col, linestyle='--', alpha=0.6, linewidth=1.2)
    axes[1, 1].set_title('校准后 — 电平直方图')
    axes[1, 1].set_xlabel('电平')
    axes[1, 1].set_ylabel('计数')
    axes[1, 1].set_xlim(-5, 5)

    plt.tight_layout()
    return fig


def plot_stage6d_phase_sweep(all_phases_narrow, all_phases_wide, ph_opt_narrow,
                              ph_opt_wide, name):
    """图4: 窄范围 [-4,4] vs 宽范围 [-8,8] 残差曲线对比 + 最优星座。"""
    phases_n = [p['ph'] for p in all_phases_narrow]
    resids_n = [p['resid'] for p in all_phases_narrow]
    phases_w = [p['ph'] for p in all_phases_wide]
    resids_w = [p['resid'] for p in all_phases_wide]

    idx_opt_n = np.argmin(resids_n)
    idx_opt_w = np.argmin(resids_w)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"相位扫描范围对比 — {name}\n"
                 f"窄 [-4,4] 最优: ph={ph_opt_narrow:+.2f} resid={resids_n[idx_opt_n]:.4f}  |  "
                 f"宽 [-8,8] 最优: ph={ph_opt_wide:+.2f} resid={resids_w[idx_opt_w]:.4f}",
                 fontsize=13, fontweight='bold')

    # ── 上: 残差曲线对比 ──
    ax1 = fig.add_axes([0.06, 0.55, 0.44, 0.38])
    ax1.plot(phases_n, resids_n, 'o-', color='darkblue', ms=5, linewidth=1.2,
             alpha=0.8, label='[-4, 4] 33步')
    ax1.plot(phases_w, resids_w, 's--', color='darkred', ms=5, linewidth=1.2,
             alpha=0.8, label='[-8, 8] 65步')
    ax1.axvline(ph_opt_narrow, color='darkblue', linestyle=':', linewidth=1.5)
    ax1.axvline(ph_opt_wide, color='darkred', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('亚符号相位偏移 (样点)')
    ax1.set_ylabel('4 电平星座残差 (MSE)')
    ax1.set_title('残差曲线对比（全范围）')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 上右: 放大 [-5,1] 区域 ──
    ax2 = fig.add_axes([0.55, 0.55, 0.42, 0.38])
    ax2.plot(phases_n, resids_n, 'o-', color='darkblue', ms=6, linewidth=1.5,
             alpha=0.9, label='[-4, 4]')
    ax2.plot(phases_w, resids_w, 's--', color='darkred', ms=6, linewidth=1.5,
             alpha=0.9, label='[-8, 8]')
    ax2.axvline(ph_opt_narrow, color='darkblue', linestyle=':', linewidth=2,
                label=f'窄最优 {ph_opt_narrow:+.2f}')
    ax2.axvline(ph_opt_wide, color='darkred', linestyle=':', linewidth=2,
                label=f'宽最优 {ph_opt_wide:+.2f}')
    ax2.set_xlim(-5, 1)
    ax2.set_xlabel('亚符号相位偏移 (样点)')
    ax2.set_ylabel('4 电平星座残差 (MSE)')
    ax2.set_title('残差曲线对比（放大）')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── 下: 最优相位的星座散点对比 ──
    ax3 = fig.add_axes([0.06, 0.05, 0.44, 0.40])
    seg_n = all_phases_narrow[idx_opt_n]['seg_cal']
    ax3.scatter(np.arange(132), seg_n, c='darkblue', s=6, alpha=0.5)
    for lv, col in [(-3, 'red'), (-1, 'orange'), (1, 'orange'), (3, 'red')]:
        ax3.axhline(lv, color=col, linestyle='--', alpha=0.35, linewidth=0.6)
    ax3.set_title(f'窄 [-4,4] 最优 ph={ph_opt_narrow:+.2f} resid={resids_n[idx_opt_n]:.4f}')
    ax3.set_ylabel('电平')
    ax3.set_ylim(-5, 5)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_axes([0.55, 0.05, 0.42, 0.40])
    seg_w = all_phases_wide[idx_opt_w]['seg_cal']
    ax4.scatter(np.arange(132), seg_w, c='darkred', s=6, alpha=0.5)
    for lv, col in [(-3, 'red'), (-1, 'orange'), (1, 'orange'), (3, 'red')]:
        ax4.axhline(lv, color=col, linestyle='--', alpha=0.35, linewidth=0.6)
    ax4.set_title(f'宽 [-8,8] 最优 ph={ph_opt_wide:+.2f} resid={resids_w[idx_opt_w]:.4f}')
    ax4.set_ylim(-5, 5)
    ax4.grid(True, alpha=0.3)

    return fig


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else TARGET_FILE
    if not os.path.exists(target):
        print(f"错误: 文件 {target} 不存在")
        sys.exit(1)

    print("=" * 70)
    print(f"  可视化 Stage 6: find_data_sync_positions + recover_burst_symbols")
    print(f"  输入文件: {target}")
    print("=" * 70)

    # ── 加载信号 ──
    iq = P.read_rawiq(target)

    # 判断是宽带还是窄带
    is_wideband = '78125' not in target
    if is_wideband:
        # ── 宽带: DDC → 只取第一个候选 ──
        f_w, psd_w = signal.welch(iq, fs=P.Fs_wide, nperseg=4096,
                                  return_onesided=False)
        f_w = np.fft.fftshift(f_w)
        psd_db = 10 * np.log10(np.fft.fftshift(psd_w))
        nf = np.median(psd_db)
        peaks, _ = signal.find_peaks(psd_db, height=nf + 15, distance=20)
        if len(peaks) == 0:
            print("未检测到候选信号")
            return
        idx_c = min(CANDIDATE_IDX, len(peaks) - 1)
        fo = f_w[peaks[idx_c]]
        print(f"宽带模式: 候选 {idx_c+1}/{len(peaks)}, LO={fo/1e3:+.2f} kHz")
        t = np.arange(len(iq)) / P.Fs_wide
        iq_dec = signal.resample_poly(
            iq * np.exp(-1j * 2 * np.pi * fo * t),
            P.UP_FACTOR, P.DOWN_FACTOR)
    else:
        # ── 窄带: 直接重采样 ──
        print("窄带模式: 直接重采样到 48kHz")
        iq_dec = signal.resample_poly(iq, 384, 625)

    # ── 运行 lc_front_end ──
    y_lc = P.lc_front_end(iq_dec)
    print(f"lc_front_end: {len(y_lc)} 样点 @ 48kHz = {len(y_lc)/P.Fs_dec:.2f}s")

    # ── 对两个模板都试 ──
    for name in ("MS Sourced", "BS Sourced"):
        print(f"\n{'─'*50}")
        print(f"  模板: {name}")
        print(f"{'─'*50}")

        # ═══════════════════════════════════════════════════════
        # 阶段 A: find_data_sync_positions
        # ═══════════════════════════════════════════════════════
        # dmr_pipeline_v2 的 find_data_sync_positions 只返回位置列表，
        # 不返回 ncc。我们手动算一份 ncc 用来画图。
        ref = P.data_sync_sym[name]
        rwave = np.repeat(ref, P.SPS)
        c = signal.correlate(y_lc, rwave, mode='same')
        e = np.convolve(y_lc ** 2, np.ones(len(rwave)), mode='same')
        e = np.where(e <= 0, 1e-9, e)
        ncc = c / np.sqrt(e * np.sum(rwave ** 2))

        syncs = P.find_data_sync_positions(y_lc, name)
        print(f"  检测到 {len(syncs)} 个数据同步候选")

        # 估算残余载波（从 lc_front_end 的逻辑反推）
        f_d, ps_d = signal.welch(iq_dec, fs=P.Fs_dec, nperseg=4096,
                                 return_onesided=False)
        cf = np.fft.fftshift(f_d)[np.argmax(np.fft.fftshift(ps_d))]

        fig1 = plot_stage6a_front_end_and_ncc(y_lc, ncc, syncs, name, cf)
        fig1.savefig(f'output/stage6a_ncc_{name.replace(" ", "_")}.png', dpi=150)
        print(f"  → 保存: output/stage6a_ncc_{name.replace(' ', '_')}.png")

        if not syncs:
            print(f"  无数据同步候选，跳过 recover 演示")
            continue

        # ═══════════════════════════════════════════════════════
        # 阶段 B+C+D: recover_burst_symbols 全流程可视化
        # ═══════════════════════════════════════════════════════
        sc, sgn = syncs[min(PEAK_IDX, len(syncs) - 1)]
        print(f"  选用峰 #{min(PEAK_IDX, len(syncs)-1)}: "
              f"中心={sc}, 极性={'正' if sgn>0 else '负'}")

        fig2, result = plot_stage6b_recover_comparison(y_lc, sc, sgn, name)
        if fig2 is None:
            continue
        fig2.savefig(f'output/stage6b_recover_{name.replace(" ", "_")}.png', dpi=150)
        print(f"  → 保存: output/stage6b_recover_{name.replace(' ', '_')}.png")

        seg_raw, seg_cal, all_narrow, all_wide, ph_opt, resid_opt, ph_opt_wide, resid_opt_wide = result

        fig3 = plot_stage6c_constellation(seg_raw, seg_cal, name, ph_opt, resid_opt)
        fig3.savefig(f'output/stage6c_constellation_{name.replace(" ", "_")}.png', dpi=150)
        print(f"  → 保存: output/stage6c_constellation_{name.replace(' ', '_')}.png")

        fig4 = plot_stage6d_phase_sweep(all_narrow, all_wide, ph_opt, ph_opt_wide, name)
        fig4.savefig(f'output/stage6d_phase_sweep_{name.replace(" ", "_")}.png', dpi=150)
        print(f"  → 保存: output/stage6d_phase_sweep_{name.replace(' ', '_')}.png")

        # ── 打印关键数值 ──
        idx_opt_n = np.argmin([p['resid'] for p in all_narrow])
        idx_opt_w = np.argmin([p['resid'] for p in all_wide])
        print(f"\n  关键数值:")
        print(f"    窄 [-4,4]: 最优相位={ph_opt:+.2f} 样点  残差={resid_opt:.4f}  "
              f"增益={all_narrow[idx_opt_n]['a']:.4f}  偏置={all_narrow[idx_opt_n]['b']:+.4f}")
        print(f"    宽 [-8,8]: 最优相位={ph_opt_wide:+.2f} 样点  残差={resid_opt_wide:.4f}  "
              f"增益={all_wide[idx_opt_w]['a']:.4f}  偏置={all_wide[idx_opt_w]['b']:+.4f}")
        delta = abs(ph_opt_wide - ph_opt)
        delta_r = resid_opt_wide - resid_opt
        if delta > 0.25:
            print(f"    ⚠ 最优相位偏移 {delta:+.2f} 样点! 窄范围可能不够")
        else:
            print(f"    ✓ 最优相位一致 (差 {delta:.2f} 样点)，窄范围 [-4,4] 足够")
        if delta_r < -0.01:
            print(f"    ⚠ 宽范围残差降低 {delta_r:.4f}，扩大范围有实质收益")
        else:
            print(f"    ✓ 残差无明显改善 ({delta_r:+.4f})")

    print(f"\n{'='*70}")
    print("  可视化完成。生成文件:")
    for name in ("MS_Sourced", "BS_Sourced"):
        for prefix in ("stage6a_ncc", "stage6b_recover",
                       "stage6c_constellation", "stage6d_phase_sweep"):
            fname = f"output/{prefix}_{name}.png"
            if os.path.exists(fname):
                print(f"    {fname}")
    print("=" * 70)
    plt.show()


if __name__ == '__main__':
    main()
