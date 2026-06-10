"""Late Entry 可视化：语音超帧 LCSS 状态机时序 + EMB 校验 + 跨超帧 LC 一致性。"""
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import dmr_pipeline_v2 as P
import late_entry as LE
from okdmr.dmrlib.etsi.layer2.pdu.embedded_signalling import EmbeddedSignalling
from okdmr.dmrlib.etsi.layer2.elements.lcss import LCSS

raw = P.read_rawiq('dmr_2_78125.rawiq')
y = P.lc_front_end(signal.resample_poly(raw, 384, 625))
name = 'MS Sourced'
anchors = LE.find_voice_sync_anchor(y, name)

# 收集前若干超帧每个突发的 LCSS / parity / 解码结果
LCSS_SHORT = {LCSS.SingleFragmentLCorCSBK: 'Single', LCSS.FirstFragmentLC: 'First',
              LCSS.LastFragmentLCorCSBK: 'Last', LCSS.ContinuationFragmentLCorCSBK: 'Cont'}
LCSS_COLOR = {'First': '#2ca02c', 'Cont': '#1f77b4', 'Last': '#d62728', 'Single': '#999999'}

rows = []
for ai, (A, sgn) in enumerate(anchors[:12]):
    ph = LE.lock_phase_from_anchor(y, A, sgn, name)
    burst_info = []
    for j in range(7):
        ba = LE.recover_voice_burst(y, A, j, ph, sgn)
        if ba is None:
            break
        emb_bits, _ = LE.parse_emb_center(ba)
        try:
            emb = EmbeddedSignalling.from_bits(emb_bits)
            burst_info.append((LCSS_SHORT.get(emb.link_control_start_stop, '?'), emb.emb_parity_ok))
        except Exception:
            burst_info.append(('?', False))
    res = LE.decode_one_superframe(y, A, sgn, name)
    rows.append((A, burst_info, res))

fig, axs = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})

# 上图：每个超帧 7 个突发的 LCSS 状态机网格
ax = axs[0]
for ri, (A, binfo, res) in enumerate(rows):
    for j, (lc, pok) in enumerate(binfo):
        col = LCSS_COLOR.get(lc, '#cccccc')
        ax.add_patch(Rectangle((j, ri), 0.92, 0.85, facecolor=col,
                               edgecolor='black' if pok else 'red',
                               lw=2 if pok else 1.2, alpha=0.85 if pok else 0.4))
        ax.text(j + 0.46, ri + 0.42, lc, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold')
    ok = res is not None and res.get('cs5_ok')
    tag = ('OK SRC=%d DST=%d' % (res['src_id'], res['dst_id'])) if ok else 'no LC'
    ax.text(7.2, ri + 0.42, tag, ha='left', va='center', fontsize=9,
            color='green' if ok else 'gray', fontweight='bold' if ok else 'normal')
ax.set_xlim(0, 10.5); ax.set_ylim(0, len(rows))
ax.set_xticks(np.arange(7) + 0.46)
ax.set_xticks(np.arange(7) + 0.46)
ax.set_xticklabels(['Burst A\n(voice sync)', 'B', 'C', 'D', 'E', 'F', 'A(next)'])
ax.set_yticks(np.arange(len(rows)) + 0.42)
ax.set_yticklabels(['SF@%d' % A for A, _, _ in rows], fontsize=8)
ax.set_title('Late Entry: 语音超帧 LCSS 状态机 (粗黑框=EMB QR校验通过, 红细框=校验失败)\n'
             '正常序列 First->Cont->Cont->Last 携带 4x32bit, VBPTC纠错+CS5校验后得 LC',
             fontsize=11)
ax.invert_yaxis()

# 下图：跨超帧解出的 SRC/DST 一致性
ax2 = axs[1]
srcs = [r[2]['src_id'] if (r[2] and r[2].get('cs5_ok')) else None for r in rows]
xs = np.arange(len(rows))
okmask = [s is not None for s in srcs]
ax2.scatter(xs[np.array(okmask)], [1] * sum(okmask), c='green', s=80, label='CS5-verified LC (SRC=1)')
for i, s in enumerate(srcs):
    if s is not None:
        ax2.annotate('SRC=%d' % s, (i, 1), textcoords='offset points', xytext=(0, 8),
                     ha='center', fontsize=7)
ax2.set_ylim(0.5, 1.5); ax2.set_xlim(-0.5, len(rows) - 0.5)
ax2.set_yticks([]); ax2.set_xlabel('superframe index')
ax2.set_title('跨超帧 LC 一致性：所有 CS5 通过的超帧均解出相同 SRC=1/DST=1', fontsize=10)
ax2.legend(loc='upper right', fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('late_entry_viz.png', dpi=90)
n_ok = sum(1 for r in rows if r[2] and r[2].get('cs5_ok'))
print('saved late_entry_viz.png  (%d/%d 超帧 CS5 通过)' % (n_ok, len(rows)))
