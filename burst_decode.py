"""Clean single-burst DMR decoder at NATIVE 78125 Hz (no resample artifacts).
Step-1 goal: get a clean +-1/+-3 constellation and pass Golay on the slot type."""
import numpy as np, scipy.signal as signal
import dmr_pipeline_v2 as P
from bitarray import bitarray
from bitarray.util import ba2int
from okdmr.dmrlib.etsi.fec.golay_20_8_7 import Golay2087

Fs0 = 78125.0
SPS = Fs0 / 4800.0                      # 16.2760, fractional
SYNCS = {'BS': P.hex_to_symbols('DFF57D75DF5D'),
         'MS': P.hex_to_symbols('D5D7F77FD757')}
LV = np.array([-3, -1, 1, 3])


def interp(a, pos):
    i = np.floor(pos).astype(int); fr = pos - i
    i = np.clip(i, 0, len(a) - 2)
    return a[i] * (1 - fr) + a[i + 1] * fr


def decode_burst(seg_iq):
    iqf = signal.filtfilt(signal.firwin(63, 6500.0, fs=Fs0), [1.0], seg_iq)
    amp = np.abs(iqf)
    hz = np.angle(iqf[1:] * np.conj(iqf[:-1])) / (2 * np.pi) * Fs0
    act = amp[:-1] > 0.3 * amp.max()
    hz = hz - np.median(hz[act])         # DC from discriminator median (robust)
    win = int(round(SPS))
    yb = np.convolve(hz, np.ones(win) / win, mode='same')

    best = None
    for nm, sync in SYNCS.items():
        rw = np.repeat(sync, win).astype(float)
        c = signal.correlate(yb, rw, mode='same')
        sp = int(np.argmax(np.abs(c)))
        if best is None or abs(c[sp]) > best[0]:
            best = (abs(c[sp]), nm, sp, np.sign(c[sp]), sync)
    _, nm, sp, sgn, sync = best

    res = (1e18, None, 0.0)
    for ph in np.linspace(-SPS, SPS, 49):
        start = sp - 66 * SPS + ph
        pos = start + np.arange(132) * SPS
        if pos[0] < 0 or pos[-1] >= len(yb) - 1:
            continue
        vals = sgn * interp(yb, pos)
        sy = vals[54:78]
        A = np.vstack([sy, np.ones(24)]).T
        a, b = np.linalg.lstsq(A, sync, rcond=None)[0]
        vc = a * vals + b
        near = LV[np.argmin(np.abs(vc[:, None] - LV[None, :]), axis=1)]
        r = np.mean((vc - near) ** 2)
        if r < res[0]:
            res = (r, vc, ph)
    return nm, sp, res[1], np.sqrt(res[0])


def slot_from_symbols(vc):
    bits = []
    for v in vc:
        bits.extend(P.symbol_to_dibit(v))
    ba = bitarray(bits)
    slot = ba[98:108] + ba[156:166]
    ok = Golay2087.check(slot.copy())
    return ok, ba2int(slot[0:4]), ba2int(slot[4:8]), ba
