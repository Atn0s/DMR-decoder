import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int

from okdmr.dmrlib.etsi.fec.golay_20_8_7 import Golay2087
from okdmr.dmrlib.etsi.fec.bptc_196_96 import BPTC19696
from okdmr.dmrlib.etsi.fec.reed_solomon_12_9_4 import ReedSolomon1294
from okdmr.dmrlib.etsi.fec.vbptc_128_72 import VBPTC12873
from okdmr.dmrlib.etsi.fec.five_bit_checksum import FiveBitChecksum
from okdmr.dmrlib.etsi.layer2.pdu.full_link_control import FullLinkControl
from okdmr.dmrlib.etsi.layer2.pdu.csbk import CSBK
from okdmr.dmrlib.etsi.layer2.pdu.embedded_signalling import EmbeddedSignalling
from okdmr.dmrlib.etsi.layer2.elements.lcss import LCSS

from core.burst_type import SlotDataType, VLC_RS_MASK
from core.dsp import adaptive_slice_bits


def decode_burst(symbols: np.ndarray, sync_type: str) -> dict | None:
    """Data Sync burst decoder: dispatches to LC_HEADER / TERMINATOR / CSBK.
    Voice Sync bursts are NOT handled here — use LateEntryCollector.feed() instead.
    Returns a unified PDU dict or None on FEC failure."""
    ba = adaptive_slice_bits(symbols)
    slot_bits = ba[98:108] + ba[156:166]   # 20-bit Slot Type field
    if not Golay2087.check(slot_bits.copy()):
        return None
    color_code = ba2int(slot_bits[0:4])
    data_type  = ba2int(slot_bits[4:8])
    info = ba[0:98] + ba[166:264]          # 196-bit info field

    if data_type == SlotDataType.VOICE_LC_HEADER:
        return _decode_lc_or_terminator(ba, info, color_code, "LC_HEADER")
    elif data_type == SlotDataType.TERMINATOR_WITH_LC:
        return _decode_lc_or_terminator(ba, info, color_code, "TERMINATOR")
    elif data_type == SlotDataType.CSBK:
        return _decode_csbk(ba, info, color_code)
    return None


def _decode_lc_or_terminator(ba264: bitarray, info196: bitarray,
                               color_code: int, pdu_type: str) -> dict | None:
    decoded = BPTC19696.deinterleave_data_bits(info196, repair_if_necessary=True)
    data12  = decoded[0:96].tobytes()
    if not ReedSolomon1294.check(data12, VLC_RS_MASK):
        return None
    try:
        flc = FullLinkControl.from_bits(decoded[0:96])
    except Exception:
        return None
    dst = flc.group_address or flc.target_address
    return {
        "type":     pdu_type,
        "src":      flc.source_address,
        "dst":      dst,
        "ts":       0,
        "flco":     flc.full_link_control_opcode.name,
        "extra":    {"color_code": color_code},
        "raw_bits": ba264.tobytes(),
    }


def _decode_csbk(ba264: bitarray, info196: bitarray, color_code: int) -> dict | None:
    decoded = BPTC19696.deinterleave_data_bits(info196, repair_if_necessary=True)
    csbk_bits = decoded[0:96]
    try:
        csbk = CSBK.from_bits(csbk_bits)
    except Exception:
        return None
    return {
        "type":    "CSBK",
        "src":     csbk.source_address or 0,
        "dst":     csbk.target_address or 0,
        "ts":      0,
        "flco":    csbk.csbko.name,
        "extra":   {"color_code": color_code, "last_block": csbk.last_block},
        "raw_bits": ba264.tobytes(),
    }


class LateEntryCollector:
    """Stateful EMB fragment collector. Call feed() for each Voice Sync burst.
    After collecting First+Cont+Cont+Last (4 fragments), triggers VBPTC decode
    and returns a PDU dict. Call reset() to restart collection."""

    def __init__(self):
        self._frags: list = []
        self._collecting: bool = False

    def reset(self):
        self._frags = []
        self._collecting = False

    def feed(self, ba264: bitarray, sync_type: str) -> dict | None:
        """Feed one Voice Sync burst (264 bits). Returns PDU dict when 4 fragments
        are assembled and CS5 passes, otherwise None."""
        center = ba264[108:156]
        emb_bits   = center[0:8] + center[40:48]   # 16-bit EMB header
        signalling = center[8:40]                   # 32-bit fragment

        try:
            emb = EmbeddedSignalling.from_bits(emb_bits)
        except Exception:
            return None
        lcss = emb.link_control_start_stop

        if not self._collecting:
            if lcss == LCSS.FirstFragmentLC:
                self._collecting = True
                self._frags = [signalling]
        else:
            self._frags.append(signalling)
            if len(self._frags) == 4:
                return self._decode_assembled(ba264)
        return None

    def _decode_assembled(self, last_ba264: bitarray) -> dict | None:
        b128 = self._frags[0] + self._frags[1] + self._frags[2] + self._frags[3]
        self.reset()
        lc77 = VBPTC12873.deinterleave_data_bits(b128, include_cs5=True)
        lc72 = lc77[0:72]
        rx_cs5 = ba2int(lc77[72:77])
        cs5_ok = (rx_cs5 <= 30) and FiveBitChecksum.verify(lc72.tobytes(), rx_cs5)
        if not cs5_ok:
            return None
        try:
            flc = FullLinkControl.from_bits(lc77)
        except Exception:
            return None
        dst = flc.group_address or flc.target_address
        return {
            "type":    "LATE_ENTRY",
            "src":     flc.source_address,
            "dst":     dst,
            "ts":      0,
            "flco":    flc.full_link_control_opcode.name,
            "extra":   {"cs5_ok": cs5_ok},
            "raw_bits": last_ba264.tobytes(),
        }
