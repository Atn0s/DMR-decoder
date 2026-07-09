import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int

from dmr.constants import SlotDataType, VLC_RS_MASK
from dmr.dsp import adaptive_slice_bits
from dmr.fec import (
    bptc_196_96_decode,
    five_bit_checksum_verify,
    golay_20_8_7_check,
    rs_12_9_4_check,
    vbptc_128_72_decode,
)
from dmr.layer2 import (
    LCSS,
    FullLinkControl,
    CSBK,
    parse_csbk,
    parse_embedded_signalling,
    parse_full_link_control,
)


def decode_burst(symbols: np.ndarray, sync_type: str) -> dict | None:
    """Data Sync burst decoder: dispatches to LC_HEADER / TERMINATOR / CSBK.
    Voice Sync bursts are NOT handled here — use LateEntryCollector.feed() instead.
    Returns a unified PDU dict or None on FEC failure."""
    ba = adaptive_slice_bits(symbols)
    slot_bits = ba[98:108] + ba[156:166]   # 20-bit Slot Type field
    if not golay_20_8_7_check(slot_bits.copy()):
        return None
    color_code = ba2int(slot_bits[0:4])
    data_type  = ba2int(slot_bits[4:8])
    info = ba[0:98] + ba[166:264]          # 196-bit info field
    slot_extra = _slot_extra(slot_bits, color_code, data_type, sync_type)

    if data_type == SlotDataType.VOICE_LC_HEADER:
        return _decode_lc_or_terminator(ba, info, color_code, "LC_HEADER", slot_extra)
    elif data_type == SlotDataType.TERMINATOR_WITH_LC:
        return _decode_lc_or_terminator(ba, info, color_code, "TERMINATOR", slot_extra)
    elif data_type == SlotDataType.CSBK:
        return _decode_csbk(ba, info, color_code, slot_extra)
    return None


def _bits_hex(bits: bitarray) -> str:
    return bits.tobytes().hex()


def _data_type_name(data_type: int) -> str:
    try:
        return SlotDataType(data_type).name
    except ValueError:
        return f"UNKNOWN_{data_type}"


def _slot_extra(slot_bits: bitarray, color_code: int, data_type: int, sync_type: str) -> dict:
    return {
        "sync_type": sync_type,
        "color_code": color_code,
        "data_type": data_type,
        "data_type_name": _data_type_name(data_type),
        "slot_type_bits": slot_bits.to01(),
        "slot_type_hex": _bits_hex(slot_bits),
        "fec": {"golay_ok": True},
    }


def _lc_extra(
    slot_extra: dict,
    info196: bitarray,
    decoded96: bitarray,
    data12: bytes,
    flc: FullLinkControl,
    rs_ok: bool,
) -> dict:
    extra = dict(slot_extra)
    fec = dict(extra.get("fec", {}))
    fec.update({"bptc_196_96_ok": True, "rs_12_9_4_ok": rs_ok})
    extra.update({
        "fec": fec,
        "info196_hex": _bits_hex(info196),
        "decoded96_hex": _bits_hex(decoded96[0:96]),
        "lc_payload_hex": data12[:9].hex(),
        "rs_parity_hex": data12[9:12].hex(),
        "flc": flc.to_extra(),
    })
    return extra


def _csbk_extra(
    slot_extra: dict,
    info196: bitarray,
    decoded96: bitarray,
    csbk: CSBK,
) -> dict:
    extra = dict(slot_extra)
    fec = dict(extra.get("fec", {}))
    fec.update({"bptc_196_96_ok": True})
    extra.update({
        "fec": fec,
        "info196_hex": _bits_hex(info196),
        "decoded96_hex": _bits_hex(decoded96[0:96]),
        "csbk_payload_hex": _bits_hex(decoded96[0:96]),
        "csbk": csbk.to_extra(),
        "last_block": csbk.last_block,
    })
    return extra


def _decode_lc_or_terminator(
    ba264: bitarray,
    info196: bitarray,
    color_code: int,
    pdu_type: str,
    slot_extra: dict | None = None,
) -> dict | None:
    slot_extra = slot_extra or {"color_code": color_code}
    decoded = bptc_196_96_decode(info196, repair_if_necessary=True)
    data12  = decoded[0:96].tobytes()
    if not rs_12_9_4_check(data12, VLC_RS_MASK):
        return None
    try:
        flc = parse_full_link_control(decoded[0:96])
    except Exception:
        return None
    dst = flc.group_address or flc.target_address
    return {
        "type":     pdu_type,
        "src":      flc.source_address,
        "dst":      dst,
        "ts":       0,
        "flco":     flc.flco_name,
        "fid":      flc.fid_name,
        "extra":    _lc_extra(slot_extra, info196, decoded, data12, flc, True),
        "raw_bits": ba264.tobytes(),
    }


def _decode_csbk(
    ba264: bitarray,
    info196: bitarray,
    color_code: int,
    slot_extra: dict | None = None,
) -> dict | None:
    slot_extra = slot_extra or {"color_code": color_code}
    decoded = bptc_196_96_decode(info196, repair_if_necessary=True)
    csbk_bits = decoded[0:96]
    try:
        csbk = parse_csbk(csbk_bits)
    except Exception:
        return None
    return {
        "type":    "CSBK",
        "src":     csbk.source_address or 0,
        "dst":     csbk.target_address or 0,
        "ts":      0,
        "flco":    csbk.csbko_name,
        "fid":     csbk.fid_name,
        "extra":   _csbk_extra(slot_extra, info196, decoded, csbk),
        "raw_bits": ba264.tobytes(),
    }


class LateEntryCollector:
    """Stateful EMB fragment collector. Call feed() for each Voice Sync burst.
    After collecting First+Cont+Cont+Last (4 fragments), triggers VBPTC decode
    and returns a PDU dict. Call reset() to restart collection."""

    def __init__(self):
        self._frags: list = []
        self._embs: list[dict] = []
        self._collecting: bool = False

    def reset(self):
        self._frags = []
        self._embs = []
        self._collecting = False

    def feed(self, ba264: bitarray, sync_type: str) -> dict | None:
        """Feed one Voice Sync burst (264 bits). Returns PDU dict when 4 fragments
        are assembled and CS5 passes, otherwise None."""
        center = ba264[108:156]
        emb_bits   = center[0:8] + center[40:48]   # 16-bit EMB header
        signalling = center[8:40]                   # 32-bit fragment

        try:
            emb = parse_embedded_signalling(emb_bits)
        except Exception:
            return None
        lcss = emb.link_control_start_stop

        if not self._collecting:
            if lcss == LCSS.FirstFragmentLC:
                self._collecting = True
                self._frags = [signalling]
                self._embs = [emb.to_extra()]
        else:
            expected_cont = len(self._frags) < 3
            if expected_cont:
                if lcss != LCSS.ContinuationFragmentLCorCSBK:
                    # wrong LCSS mid-sequence, restart if it's a new First
                    self.reset()
                    if lcss == LCSS.FirstFragmentLC:
                        self._collecting = True
                        self._frags = [signalling]
                        self._embs = [emb.to_extra()]
                    return None
            else:
                # expecting LastFragmentLCorCSBK for the 4th fragment
                if lcss != LCSS.LastFragmentLCorCSBK:
                    self.reset()
                    if lcss == LCSS.FirstFragmentLC:
                        self._collecting = True
                        self._frags = [signalling]
                        self._embs = [emb.to_extra()]
                    return None
            self._frags.append(signalling)
            self._embs.append(emb.to_extra())
            if len(self._frags) == 4:
                return self._decode_assembled(ba264)
        return None

    def _decode_assembled(self, last_ba264: bitarray) -> dict | None:
        b128 = self._frags[0] + self._frags[1] + self._frags[2] + self._frags[3]
        emb_records = list(self._embs)
        self.reset()
        lc77 = vbptc_128_72_decode(b128, include_cs5=True)
        lc72 = lc77[0:72]
        rx_cs5 = ba2int(lc77[72:77])
        cs5_ok = (rx_cs5 <= 30) and five_bit_checksum_verify(lc72.tobytes(), rx_cs5)
        if not cs5_ok:
            return None
        try:
            flc = parse_full_link_control(lc77)
        except Exception:
            return None
        dst = flc.group_address or flc.target_address
        extra = {
            "cs5_ok": cs5_ok,
            "fec": {
                "vbptc_128_72_ok": True,
                "cs5_ok": cs5_ok,
                "emb_qr_ok_count": sum(1 for emb in emb_records if emb.get("emb_parity_ok")),
            },
            "embedded_signalling": emb_records,
            "fragment_count": len(emb_records),
            "embedded_lc_hex": lc72.tobytes().hex(),
            "cs5_value": rx_cs5,
            "flc": flc.to_extra(),
        }
        return {
            "type":    "LATE_ENTRY",
            "src":     flc.source_address,
            "dst":     dst,
            "ts":      0,
            "flco":    flc.flco_name,
            "fid":     flc.fid_name,
            "extra":   extra,
            "raw_bits": last_ba264.tobytes(),
        }
