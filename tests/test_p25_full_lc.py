from bitarray import bitarray

from p25.constants import (
    ES_HEXBIT_POSITIONS,
    HDU_DATA_HEXBIT_POSITIONS,
    HDU_GOLAY_PARITY_POSITIONS,
    LC_HEXBIT_POSITIONS,
)
from p25.fec import (
    bch_63_16_decode,
    bch_63_16_encode,
    crc16_ccitt,
    golay_24_6_encode,
    hamming_10_6_3_decode,
    rs_24_16_9_encode,
    rs_36_20_17_encode,
    rs_24_12_13_decode,
    rs_24_12_13_encode,
)
from p25.hdu_decode import decode_hdu_hcw
from p25.ldu2_decode import decode_ldu2_es
from p25.lc_decode import decode_ldu1_lc
from p25.link_control import LinkControl, parse_link_control
from p25.session import P25SessionAssembler
from p25.framing import frame_info_from_nid
from p25.nid import decode_nid


def _hamming_encode(d6: list[int]) -> bitarray:
    p0 = d6[0] ^ d6[1] ^ d6[2] ^ d6[5]
    p1 = d6[0] ^ d6[1] ^ d6[3] ^ d6[5]
    p2 = d6[0] ^ d6[2] ^ d6[3] ^ d6[4]
    p3 = d6[1] ^ d6[2] ^ d6[3] ^ d6[4]
    out = bitarray(endian="big")
    out.extend("".join(str(x) for x in d6 + [p0, p1, p2, p3]))
    return out


def _frame(nac: int, duid: int):
    b = bitarray(endian="big")
    b.extend(f"{nac:012b}{duid:04b}")
    b.extend("0" * 48)
    return frame_info_from_nid(decode_nid(b))


def test_p25_fec_key_paths():
    hcw = _hamming_encode([1, 0, 1, 1, 0, 0])
    hcw[2] ^= 1
    data, corrected = hamming_10_6_3_decode(hcw)
    assert data.to01() == "101100"
    assert corrected is True

    rs_data = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    cw = rs_24_12_13_encode(rs_data)
    for idx in (0, 3, 7, 11, 18, 23):
        cw[idx] ^= 0x2A
    out, ok = rs_24_12_13_decode(cw)
    assert ok is True
    assert out == rs_data

    info = bitarray("0010100100110101")
    bch = bch_63_16_encode(info)
    bch[20] ^= 1
    decoded, ok = bch_63_16_decode(bch)
    assert ok is True
    assert decoded.to01() == info.to01()

    bch = bch_63_16_encode(info)
    bch[2] ^= 1
    decoded, corrected = bch_63_16_decode(bch)
    assert corrected is True
    assert decoded.to01() == info.to01()

    crc_bits = bitarray(endian="big")
    crc_bits.frombytes(b"123456789")
    assert crc16_ccitt(crc_bits) == 0x29B1


def test_decode_ldu1_lc_full_chain():
    lc = bitarray(endian="big")
    lc.extend(f"{0x00:08b}{0x00:08b}{0x00:08b}{0x00:08b}{58:016b}{1234567:024b}")
    lc.extend("0" * (72 - len(lc)))
    data_hexbits = [int(lc[i * 6:(i + 1) * 6].to01(), 2) for i in range(12)]
    data_hexbits = list(reversed(data_hexbits))
    cw = rs_24_12_13_encode(data_hexbits)

    encoded = bitarray(endian="big")
    for hx in cw:
        encoded.extend(_hamming_encode([(hx >> (5 - k)) & 1 for k in range(6)]))

    frame = bitarray(1728, endian="big")
    frame.setall(0)
    for i, pos in enumerate(LC_HEXBIT_POSITIONS):
        frame[pos] = encoded[i]

    out = decode_ldu1_lc(frame)

    assert out is not None
    assert out.src == 1234567
    assert out.tgid == 58
    assert out.is_group is True


def test_link_control_and_session_call_pdu():
    raw = bitarray(endian="big")
    raw.extend(f"{0x00:08b}{0x00:08b}{0x00:08b}{0x00:08b}{58:016b}{111:024b}")
    raw.extend("0" * (72 - len(raw)))
    lc = parse_link_control(raw)
    assert isinstance(lc, LinkControl)
    assert lc.src == 111
    assert lc.dst == 58
    assert lc.emergency is False
    assert lc.reserved_bits == 15
    assert lc.reserved == 0

    session = P25SessionAssembler()
    assert session.feed(_frame(0x293, 0x5), lc, fs_start=0) is None
    assert session.feed(_frame(0x293, 0xA), None, fs_start=8640) is None
    pdu = session.feed(_frame(0x293, 0x3), None, fs_start=17280)
    assert pdu is not None
    assert pdu["type"] == "P25_CALL"
    assert pdu["src"] == 111
    assert pdu["dst"] == 58


def test_link_control_unit_to_unit_voice():
    raw = bitarray(endian="big")
    raw.extend(f"{0x03:08b}{0x00:08b}{0x40:08b}{222:024b}{111:024b}")
    lc = parse_link_control(raw)

    assert lc is not None
    assert lc.src == 111
    assert lc.dst == 222
    assert lc.tgid == 0
    assert lc.svc == 0x40
    assert lc.lc_info == 0x4000 | ((222 >> 16) & 0xFF)
    assert lc.emergency is None
    assert lc.reserved_bits == 8
    assert lc.reserved == 0x40
    assert lc.is_group is False
    assert lc.call_type == "unit_to_unit"


def test_link_control_group_voice_emergency_and_reserved():
    raw = bitarray(endian="big")
    raw.extend(f"{0x00:08b}{0x00:08b}{0x80:08b}{0x12:08b}{58:016b}{111:024b}")
    lc = parse_link_control(raw)

    assert lc is not None
    assert lc.emergency is True
    assert lc.reserved_bits == 15
    assert lc.reserved == 0x12
    assert lc.tgid == 58
    assert lc.src == 111


def test_decode_hdu_header_code_word_full_chain():
    hcw = bitarray(endian="big")
    hcw.extend(f"{0x0123456789ABCDEF12:072b}")
    hcw.extend(f"{0x00:08b}{0x80:08b}{0x1234:016b}{58:016b}")
    symbols = [int(hcw[i * 6:(i + 1) * 6].to01(), 2) for i in range(20)]
    cw = rs_36_20_17_encode(list(reversed(symbols)))

    frame = bitarray(1728, endian="big")
    frame.setall(0)
    for i, sym in enumerate(cw):
        data = bitarray(f"{sym:06b}", endian="big")
        parity = golay_24_6_encode(sym)
        for j, pos in enumerate(HDU_DATA_HEXBIT_POSITIONS[i * 6:(i + 1) * 6]):
            frame[pos] = data[j]
        for j, pos in enumerate(HDU_GOLAY_PARITY_POSITIONS[i * 12:(i + 1) * 12]):
            frame[pos] = parity[j]

    out = decode_hdu_hcw(frame)

    assert out is not None
    assert out.mi == 0x0123456789ABCDEF12
    assert out.mfid == 0x00
    assert out.algid == 0x80
    assert out.kid == 0x1234
    assert out.tgid == 58


def test_decode_ldu2_encryption_sync_full_chain():
    es = bitarray(endian="big")
    es.extend(f"{0x0123456789ABCDEF12:072b}{0x80:08b}{0x1234:016b}")
    symbols = [int(es[i * 6:(i + 1) * 6].to01(), 2) for i in range(16)]
    cw = rs_24_16_9_encode(list(reversed(symbols)))

    encoded = bitarray(endian="big")
    for hx in cw:
        encoded.extend(_hamming_encode([(hx >> (5 - k)) & 1 for k in range(6)]))

    frame = bitarray(1728, endian="big")
    frame.setall(0)
    for i, pos in enumerate(ES_HEXBIT_POSITIONS):
        frame[pos] = encoded[i]

    out = decode_ldu2_es(frame)

    assert out is not None
    assert out.mi == 0x0123456789ABCDEF12
    assert out.algid == 0x80
    assert out.kid == 0x1234
