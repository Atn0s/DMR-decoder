from bitarray import bitarray

from p25.nid import P25NID, decode_nid


def make_nid_bits(nac: int, duid: int) -> bitarray:
    bits = bitarray(endian="big")
    bits.extend(f"{nac:012b}{duid:04b}")
    bits.extend("0" * 48)
    return bits


def test_decode_nid_extracts_nac_and_duid_schema():
    nid = decode_nid(make_nid_bits(0x293, 0x5))

    assert isinstance(nid, P25NID)
    assert nid.nac == 0x293
    assert nid.duid == 0x5
    assert nid.duid_name == "LDU1"
    assert nid.corrected is False
    assert nid.valid_bch is None


def test_decode_nid_rejects_wrong_length():
    short_bits = bitarray("0" * 63)
    try:
        decode_nid(short_bits)
    except ValueError as exc:
        assert "64 bits" in str(exc)
    else:
        raise AssertionError("decode_nid should reject non-64-bit input")


def test_decode_nid_names_unknown_duid():
    nid = decode_nid(make_nid_bits(0x123, 0x2))
    assert nid.duid_name == "UNKNOWN_0x2"
