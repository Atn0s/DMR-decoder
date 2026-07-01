from dpmr.cch import (
    air_interface_id_to_str,
    crc7,
    deinterleave_6x12,
    descramble,
    hamming_12_8_decode,
)


def test_crc7_known_vector():
    assert crc7([0] * 41) == 0
    assert crc7([1] + [0] * 40) == 0x3A


def test_hamming_12_8_decodes_and_corrects_single_bit():
    codeword = [0] * 12
    decoded, ok, corrected = hamming_12_8_decode(codeword)
    assert ok is True
    assert corrected == 0
    assert decoded == codeword[:8]

    damaged = codeword.copy()
    damaged[3] ^= 1
    decoded, ok, corrected = hamming_12_8_decode(damaged)
    assert ok is True
    assert corrected == 1
    assert decoded == codeword[:8]


def test_deinterleave_6x12_transposes_matrix():
    bits = list(range(72))
    out = deinterleave_6x12(bits)
    assert out[:12] == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66]
    assert out[12:24] == [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67]


def test_descramble_is_self_inverse_with_same_seed():
    bits = [0, 1, 1, 0, 1, 0, 0, 1] * 9
    scrambled = descramble(bits)
    assert descramble(scrambled) == bits


def test_air_interface_id_to_str():
    assert air_interface_id_to_str(0) == "0000000"
    assert air_interface_id_to_str(10) == "000000*"
    assert air_interface_id_to_str(1464100 * 10) == "*000000"
