from dpmr.color_code import get_color_code


def _bits(value: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(23, -1, -1)]


def test_get_color_code_known_values():
    assert get_color_code(_bits(0x575F77)) == 0
    assert get_color_code(_bits(0x57DD75)) == 2
    assert get_color_code(_bits(0xFDFD77)) == 63


def test_get_color_code_masks_dibit_lsb_noise():
    assert get_color_code(_bits(0x000000)) == -1
    assert get_color_code(_bits(0x57DD20)) == 2
