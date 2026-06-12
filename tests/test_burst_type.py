from core.burst_type import SlotDataType, SYNC_TEMPLATES, SPS, _hex_to_symbols

def test_slot_data_type_values():
    assert SlotDataType.VOICE_LC_HEADER.value == 1
    assert SlotDataType.TERMINATOR_WITH_LC.value == 2
    assert SlotDataType.CSBK.value == 3

def test_sync_templates_shape():
    import numpy as np
    for key in ("MS_VOICE", "BS_VOICE", "DATA_MS", "DATA_BS"):
        assert key in SYNC_TEMPLATES
        assert len(SYNC_TEMPLATES[key]) == 24  # 24 symbols

    # Verify first 4 symbol values of BS_VOICE sync word "755FD7DF75F7":
    # hex "7" = 0111 -> dibits "01","11" -> symbols +3, -3
    # hex "5" = 0101 -> dibits "01","01" -> symbols +3, +3
    assert list(SYNC_TEMPLATES["BS_VOICE"][0:4]) == [3, -3, 3, 3]

def test_hex_to_symbols_all_dibits():
    # "0" (0000) -> dibits "00","00" -> [+1, +1]
    assert list(_hex_to_symbols("0")) == [1, 1]
    # "A" (1010) -> dibits "10","10" -> [-1, -1]
    assert list(_hex_to_symbols("A")) == [-1, -1]
    # "5" (0101) -> dibits "01","01" -> [+3, +3]
    assert list(_hex_to_symbols("5")) == [3, 3]
    # "F" (1111) -> dibits "11","11" -> [-3, -3]
    assert list(_hex_to_symbols("F")) == [-3, -3]

def test_sps():
    assert SPS == 10
