from core.burst_type import SlotDataType, SYNC_TEMPLATES, SPS

def test_slot_data_type_values():
    assert SlotDataType.VOICE_LC_HEADER.value == 1
    assert SlotDataType.TERMINATOR_WITH_LC.value == 2
    assert SlotDataType.CSBK.value == 3

def test_sync_templates_shape():
    import numpy as np
    for key in ("MS_VOICE", "BS_VOICE", "DATA_MS", "DATA_BS"):
        assert key in SYNC_TEMPLATES
        assert len(SYNC_TEMPLATES[key]) == 24  # 24 symbols

def test_sps():
    assert SPS == 10
