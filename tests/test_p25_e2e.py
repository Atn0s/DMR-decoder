import os

import pytest

import scanner

SAMPLE = "data/p25_1_78125.rawiq"


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample file absent")
def test_p25_sample_yields_multiple_nac_frames():
    pdus = scanner.scan_file(SAMPLE)
    p25 = [p for p in pdus if p.get("protocol") == "P25"]

    assert len(p25) > 5
    assert 0x293 in {p.get("extra", {}).get("nac") for p in p25}
    assert any(p["type"] == "P25_LDU1" and p["src"] == 1 and p["dst"] == 1 for p in p25)
