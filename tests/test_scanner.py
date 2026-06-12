import os
import pytest
from scanner import detect_sample_rate, scan_file


def test_detect_sample_rate_known():
    assert detect_sample_rate("data/dmr_1_78125.rawiq") == 78125
    assert detect_sample_rate("data/dmr_2_78125.rawiq") == 78125


def test_detect_sample_rate_unknown():
    assert detect_sample_rate("data/synthesized_wideband_2.5MHz.rawiq") is None
    assert detect_sample_rate("signal.rawiq") is None


def test_scan_file_returns_list():
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")
    results = scan_file(path)
    assert isinstance(results, list)


def test_scan_file_pdu_schema():
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")
    results = scan_file(path)
    for pdu in results:
        for key in ("type", "src", "dst", "ts", "flco", "extra", "raw_bits"):
            assert key in pdu, f"Missing key '{key}' in PDU: {pdu}"


def test_scan_file_wideband():
    path = "data/synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")
    results = scan_file(path)
    assert isinstance(results, list)
    types = [r["type"] for r in results]
    assert any(t in ("LC_HEADER", "LATE_ENTRY", "CSBK", "TERMINATOR") for t in types), \
        f"Expected at least one DMR PDU, got types: {types}"
    non_trivial = [r for r in results if r["src"] != 0 or r["dst"] != 0]
    assert len(non_trivial) > 0, f"All PDUs have zero src/dst, likely spurious: {results}"


def test_scan_file_json_output(tmp_path):
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")
    import json
    out = str(tmp_path / "result.json")
    results = scan_file(path, output_json=out)
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == len(results)
    # raw_bits should NOT appear in JSON output
    for item in data:
        assert "raw_bits" not in item
