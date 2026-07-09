import os
import wave
import numpy as np
import pytest
from scanner import detect_sample_rate, scan_file
import scanner
from common.config import DEFAULT_RADIO_CONFIG
from common.io import default_iq_scale, read_rawiq
from radio import output as radio_output
from radio.pdu import PDU


def test_detect_sample_rate_known():
    assert detect_sample_rate("data/dmr_1_78125.rawiq") == 78125
    assert detect_sample_rate("data/dmr_2_78125.rawiq") == 78125
    assert detect_sample_rate("data/dpmr_1_48000.rawiq") == 48000


def test_detect_sample_rate_unknown():
    assert detect_sample_rate("signal.rawiq") is None


def test_detect_sample_rate_mhz_suffix():
    assert detect_sample_rate("data/synthesized_wideband_2.5MHz.rawiq") == 2_500_000


def test_read_rawiq_scale_follows_dtype(tmp_path):
    path = tmp_path / "int8.rawiq"
    np.array([64, -64, 127, -128], dtype=np.int8).tofile(path)

    iq = read_rawiq(str(path), dtype="int8")

    assert default_iq_scale("int8") == 128.0
    assert np.allclose(iq, np.array([0.5 - 0.5j, 127 / 128 - 1j]))


def test_read_rawiq_accepts_stereo_pcm_wav(tmp_path):
    path = tmp_path / "sample.wav"
    frames = np.array(
        [
            [16384, -16384],
            [32767, -32768],
            [0, 8192],
        ],
        dtype="<i2",
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(frames.tobytes())

    iq = read_rawiq(str(path))

    assert np.allclose(iq, np.array([0.5 - 0.5j, 32767 / 32768 - 1j, 0 + 0.25j]))


def test_detect_sample_rate_reads_wav_header(tmp_path):
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(np.zeros((4, 2), dtype="<i2").tobytes())

    assert detect_sample_rate(str(path)) == 48_000


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
    results = scan_file(path, blind_search=True)
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


def test_scan_file_delegates_iq_processing_to_radio_pipeline(monkeypatch):
    iq = object()
    pdus = [{"protocol": "DMR", "type": "LC_HEADER"}]
    calls = []

    monkeypatch.setattr(scanner, "read_rawiq", lambda path, dtype="int16": iq)
    monkeypatch.setattr(scanner, "detect_sample_rate", lambda path: 48_000)
    monkeypatch.setattr(
        scanner.radio_output,
        "print_results",
        lambda result: calls.append(("print", result)),
    )

    def fake_scan_iq(iq_arg, sample_rate, freq_list, blind_search, protocol_names, radio_config):
        calls.append((
            "scan_iq",
            iq_arg is iq,
            sample_rate,
            freq_list,
            blind_search,
            protocol_names,
            radio_config,
        ))
        return pdus

    monkeypatch.setattr(scanner.radio_pipeline, "scan_iq", fake_scan_iq)

    result = scanner.scan_file(
        "example.rawiq",
        freq_list=[1000.0],
        protocol_names=["dmr"],
        blind_search=True,
    )

    assert result is pdus
    assert calls == [
        ("scan_iq", True, 48_000, [1000.0], True, ["dmr"], DEFAULT_RADIO_CONFIG),
        ("print", pdus),
    ]


def test_scan_file_warns_on_high_rate_baseband_no_results(monkeypatch, capsys):
    monkeypatch.setattr(scanner, "read_rawiq", lambda path, dtype="int16": np.ones(4, dtype=complex))
    monkeypatch.setattr(scanner, "detect_sample_rate", lambda path: 2_500_000)
    monkeypatch.setattr(scanner.radio_pipeline, "scan_iq", lambda *args, **kwargs: [])

    result = scanner.scan_file("wideband_2.5MHz.rawiq")

    assert result == []
    out = capsys.readouterr().out
    assert "high-rate IQ is being treated as centered baseband" in out
    assert "No PDUs decoded" in out
    assert "--blind-search" in out


def test_write_json_accepts_pdu_dataclass(tmp_path):
    out = str(tmp_path / "result.json")
    pdu = PDU.from_dict({
        "protocol": "DMR",
        "type": "LC_HEADER",
        "src": 1,
        "dst": 2,
        "raw_bits": b"abc",
        "_fo_hz": 1250.0,
    })

    radio_output.write_json([pdu], out)

    import json
    with open(out) as f:
        data = json.load(f)
    assert data == [
        {
            "protocol": "DMR",
            "type": "LC_HEADER",
            "src": 1,
            "dst": 2,
            "ts": None,
            "flco": "",
            "fid": "",
            "extra": {},
            "_fo_hz": 1250.0,
        }
    ]
