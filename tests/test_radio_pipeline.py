import numpy as np

from radio import pipeline
from radio.pdu import PDU


def test_process_candidate_tags_frequency_offset(monkeypatch):
    calls = []

    def fake_decode_iq(iq, enabled_protocols, sample_rate=0):
        calls.append((len(iq), enabled_protocols, sample_rate))
        return [{"protocol": "DMR", "type": "LC_HEADER"}]

    monkeypatch.setattr(pipeline.registry, "decode_iq_enabled", fake_decode_iq)

    result = pipeline.process_candidate(
        np.ones(8, dtype=complex),
        fo=1250.0,
        source_sample_rate=48_000.0,
        enabled_protocols={"DMR"},
    )

    assert calls == [(8, {"DMR"}, 48_000.0)]
    assert result == [{"protocol": "DMR", "type": "LC_HEADER", "_fo_hz": 1250.0}]


def test_process_candidate_tags_pdu_metadata(monkeypatch):
    def fake_decode_iq(iq, enabled_protocols, sample_rate=0):
        return [PDU.from_dict({"protocol": "DMR", "type": "LC_HEADER"})]

    monkeypatch.setattr(pipeline.registry, "decode_iq_enabled", fake_decode_iq)

    result = pipeline.process_candidate(
        np.ones(8, dtype=complex),
        fo=1250.0,
        source_sample_rate=48_000.0,
        enabled_protocols={"DMR"},
    )

    assert isinstance(result[0], PDU)
    assert result[0].meta == {"fo_hz": 1250.0}
    assert result[0].to_dict()["_fo_hz"] == 1250.0


def test_scan_iq_runs_narrowband_decode_postprocess_and_dedup(monkeypatch):
    calls = []

    def fake_decode_iq(iq, enabled_protocols, sample_rate=0):
        calls.append(("decode", len(iq), tuple(sorted(enabled_protocols)), sample_rate))
        return [{"protocol": "DMR", "type": "LC_HEADER"}]

    def fake_postprocess(pdus, enabled_protocols):
        calls.append(("postprocess", len(pdus), tuple(sorted(enabled_protocols))))
        return [dict(pdus[0], stable=True)]

    def fake_deduplicate(pdus):
        calls.append(("dedup", len(pdus)))
        return pdus

    monkeypatch.setattr(pipeline.registry, "decode_iq_enabled", fake_decode_iq)
    monkeypatch.setattr(pipeline.registry, "postprocess_pdus_enabled", fake_postprocess)
    monkeypatch.setattr(pipeline.registry, "deduplicate_pdus", fake_deduplicate)

    result = pipeline.scan_iq(
        np.ones(4, dtype=complex),
        sample_rate=48_000.0,
        protocol_names=["dmr"],
    )

    assert result == [{"protocol": "DMR", "type": "LC_HEADER", "stable": True}]
    assert calls == [
        ("decode", 4, ("DMR",), 48_000.0),
        ("postprocess", 1, ("DMR",)),
        ("dedup", 1),
    ]


def test_scan_iq_explicit_frequency_list_uses_candidate_path(monkeypatch):
    calls = []

    def fake_process_candidate(iq, fo, source_sample_rate, enabled_protocols, radio_config):
        calls.append((fo, source_sample_rate, tuple(sorted(enabled_protocols))))
        return [{"protocol": "P25", "type": f"candidate-{fo:g}"}]

    monkeypatch.setattr(pipeline, "process_candidate", fake_process_candidate)
    monkeypatch.setattr(pipeline.registry, "postprocess_pdus_enabled", lambda pdus, names: pdus)
    monkeypatch.setattr(pipeline.registry, "deduplicate_pdus", lambda pdus: pdus)

    result = pipeline.scan_iq(
        np.ones(4, dtype=complex),
        sample_rate=2_500_000.0,
        freq_list=[12_500.0],
        protocol_names={"p25"},
    )

    assert calls == [(12_500.0, 2_500_000.0, ("P25",))]
    assert result == [{"protocol": "P25", "type": "candidate-12500"}]


def test_scan_iq_requires_sample_rate():
    try:
        pipeline.scan_iq(np.ones(4, dtype=complex), sample_rate=None)
    except ValueError as exc:
        assert "sample_rate is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_scan_iq_blind_search_uses_candidate_path(monkeypatch):
    calls = []

    monkeypatch.setattr(pipeline, "psd_blind_search", lambda iq, fs, radio_config: [1000.0])

    def fake_process_candidate(iq, fo, source_sample_rate, enabled_protocols, radio_config):
        calls.append((fo, source_sample_rate, tuple(sorted(enabled_protocols))))
        return [{"protocol": "DMR", "type": "candidate"}]

    monkeypatch.setattr(pipeline, "process_candidate", fake_process_candidate)
    monkeypatch.setattr(pipeline.registry, "postprocess_pdus_enabled", lambda pdus, names: pdus)
    monkeypatch.setattr(pipeline.registry, "deduplicate_pdus", lambda pdus: pdus)

    result = pipeline.scan_iq(
        np.ones(4, dtype=complex),
        sample_rate=2_500_000.0,
        blind_search=True,
        protocol_names={"dmr"},
    )

    assert calls == [(1000.0, 2_500_000.0, ("DMR",))]
    assert result == [{"protocol": "DMR", "type": "candidate"}]
