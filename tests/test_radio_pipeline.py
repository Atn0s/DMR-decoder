import numpy as np

from radio import pipeline


def test_process_candidate_tags_frequency_offset(monkeypatch):
    calls = []

    def fake_decode_iq(iq, protocol_names=None, sample_rate=0):
        calls.append((len(iq), protocol_names, sample_rate))
        return [{"protocol": "DMR", "type": "LC_HEADER"}]

    monkeypatch.setattr(pipeline.protocols, "decode_iq", fake_decode_iq)

    result = pipeline.process_candidate(
        np.ones(8, dtype=complex),
        fo=1250.0,
        source_sample_rate=48_000.0,
        protocol_names={"dmr"},
    )

    assert calls == [(8, {"dmr"}, 48_000.0)]
    assert result == [{"protocol": "DMR", "type": "LC_HEADER", "_fo_hz": 1250.0}]


def test_scan_iq_runs_narrowband_decode_postprocess_and_dedup(monkeypatch):
    calls = []

    def fake_decode_iq(iq, protocol_names=None, sample_rate=0):
        calls.append(("decode", len(iq), tuple(sorted(protocol_names)), sample_rate))
        return [{"protocol": "DMR", "type": "LC_HEADER"}]

    def fake_postprocess(pdus, protocol_names=None):
        calls.append(("postprocess", len(pdus), tuple(sorted(protocol_names))))
        return [dict(pdus[0], stable=True)]

    def fake_deduplicate(pdus):
        calls.append(("dedup", len(pdus)))
        return pdus

    monkeypatch.setattr(pipeline.protocols, "decode_iq", fake_decode_iq)
    monkeypatch.setattr(pipeline.protocols, "postprocess_pdus", fake_postprocess)
    monkeypatch.setattr(pipeline.protocols, "deduplicate_pdus", fake_deduplicate)

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

    def fake_process_candidate(iq, fo, source_sample_rate, protocol_names, radio_config):
        calls.append((fo, source_sample_rate, tuple(sorted(protocol_names))))
        return [{"protocol": "P25", "type": f"candidate-{fo:g}"}]

    monkeypatch.setattr(pipeline, "process_candidate", fake_process_candidate)
    monkeypatch.setattr(pipeline.protocols, "postprocess_pdus", lambda pdus, names: pdus)
    monkeypatch.setattr(pipeline.protocols, "deduplicate_pdus", lambda pdus: pdus)

    result = pipeline.scan_iq(
        np.ones(4, dtype=complex),
        sample_rate=None,
        freq_list=[12_500.0],
        protocol_names={"p25"},
    )

    assert calls == [(12_500.0, 2_500_000.0, ("P25",))]
    assert result == [{"protocol": "P25", "type": "candidate-12500"}]
