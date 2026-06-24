import numpy as np

import protocols


def test_decode_all_combines_dmr_and_p25(monkeypatch):
    def fake_dmr(y):
        return [{"protocol": "DMR", "type": "LC_HEADER", "src": 1, "dst": 2}]

    def fake_p25(y):
        return [{"protocol": "P25", "type": "P25_NID", "src": 0, "dst": 0}]

    monkeypatch.setattr(protocols, "decode_dmr", fake_dmr)
    monkeypatch.setattr(protocols, "decode_p25", fake_p25)

    result = protocols.decode_all(np.zeros(1000))

    assert [p["protocol"] for p in result] == ["DMR", "P25"]


def test_decode_dmr_adds_protocol_key(monkeypatch):
    def fake_loop(y):
        return [{"type": "CSBK", "src": 10, "dst": 20}]

    monkeypatch.setattr(protocols, "_dmr_decode_loop", fake_loop)

    result = protocols.decode_dmr(np.zeros(1000))

    assert result[0]["protocol"] == "DMR"
    assert result[0]["type"] == "CSBK"
