import numpy as np

import protocols


def test_decode_all_combines_dmr_and_p25(monkeypatch):
    def fake_dmr(y):
        return [{"protocol": "DMR", "type": "LC_HEADER", "src": 1, "dst": 2}]

    def fake_p25(y):
        return [{"protocol": "P25", "type": "P25_NID", "src": 0, "dst": 0}]

    def fake_dpmr(y):
        return [{"protocol": "dPMR", "type": "DPMR_VOICE", "src": "", "dst": ""}]

    monkeypatch.setattr(protocols, "decode_dmr", fake_dmr)
    monkeypatch.setattr(protocols, "decode_p25", fake_p25)
    monkeypatch.setattr(protocols, "decode_dpmr", fake_dpmr)

    result = protocols.decode_all(np.zeros(1000))

    assert [p["protocol"] for p in result] == ["DMR", "P25", "dPMR"]


def test_decode_dmr_adds_protocol_key(monkeypatch):
    def fake_loop(y):
        return [{"type": "CSBK", "src": 10, "dst": 20}]

    monkeypatch.setattr(protocols, "_dmr_decode_loop", fake_loop)

    result = protocols.decode_dmr(np.zeros(1000))

    assert result[0]["protocol"] == "DMR"
    assert result[0]["type"] == "CSBK"


def test_print_results_accepts_p25_nid(capsys):
    import scanner

    scanner._print_results([
        {
            "protocol": "P25",
            "type": "P25_NID",
            "src": 0,
            "dst": 0,
            "flco": "LDU1",
            "fid": "",
            "extra": {"nac": 0x293, "duid": 0x5},
        }
    ])

    out = capsys.readouterr().out
    assert "P25_NID" in out
    assert "PROTO=P25" in out
    assert "NAC=0x293" in out


def test_print_results_accepts_dpmr_voice(capsys):
    import scanner

    scanner._print_results([
        {
            "protocol": "dPMR",
            "type": "DPMR_VOICE",
            "src": "3939*5*",
            "dst": "3939*5*",
            "flco": "VOICE",
            "fid": "",
            "extra": {
                "color_code": 2,
                "polarity_inverted": True,
                "segment_timing": {
                    "cc": {"decision_error_p90": 0.45, "ambiguous_symbols": 12}
                },
            },
        }
    ])

    out = capsys.readouterr().out
    assert "DPMR_VOICE" in out
    assert "PROTO=dPMR" in out
    assert "CC=02" in out
    assert "POL=INV" in out
    assert "QUAL=" in out
    assert "E90=0.45" in out
    assert "AMB=12" in out


def test_print_results_accepts_dpmr_header(capsys):
    import scanner

    scanner._print_results([
        {
            "protocol": "dPMR",
            "type": "DPMR_HEADER",
            "src": "",
            "dst": "1374803",
            "flco": "HEADER",
            "fid": "",
            "extra": {
                "color_code": -1,
                "polarity_inverted": False,
                "sync_type": "FS1",
                "quality": {"front_end_confidence": "high", "crc_ok_count": 2, "hamming_ok_count": 2},
                "segment_timing": {
                    "header": {"decision_error_p90": 0.12, "ambiguous_symbols": 0}
                },
            },
        }
    ])

    out = capsys.readouterr().out
    assert "DPMR_HEADER" in out
    assert "PROTO=dPMR" in out
    assert "SYNC=FS1" in out
    assert "CC=--" in out
