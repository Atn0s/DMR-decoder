from dataclasses import replace

import numpy as np

from dmr.config import DMRConfig
from dpmr.config import DPMRConfig
from p25.config import P25Config
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


def test_decode_all_uses_registered_dpmr_frontend(monkeypatch):
    dmr_frontend = np.zeros(3)
    dpmr_frontend = np.ones(3)

    monkeypatch.setattr(protocols, "decode_dmr", lambda y: [])
    monkeypatch.setattr(protocols, "decode_p25", lambda y: [])
    monkeypatch.setattr(
        protocols,
        "decode_dpmr",
        lambda y: [{"protocol": "dPMR", "sum": int(np.sum(y))}],
    )

    result = protocols.decode_all(
        dmr_frontend,
        protocol_names={"dpmr"},
        frontends={"dPMR": dpmr_frontend},
    )

    assert result == [{"protocol": "dPMR", "sum": 3}]


def test_normalize_protocol_names_accepts_aliases():
    assert protocols.normalize_protocol_names(["dmr", "P25", "dpmr"]) == {
        "DMR",
        "P25",
        "dPMR",
    }


def test_decode_dmr_adds_protocol_key(monkeypatch):
    def fake_loop(y):
        return [{"type": "CSBK", "src": 10, "dst": 20}]

    monkeypatch.setattr(protocols, "_dmr_decode_loop", fake_loop)

    result = protocols.decode_dmr(np.zeros(1000))

    assert result[0]["protocol"] == "DMR"
    assert result[0]["type"] == "CSBK"


def test_protocol_dedup_key_is_protocol_aware():
    p25 = {
        "protocol": "P25",
        "type": "P25_LDU1",
        "extra": {"nac": 0x293, "fs_start": 8641},
    }
    dmr = {
        "protocol": "DMR",
        "type": "LC_HEADER",
        "src": 1,
        "dst": 1,
        "_fo_hz": 1200,
    }
    dpmr = {
        "protocol": "dPMR",
        "type": "DPMR_VOICE",
        "src": "1",
        "dst": "2",
        "extra": {"color_code": 2, "fs_start": 3841},
    }

    assert protocols.dedup_key(p25) == ("P25", 0x293, "P25_LDU1", 1)
    assert protocols.dedup_key(dmr) == ("DMR", 1, 1, "LC_HEADER", 0)
    assert protocols.dedup_key(dpmr) == ("dPMR", "1", "2", 2, 1)


def test_protocol_spec_exposes_formatter_and_dedup_key():
    spec = protocols.spec_for_protocol("p25")
    pdu = {
        "protocol": "P25",
        "type": "P25_NID",
        "src": 0,
        "dst": 0,
        "flco": "LDU1",
        "fid": "",
        "extra": {"nac": 0x293, "duid": 0x5, "fs_start": 8641},
    }

    assert spec.dedup_key(pdu) == ("P25", 0x293, "P25_NID", 1)
    assert spec.formatter(pdu, "").startswith("[P25_NID")


def test_decode_iq_uses_protocol_frontends(monkeypatch):
    calls = []

    def shared_frontend(iq, sample_rate, config):
        calls.append(("shared", sample_rate, config))
        return np.ones(3)

    def dpmr_frontend(iq, sample_rate, config):
        calls.append(("dpmr", sample_rate, config))
        return np.full(3, 2)

    patched = []
    for spec in protocols.PROTOCOL_REGISTRY:
        if spec.name in {"DMR", "P25"}:
            patched.append(
                replace(
                    spec,
                    config=f"{spec.name}-config",
                    frontend_key="shared",
                    frontend=shared_frontend,
                )
            )
        else:
            patched.append(
                replace(
                    spec,
                    config=f"{spec.name}-config",
                    frontend_key="dpmr",
                    frontend=dpmr_frontend,
                )
            )

    monkeypatch.setattr(protocols, "PROTOCOL_REGISTRY", tuple(patched))
    monkeypatch.setattr(
        protocols,
        "decode_dmr",
        lambda y, config: [{"protocol": "DMR", "sum": int(np.sum(y)), "config": config}],
    )
    monkeypatch.setattr(
        protocols,
        "decode_p25",
        lambda y, config: [{"protocol": "P25", "sum": int(np.sum(y)), "config": config}],
    )
    monkeypatch.setattr(
        protocols,
        "decode_dpmr",
        lambda y, config: [{"protocol": "dPMR", "sum": int(np.sum(y)), "config": config}],
    )

    result = protocols.decode_iq(np.zeros(5), sample_rate=123.0)

    assert calls == [("shared", 123.0, "DMR-config"), ("dpmr", 123.0, "dPMR-config")]
    assert result == [
        {"protocol": "DMR", "sum": 3, "config": "DMR-config"},
        {"protocol": "P25", "sum": 3, "config": "P25-config"},
        {"protocol": "dPMR", "sum": 6, "config": "dPMR-config"},
    ]


def test_postprocess_pdus_uses_protocol_specs(monkeypatch):
    def mark_dpmr(pdus):
        return [dict(pdu, postprocessed=True) for pdu in pdus]

    patched = [
        replace(spec, postprocess=mark_dpmr) if spec.name == "dPMR" else spec
        for spec in protocols.PROTOCOL_REGISTRY
    ]
    monkeypatch.setattr(protocols, "PROTOCOL_REGISTRY", tuple(patched))

    result = protocols.postprocess_pdus(
        [{"protocol": "dPMR", "type": "DPMR_VOICE"}],
        protocol_names={"dpmr"},
    )

    assert result == [{"protocol": "dPMR", "type": "DPMR_VOICE", "postprocessed": True}]


def test_protocol_decoder_wrappers_pass_config(monkeypatch):
    calls = []

    def fake_dmr_loop(y, config):
        calls.append((
            "dmr",
            config.sync_threshold_voice,
            config.sync_threshold_data,
            config.voice_burst_stride_samples,
        ))
        return []

    def fake_p25(y, sps=10, sync_threshold=0.62):
        calls.append(("p25", sps, sync_threshold))
        return []

    def fake_dpmr(y, sync_threshold=0.82):
        calls.append(("dpmr", sync_threshold))
        return []

    monkeypatch.setattr(protocols, "_dmr_decode_loop", fake_dmr_loop)
    monkeypatch.setattr(protocols, "_decode_p25", fake_p25)
    monkeypatch.setattr(protocols, "_decode_dpmr", fake_dpmr)

    protocols.decode_dmr(
        np.zeros(3),
        DMRConfig(
            sync_threshold_voice=0.71,
            sync_threshold_data=0.58,
            voice_burst_stride_samples=4320,
        ),
    )
    protocols.decode_p25(np.zeros(3), P25Config(samples_per_symbol=8, sync_threshold=0.7))
    protocols.decode_dpmr(np.zeros(3), DPMRConfig(sync_threshold=0.9))

    assert calls == [("dmr", 0.71, 0.58, 4320), ("p25", 8, 0.7), ("dpmr", 0.9)]


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
                "sync_type": "FS2",
                "cch": [
                    {
                        "frame_number": 2,
                        "id_half": 0x7FF,
                        "communication_mode": 0,
                        "version": 3,
                        "comms_format": 1,
                        "emergency_priority": 0,
                        "reserved": 0,
                        "slow_data": 0x12345,
                    },
                    {
                        "frame_number": 3,
                        "id_half": 0xFFF,
                        "communication_mode": 0,
                        "version": 3,
                        "comms_format": 1,
                        "emergency_priority": 0,
                        "reserved": 0,
                        "slow_data": 0x23456,
                    },
                ],
            },
        }
    ])

    out = capsys.readouterr().out
    assert "DPMR_VOICE" in out
    assert "PROTO=dPMR" in out
    assert "CC=02" in out
    assert "SYNC=FS2" in out
    assert "POL=INV" in out
    assert "QUAL=" not in out
    assert "CRC=" not in out
    assert "HAM=" not in out
    assert "E90=" not in out
    assert "AMB=" not in out
    assert "FN=2" in out
    assert "IDH=0x7FF" in out
    assert "M=0" in out
    assert "V=3" in out
    assert "F=1" in out
    assert "RES=0" in out
    assert "SLD=0x12345" in out


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
                "cch": [
                    {
                        "frame_number": 0,
                        "id_half": 0x123,
                        "communication_mode": 1,
                        "version": 0,
                        "comms_format": 2,
                        "emergency_priority": 1,
                        "reserved": 0,
                        "slow_data": 0,
                    }
                ],
            },
        }
    ])

    out = capsys.readouterr().out
    assert "DPMR_HEADER" in out
    assert "PROTO=dPMR" in out
    assert "SYNC=FS1" in out
    assert "CC=--" in out
    assert "FN=0" in out
    assert "IDH=0x123" in out
    assert "M=1" in out
    assert "F=2" in out
    assert "E=1" in out
