from dataclasses import replace

import numpy as np

from dmr.config import DMRConfig
import dmr.plugin as dmr_plugin
from dpmr.config import DPMRConfig
import dpmr.plugin as dpmr_plugin
from p25.config import P25Config
import p25.plugin as p25_plugin
import dmr.decode_flow as dmr_decode_flow
from radio import registry
from radio import output as radio_output
from radio.pdu import PDU


def test_normalize_protocol_names_accepts_aliases():
    assert registry.normalize_protocol_names(["dmr", "P25", "dpmr"]) == {
        "DMR",
        "P25",
        "dPMR",
    }


def test_protocol_registry_uses_protocol_plugins():
    assert registry.PROTOCOL_REGISTRY == (
        dmr_plugin.SPEC,
        p25_plugin.SPEC,
        dpmr_plugin.SPEC,
    )


def test_decode_dmr_adds_protocol_key(monkeypatch):
    def fake_loop(y, config):
        return [{"type": "CSBK", "src": 10, "dst": 20}]

    monkeypatch.setattr(dmr_decode_flow, "decode_dmr_flow", fake_loop)

    result = dmr_plugin.decode(np.zeros(1000))

    assert result[0]["protocol"] == "DMR"
    assert result[0]["type"] == "CSBK"
    assert result[0]["ts"] is None
    assert result[0]["extra"] == {}


def test_protocol_plugin_decodes_normalize_schema_defaults(monkeypatch):
    monkeypatch.setattr(
        p25_plugin,
        "_decode_p25",
        lambda *args, **kwargs: [{"protocol": "P25", "type": "P25_NID"}],
    )
    monkeypatch.setattr(
        dpmr_plugin,
        "_decode_dpmr",
        lambda *args, **kwargs: [{"protocol": "dPMR", "type": "DPMR_VOICE"}],
    )

    p25_result = p25_plugin.decode(np.zeros(3))[0]
    dpmr_result = dpmr_plugin.decode(np.zeros(3))[0]

    assert p25_result["src"] == 0
    assert p25_result["dst"] == 0
    assert p25_result["extra"] == {}
    assert dpmr_result["src"] == 0
    assert dpmr_result["dst"] == 0
    assert dpmr_result["extra"] == {}


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

    assert registry.dedup_key(p25) == ("P25", 0x293, "P25_LDU1", 1)
    assert registry.dedup_key(dmr) == ("DMR", 1, 1, "LC_HEADER", 0)
    assert registry.dedup_key(dpmr) == ("dPMR", "1", "2", 2, 1)


def test_protocol_boundaries_accept_pdu_dataclass():
    pdu = PDU.from_dict({
        "protocol": "DMR",
        "type": "LC_HEADER",
        "src": 1,
        "dst": 2,
        "ts": 0,
        "flco": "GroupVoiceChannelUser",
        "fid": "FID",
        "extra": {},
        "_fo_hz": 1250.0,
    })

    assert registry.dedup_key(pdu) == ("DMR", 1, 2, "LC_HEADER", 0)
    assert registry.format_pdu(pdu).endswith("FID=FID (fo=+1.2kHz)")


def test_protocol_spec_exposes_formatter_and_dedup_key():
    spec = registry.spec_for_protocol("p25")
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
    for spec in registry.PROTOCOL_REGISTRY:
        if spec.name in {"DMR", "P25"}:
            patched.append(
                replace(
                    spec,
                    config=f"{spec.name}-config",
                    frontend_key="shared",
                    frontend=shared_frontend,
                    decode=lambda y, config: [{"protocol": config.split("-")[0], "sum": int(np.sum(y)), "config": config}],
                )
            )
        else:
            patched.append(
                replace(
                    spec,
                    config=f"{spec.name}-config",
                    frontend_key="dpmr",
                    frontend=dpmr_frontend,
                    decode=lambda y, config: [{"protocol": "dPMR", "sum": int(np.sum(y)), "config": config}],
                )
            )

    monkeypatch.setattr(registry, "PROTOCOL_REGISTRY", tuple(patched))

    result = registry.decode_iq(np.zeros(5), sample_rate=123.0)

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
        for spec in registry.PROTOCOL_REGISTRY
    ]
    monkeypatch.setattr(registry, "PROTOCOL_REGISTRY", tuple(patched))

    result = registry.postprocess_pdus(
        [{"protocol": "dPMR", "type": "DPMR_VOICE"}],
        protocol_names={"dpmr"},
    )

    assert result == [{"protocol": "dPMR", "type": "DPMR_VOICE", "postprocessed": True}]


def test_dpmr_postprocess_uses_configured_stable_color_repeats(monkeypatch):
    calls = []

    def fake_filter(pdus, min_repeats):
        calls.append((pdus, min_repeats))
        return pdus

    monkeypatch.setattr(dpmr_plugin, "filter_stable_pdus", fake_filter)

    pdus = [{"protocol": "dPMR", "type": "DPMR_VOICE"}]

    assert dpmr_plugin.postprocess(pdus) == pdus
    assert calls == [(pdus, dpmr_plugin.DEFAULT_DPMR_CONFIG.stable_color_min_repeats)]


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

    def fake_p25(
        y,
        sps=10,
        sync_threshold=0.62,
        sync_min_distance_symbols=120,
        stable_nac_min_count=5,
        stable_nac_min_ratio=0.4,
    ):
        calls.append((
            "p25",
            sps,
            sync_threshold,
            sync_min_distance_symbols,
            stable_nac_min_count,
            stable_nac_min_ratio,
        ))
        return []

    def fake_dpmr(y, config=None):
        calls.append((
            "dpmr",
            config.sync_threshold,
            config.sps_search_min,
            config.sps_search_max,
            config.voice_symbol_candidate_limit,
        ))
        return []

    monkeypatch.setattr(dmr_decode_flow, "decode_dmr_flow", fake_dmr_loop)
    monkeypatch.setattr(p25_plugin, "_decode_p25", fake_p25)
    monkeypatch.setattr(dpmr_plugin, "_decode_dpmr", fake_dpmr)

    dmr_plugin.decode(
        np.zeros(3),
        DMRConfig(
            sync_threshold_voice=0.71,
            sync_threshold_data=0.58,
            voice_burst_stride_samples=4320,
        ),
    )
    p25_plugin.decode(
        np.zeros(3),
        P25Config(
            samples_per_symbol=8,
            sync_threshold=0.7,
            sync_min_distance_symbols=90,
            stable_nac_min_count=3,
            stable_nac_min_ratio=0.25,
        ),
    )
    dpmr_plugin.decode(
        np.zeros(3),
        DPMRConfig(
            sync_threshold=0.9,
            sps_search_min=18.5,
            sps_search_max=21.5,
            voice_symbol_candidate_limit=24,
        ),
    )

    assert calls == [
        ("dmr", 0.71, 0.58, 4320),
        ("p25", 8, 0.7, 90, 3, 0.25),
        ("dpmr", 0.9, 18.5, 21.5, 24),
    ]


def test_protocol_frontend_wrappers_pass_config(monkeypatch):
    calls = []

    def fake_c4fm_frontend(iq, fo, fs, cutoff, ntaps, dev_nominal, min_samples, psd_nperseg):
        calls.append((fs, cutoff, ntaps, dev_nominal, min_samples, psd_nperseg))
        return np.zeros(3)

    def fake_dpmr_frontend(iq, fs, cutoff, ntaps, dev_nominal, min_samples, psd_nperseg):
        calls.append((fs, cutoff, ntaps, dev_nominal, min_samples, psd_nperseg))
        return np.ones(3)

    monkeypatch.setattr(dmr_plugin, "_frontend_c4fm", fake_c4fm_frontend)
    monkeypatch.setattr(p25_plugin, "fsk_frontend", fake_c4fm_frontend)
    monkeypatch.setattr(dpmr_plugin, "frontend_dpmr", fake_dpmr_frontend)

    dmr_plugin.frontend(
        np.zeros(8),
        12_000.0,
        DMRConfig(
            frontend_cutoff_hz=8_000.0,
            frontend_taps=101,
            nominal_deviation_hz=1_800.0,
            frontend_min_samples=256,
            frontend_psd_nperseg=2048,
        ),
    )
    p25_plugin.frontend(
        np.zeros(8),
        24_000.0,
        P25Config(
            frontend_cutoff_hz=7_000.0,
            frontend_taps=99,
            nominal_deviation_hz=1_700.0,
            frontend_min_samples=300,
            frontend_psd_nperseg=1024,
        ),
    )
    dpmr_plugin.frontend(
        np.zeros(8),
        48_000.0,
        DPMRConfig(
            frontend_cutoff_hz=3_000.0,
            frontend_taps=77,
            nominal_deviation_hz=1_600.0,
            frontend_min_samples=384,
            frontend_psd_nperseg=512,
        ),
    )

    assert calls == [
        (12_000.0, 8_000.0, 101, 1_800.0, 256, 2048),
        (24_000.0, 7_000.0, 99, 1_700.0, 300, 1024),
        (48_000.0, 3_000.0, 77, 1_600.0, 384, 512),
    ]


def test_print_results_accepts_p25_nid(capsys):
    radio_output.print_results([
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
    radio_output.print_results([
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
    radio_output.print_results([
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
