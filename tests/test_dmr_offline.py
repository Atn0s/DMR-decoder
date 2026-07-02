import numpy as np

import dmr.offline as dmr_offline
from radio import pipeline


def test_dmr_offline_scan_file_delegates_to_unified_pipeline(monkeypatch):
    iq = np.ones(4, dtype=complex)
    pdus = [{"protocol": "DMR", "type": "LC_HEADER"}]
    calls = []

    monkeypatch.setattr(dmr_offline, "read_rawiq", lambda path: iq)
    monkeypatch.setattr(dmr_offline, "detect_sample_rate", lambda path: 78_125)

    def fake_scan_iq(iq_arg, sample_rate, freq_list, protocol_names):
        calls.append((iq_arg is iq, sample_rate, freq_list, protocol_names))
        return pdus

    monkeypatch.setattr(pipeline, "scan_iq", fake_scan_iq)

    result = dmr_offline.scan_file("sample.rawiq", freq_list=[1250.0])

    assert result is pdus
    assert calls == [(True, 78_125, [1250.0], {"dmr"})]
