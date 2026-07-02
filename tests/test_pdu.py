from radio.pdu import PDU, normalize_pdu, pdu_get, pdu_to_dict, set_pdu_meta


def test_pdu_from_dict_moves_legacy_metadata_to_meta_and_preserves_output():
    pdu = PDU.from_dict({
        "protocol": "DMR",
        "type": "LC_HEADER",
        "src": 1,
        "dst": 2,
        "ts": 0,
        "flco": "GroupVoiceChannelUser",
        "fid": "FID",
        "extra": {"color_code": 1},
        "raw_bits": b"abc",
        "_fo_hz": 1250.0,
        "_window_id": 7,
        "custom": "kept",
    })

    assert pdu.protocol == "DMR"
    assert pdu.type == "LC_HEADER"
    assert pdu.meta == {"fo_hz": 1250.0, "window_id": 7}
    assert pdu.fields == {"custom": "kept"}
    assert pdu.get("_fo_hz") == 1250.0
    assert pdu["_window_id"] == 7

    out = pdu.to_dict()
    assert out["raw_bits"] == b"abc"
    assert out["_fo_hz"] == 1250.0
    assert out["_window_id"] == 7
    assert out["custom"] == "kept"
    assert "meta" not in out


def test_pdu_to_dict_can_omit_raw_bits_and_include_meta():
    pdu = PDU.from_dict({
        "protocol": "P25",
        "type": "P25_NID",
        "raw_bits": b"abc",
        "_fo_hz": 500.0,
    })

    out = pdu.to_dict(include_raw_bits=False, include_meta=True)

    assert "raw_bits" not in out
    assert out["_fo_hz"] == 500.0
    assert out["meta"] == {"fo_hz": 500.0}


def test_pdu_helpers_accept_dict_and_pdu():
    raw = {"protocol": "dPMR", "type": "DPMR_VOICE", "_rf_hz": 435_000_000.0}
    pdu = normalize_pdu(raw)

    assert normalize_pdu(pdu) is pdu
    assert pdu_get(raw, "_rf_hz") == 435_000_000.0
    assert pdu_get(pdu, "_rf_hz") == 435_000_000.0
    assert pdu_to_dict(raw) == raw
    assert pdu_to_dict(pdu)["_rf_hz"] == 435_000_000.0


def test_set_pdu_meta_writes_legacy_dict_key_or_pdu_meta():
    raw = {}
    set_pdu_meta(raw, "fo_hz", 1250.0)
    assert raw == {"_fo_hz": 1250.0}

    pdu = PDU(type="LC_HEADER")
    set_pdu_meta(pdu, "_window_id", 3)
    assert pdu.meta == {"window_id": 3}
    assert pdu.to_dict()["_window_id"] == 3
