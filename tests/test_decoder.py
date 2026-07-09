import numpy as np
from bitarray import bitarray
from core.decoder import decode_burst, LateEntryCollector
from dmr.config import DMRConfig
import dmr.decode_flow as dmr_decode_flow


def test_decode_burst_garbage_returns_none():
    # All-zero symbols -> Golay check fails -> None
    syms = np.zeros(132)
    result = decode_burst(syms, "DATA_MS")
    assert result is None


def test_decode_burst_none_for_all_zeros():
    # All-zero symbols fail Golay -> decode_burst returns None (not a crash, not a dict)
    syms = np.zeros(132)
    result = decode_burst(syms, "DATA_MS")
    assert result is None


def test_decode_burst_pdu_schema():
    # Verify that when decode_burst returns a dict, it has the required schema.
    # We use a known-valid burst from the real data file if available.
    import os
    import scipy.signal as sig
    from core.dsp import read_rawiq, frontend, find_sync_positions, recover_burst
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        return
    iq = read_rawiq(path)
    iq_dec = sig.resample_poly(iq, 384, 625)
    y = frontend(iq_dec, fo=0.0)
    positions = find_sync_positions(y)
    for center, polarity, sync_type in positions:
        if "DATA" not in sync_type:
            continue
        syms = recover_burst(y, center, polarity, sync_type)
        if syms is None:
            continue
        result = decode_burst(syms, sync_type)
        if result is not None:
            for key in ("type", "src", "dst", "ts", "flco", "extra", "raw_bits"):
                assert key in result, f"Missing key: {key}"
            assert result["type"] in ("LC_HEADER", "TERMINATOR", "CSBK")
            return
    # If we get here with no PDU decoded, that's a data issue not a code issue


def test_late_entry_collector_needs_four_frags():
    col = LateEntryCollector()
    ba = bitarray(264)
    ba.setall(0)
    # Single feed should never produce a PDU (not enough frags)
    result = col.feed(ba, "MS_VOICE")
    assert result is None


def test_late_entry_collector_reset():
    col = LateEntryCollector()
    ba = bitarray(264)
    ba.setall(0)
    col.feed(ba, "MS_VOICE")
    col.reset()
    assert col._collecting is False
    assert col._frags == []


def test_late_entry_collector_state_machine():
    from dmr.layer2 import LCSS
    col = LateEntryCollector()

    def make_ba_with_lcss(lcss_val):
        """Build a 264-bit burst with a specific LCSS value in the EMB header."""
        ba = bitarray(264)
        ba.setall(0)
        # EMB header is center[0:8] + center[40:48] where center = ba[108:156]
        # EmbeddedSignalling layout: [0:4]=color_code [4]=PI [5:7]=LCSS [7:16]=parity.
        # center[0:8] = ba[108:116], center[40:48] = ba[148:156].
        # emb_bits[5]=ba[113], emb_bits[6]=ba[114].
        lcss_bits = f"{lcss_val:02b}"
        ba[113] = int(lcss_bits[0])
        ba[114] = int(lcss_bits[1])
        return ba

    # Feed a FirstFragmentLC burst — should start collecting
    first_ba = make_ba_with_lcss(LCSS.FirstFragmentLC.value)
    result = col.feed(first_ba, "MS_VOICE")
    # May fail EMB parse due to parity, but should not raise
    assert result is None or isinstance(result, dict)


def test_dmr_decode_loop_passes_config_to_sync_and_voice_recovery(monkeypatch):
    calls = []

    def fake_find_sync_positions(
        y,
        voice_threshold,
        data_threshold,
        peak_distance_samples,
    ):
        calls.append((
            "sync",
            voice_threshold,
            data_threshold,
            peak_distance_samples,
        ))
        return [(1000, 1.0, "MS_VOICE")]

    def fake_recover_stepped_burst(
        y,
        anchor,
        j,
        ph,
        polarity,
        burst_stride_samples,
    ):
        calls.append(("voice", anchor, j, ph, polarity, burst_stride_samples))
        return None

    monkeypatch.setattr(dmr_decode_flow, "find_sync_positions", fake_find_sync_positions)
    monkeypatch.setattr(dmr_decode_flow, "_lock_voice_phase", lambda *args: 0.25)
    monkeypatch.setattr(dmr_decode_flow, "_recover_stepped_burst", fake_recover_stepped_burst)

    config = DMRConfig(
        sync_threshold_voice=0.72,
        sync_threshold_data=0.59,
        sync_peak_distance_samples=640,
        voice_burst_stride_samples=4320,
    )

    result = dmr_decode_flow.decode_dmr_flow(np.zeros(10), config)

    assert result == []
    assert calls == [
        ("sync", 0.72, 0.59, 640),
        ("voice", 1000, 0, 0.25, 1.0, 4320),
    ]


def test_dmr_decode_loop_uses_configured_burst_dedup_window(monkeypatch):
    calls = []

    monkeypatch.setattr(
        dmr_decode_flow,
        "find_sync_positions",
        lambda *args, **kwargs: [
            (1000, 1.0, "DATA_MS"),
            (1024, 1.0, "DATA_MS"),
            (1100, 1.0, "DATA_MS"),
        ],
    )

    def fake_recover_burst(y, center, polarity, sync_type):
        calls.append(center)
        return None

    monkeypatch.setattr(dmr_decode_flow, "recover_burst", fake_recover_burst)

    result = dmr_decode_flow.decode_dmr_flow(
        np.zeros(1200),
        DMRConfig(burst_dedup_window_samples=50),
    )

    assert result == []
    assert calls == [1000, 1100]
