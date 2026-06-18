# tests/test_aggregator.py
import pytest
from realtime.aggregator import SessionAggregator, CallRecord, CALL_TIMEOUT_WINDOWS


def _pdu(type_, src=1, dst=2, fo=150000.0, wid=0, flco="GroupVoiceChannelUser"):
    return {"type": type_, "src": src, "dst": dst, "flco": flco,
            "ts": 0, "extra": {}, "raw_bits": b"\x00" * 33,
            "_fo_hz": fo, "_window_id": wid}


def test_lc_header_opens_session():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    calls = agg.active_calls()
    assert len(calls) == 1
    assert calls[0].src == 1 and calls[0].dst == 2
    assert calls[0].start_window == 0


def test_duplicate_signalling_deduped():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    agg.feed(_pdu("LATE_ENTRY", wid=1))
    agg.feed(_pdu("LATE_ENTRY", wid=2))
    # Still one active call (same fo,src,dst)
    assert len(agg.active_calls()) == 1


def test_terminator_closes_session():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    agg.feed(_pdu("TERMINATOR", wid=5))
    closed = agg.expire(current_window=5, closed_fos=[])
    assert len(closed) == 1
    assert closed[0].closed_by == "terminator"
    assert closed[0].end_window == 5
    assert agg.active_calls() == []


def test_closing_fo_closes_session():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    closed = agg.expire(current_window=4, closed_fos=[150000.0])
    assert len(closed) == 1
    assert closed[0].closed_by == "detector"


def test_timeout_closes_session():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    # No further PDUs; current window advances past timeout
    closed = agg.expire(current_window=CALL_TIMEOUT_WINDOWS + 1, closed_fos=[])
    assert len(closed) == 1
    assert closed[0].closed_by == "timeout"


def test_voice_frames_accumulate_not_dedup():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", wid=0))
    agg.feed(_pdu("LATE_ENTRY", wid=1))
    agg.feed(_pdu("LATE_ENTRY", wid=2))
    call = agg.active_calls()[0]
    # raw_bits from each LATE_ENTRY appended to voice_raw (not deduped)
    assert len(call.voice_raw) == 2


def test_concurrent_calls_tracked_separately():
    agg = SessionAggregator()
    agg.feed(_pdu("LC_HEADER", src=1, dst=2, fo=-300000.0, wid=0))
    agg.feed(_pdu("LC_HEADER", src=3, dst=4, fo=150000.0, wid=1))
    assert len(agg.active_calls()) == 2


def test_duplicate_terminator_in_overlap_no_phantom():
    """A TERMINATOR re-decoded in an overlapping window must not create a phantom call.

    Scenario: window_sec=1.0, step_sec=0.9 -> 0.1s overlap. A TERMINATOR burst
    appearing at the boundary gets decoded in windows 5 and 6. The first closes
    the call legitimately; the second must be silently ignored, not spawn a new
    empty CallRecord.
    """
    agg = SessionAggregator()

    # Open a call in window 0
    agg.feed(_pdu("LC_HEADER", wid=0))
    assert len(agg.active_calls()) == 1

    # First TERMINATOR: closes the call legitimately
    agg.feed(_pdu("TERMINATOR", wid=5))
    assert agg.active_calls() == [], "call should be closed after first TERMINATOR"

    # Second TERMINATOR for same (fo, src, dst) — the overlap re-decode
    agg.feed(_pdu("TERMINATOR", wid=6))

    # Must still be no active calls (no phantom opened)
    assert agg.active_calls() == [], "second TERMINATOR must not create a phantom call"

    # Collect everything that expire() drains
    closed = agg.expire(current_window=6, closed_fos=[])

    # Only one real closed record (from the first TERMINATOR in _pending_closed)
    assert len(closed) == 1, f"expected exactly 1 closed record, got {len(closed)}"
    assert closed[0].closed_by == "terminator"
    assert closed[0].end_window == 5
    # The legitimate record was opened at window 0 and has the right src/dst
    assert closed[0].src == 1 and closed[0].dst == 2
    assert closed[0].start_window == 0


def test_absolute_rf_key_merges_by_rf_not_subband_offset():
    from realtime.aggregator import SessionAggregator
    agg = SessionAggregator()
    # same call seen in two adjacent sub-bands (straddling): different sub-band
    # offsets but SAME absolute RF -> must merge into ONE call.
    pdu_a = {"type": "LC_HEADER", "src": 1, "dst": 2, "flco": "GroupVoiceChannelUser",
             "ts": 0, "extra": {}, "raw_bits": b"\x00" * 33,
             "_fo_hz": +200000.0, "_rf_hz": 435_000_000.0, "_window_id": 0}
    pdu_b = {"type": "LATE_ENTRY", "src": 1, "dst": 2, "flco": "GroupVoiceChannelUser",
             "ts": 0, "extra": {}, "raw_bits": b"\x00" * 33,
             "_fo_hz": -180000.0, "_rf_hz": 435_001_000.0, "_window_id": 1}
    agg.feed(pdu_a)
    agg.feed(pdu_b)
    calls = agg.active_calls()
    assert len(calls) == 1
    assert abs(calls[0].fo_hz - 435_000_000.0) < 5000.0
