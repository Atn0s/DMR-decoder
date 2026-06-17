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
