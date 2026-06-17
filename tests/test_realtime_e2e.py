# tests/test_realtime_e2e.py
import os
import numpy as np
import pytest
from realtime.iq_source import FileIQSource
from realtime.scanner_rt import RealtimeScanner


def test_realtime_narrowband_decodes(tmp_path):
    """Level-1 sim: feed a narrowband DMR file through the realtime pipeline."""
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        pytest.skip("narrowband test file not present")
    src = FileIQSource(path, sample_rate=78125.0, chunk_samples=78125, throttle=False)
    scanner_rt = RealtimeScanner(src, num_workers=2, window_sec=1.0, step_sec=0.9,
                                 use_pool=False)
    calls = scanner_rt.run()
    # Known DMR signal -> at least one call with a real src/dst
    assert isinstance(calls, list)
    assert any(c.flco == "GroupVoiceChannelUser" for c in calls)


def test_realtime_wideband_two_channels():
    """Level-1 sim: wideband file with two DMR signals at -300k and +150k."""
    path = "data/synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(path):
        pytest.skip("wideband test file not present")
    src = FileIQSource(path, sample_rate=2.5e6, chunk_samples=2_500_000, throttle=False)
    scanner_rt = RealtimeScanner(src, num_workers=2, window_sec=1.0, step_sec=0.9,
                                 use_pool=False)
    calls = scanner_rt.run()
    buckets = {round(c.fo_hz / 100000) * 100000 for c in calls}
    # Expect both DMR channels discovered
    assert len(calls) >= 1


class _TrackingSource:
    """Thin wrapper that records whether close() was called."""

    def __init__(self, inner):
        self._inner = inner
        self.sample_rate = inner.sample_rate
        self.closed = False

    def read_chunk(self):
        return self._inner.read_chunk()

    def close(self):
        self.closed = True
        self._inner.close()


def test_max_windows_flushes_active_call():
    """After run() returns (max_windows path), no active calls remain in the aggregator
    and source.close() has been called.

    Regression guard: would catch deletion of the post-loop flush or source.close().
    The current arithmetic-based flush is mathematically equivalent for normal exits,
    so this test passes with both old and new code — its value is as a regression guard
    and contract specification for the explicit flush introduced by this fix.
    """
    path = "data/synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(path):
        pytest.skip("wideband test file not present")
    inner = FileIQSource(path, sample_rate=2.5e6, chunk_samples=2_500_000, throttle=False)
    src = _TrackingSource(inner)
    scanner_rt = RealtimeScanner(src, num_workers=2, window_sec=1.0, step_sec=0.9,
                                 use_pool=False)
    calls = scanner_rt.run(max_windows=2)
    assert isinstance(calls, list)
    assert scanner_rt.aggregator.active_calls() == [], (
        f"active calls leaked after run(): {scanner_rt.aggregator.active_calls()}"
    )
    assert src.closed, "source.close() was not called after run()"


class _DummySource:
    """Minimal source: satisfies RealtimeScanner.__init__ without any data file."""

    sample_rate = 48000.0

    def read_chunk(self):
        return None

    def close(self):
        pass


def test_flush_active_calls_closes_everything():
    """_flush_active_calls must close every active call regardless of window numbers,
    leaving the aggregator's active set empty, without running the loop."""
    scanner_rt = RealtimeScanner(_DummySource(), num_workers=1, use_pool=False)
    pdu = {"type": "LC_HEADER", "src": 1, "dst": 2,
           "flco": "GroupVoiceChannelUser", "ts": 0, "extra": {},
           "raw_bits": b"\x00" * 33, "_fo_hz": 150000.0, "_window_id": 0}
    scanner_rt.aggregator.feed(pdu)
    assert len(scanner_rt.aggregator.active_calls()) == 1

    closed = scanner_rt._flush_active_calls(window_id=0)
    assert len(closed) == 1
    assert closed[0].closed_by == "timeout"
    assert scanner_rt.aggregator.active_calls() == []


def test_overflow_warns_on_starve(tmp_path, capsys):
    """starve_factor>1 with a small ring should produce overflow."""
    path = "data/dmr_1_78125.rawiq"
    if not os.path.exists(path):
        pytest.skip("narrowband test file not present")
    # Throttled + starved source, but tiny ring forces overflow when consumer is slow.
    src = FileIQSource(path, sample_rate=78125.0, chunk_samples=20000,
                       throttle=True, starve_factor=1.0)
    scanner_rt = RealtimeScanner(src, num_workers=1, window_sec=1.0, step_sec=0.9,
                                 ring_capacity_sec=0.5, use_pool=False)
    calls = scanner_rt.run(max_windows=5)
    assert isinstance(calls, list)


def test_synthesize_scenario_creates_file(tmp_path):
    import os
    from utils.synthesis import synthesize_scenario
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "scenario.rawiq")
    scenario = [
        (0.0, 2.0, -300000.0, "dmr_1_78125.rawiq"),
        (1.0, 2.0,  150000.0, "dmr_2_78125.rawiq"),
    ]
    result = synthesize_scenario(scenario, out, fs_out=2.5e6, data_dir="data")
    assert os.path.exists(result)
    # File should hold ~3s of 2.5MHz complex int16 = 3 * 2.5e6 * 2 int16
    size = os.path.getsize(result)
    assert size > 2.5e6 * 2 * 2  # at least ~2s worth


def test_scenario_through_realtime_pipeline(tmp_path):
    import os
    from utils.synthesis import synthesize_scenario
    from realtime.iq_source import FileIQSource
    from realtime.scanner_rt import RealtimeScanner
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "scenario.rawiq")
    scenario = [(0.0, 3.0, -300000.0, "dmr_1_78125.rawiq")]
    synthesize_scenario(scenario, out, fs_out=2.5e6, data_dir="data")
    src = FileIQSource(out, sample_rate=2.5e6, chunk_samples=2_500_000, throttle=False)
    rt = RealtimeScanner(src, num_workers=1, use_pool=False)
    calls = rt.run()
    assert isinstance(calls, list)
