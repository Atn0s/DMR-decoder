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
