# tests/test_wideband_e2e.py
import os
import numpy as np
import pytest
from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner


def test_wideband_two_channels_different_subbands(tmp_path):
    """End-to-end: two DMR signals far apart in a 5MHz band, beyond a single
    2.5MHz sub-band's reach -> only channelization can catch both."""
    from utils.synthesis import synthesize_wideband_grid
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband files not present")
    out = str(tmp_path / "wb_e2e.rawiq")
    synthesize_wideband_grid(
        [(-1_800_000.0, "dmr_1_78125.rawiq"), (+1_800_000.0, "dmr_2_78125.rawiq")],
        out, fs_out=5e6, dur_sec=10.0, data_dir="data")
    src = FileWidebandSource(out, sample_rate=5e6, center_hz=435e6,
                             chunk_samples=5_000_000, throttle=False)
    scanner = WidebandScanner(src, num_subbands=4, taps_per_phase=12, oversample=2,
                              window_sec=1.0, step_sec=0.9)
    calls = scanner.run()
    assert isinstance(calls, list)
    # Both signals must be decoded: one below 435 MHz (~433.2 MHz) and one above (~436.8 MHz)
    voice = [c for c in calls if c.flco == "GroupVoiceChannelUser"]
    assert any(c.fo_hz < 435e6 for c in voice), \
        f"No voice call below 435 MHz; got RFs: {[c.fo_hz for c in voice]}"
    assert any(c.fo_hz > 435e6 for c in voice), \
        f"No voice call above 435 MHz; got RFs: {[c.fo_hz for c in voice]}"
    # No phantom calls: every voice call must be within 0.3 MHz of one of the two
    # real signal RFs (433.2 MHz and 436.8 MHz).  The owning-sub-band guard
    # (nearest-center rule) eliminates alias detections; this assertion pins that
    # they do not come back.
    real_rfs = [433.2e6, 436.8e6]
    for c in voice:
        assert any(abs(c.fo_hz - rf) <= 0.3e6 for rf in real_rfs), \
            f"Spurious phantom call at {c.fo_hz/1e6:.3f} MHz (not within 0.3 MHz of 433.2 or 436.8); " \
            f"all RFs: {[round(x.fo_hz/1e6, 3) for x in voice]}"


def test_wideband_cli_runs(tmp_path, capsys):
    import os
    from utils.synthesis import synthesize_wideband_grid
    from realtime.scanner_rt import run_wideband_cli
    if not os.path.exists("data/dmr_1_78125.rawiq"):
        pytest.skip("source narrowband file not present")
    out = str(tmp_path / "cli_wb.rawiq")
    synthesize_wideband_grid([(-1_800_000.0, "dmr_1_78125.rawiq")],
                             out, fs_out=5e6, dur_sec=10.0, data_dir="data")

    class Args:
        path = out
        fs = 5e6
        center = 435e6
        nsub = 4
        oversample = 2
    calls = run_wideband_cli(Args())
    assert isinstance(calls, list)


def test_wideband_returns_list_on_noise(tmp_path):
    """Pure-noise wideband file: pipeline runs clean, returns a list (no crash)."""
    path = str(tmp_path / "noise.rawiq")
    n = 2_000_000
    rng = np.random.default_rng(0)
    data = np.empty(2 * n, dtype=np.int16)
    data[0::2] = (rng.standard_normal(n) * 200).astype(np.int16)
    data[1::2] = (rng.standard_normal(n) * 200).astype(np.int16)
    data.tofile(path)
    src = FileWidebandSource(path, sample_rate=5e6, center_hz=435e6,
                             chunk_samples=1_000_000, throttle=False)
    scanner = WidebandScanner(src, num_subbands=4, oversample=2)
    calls = scanner.run(max_windows=3)
    assert isinstance(calls, list)
