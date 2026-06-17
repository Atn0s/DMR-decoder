# tests/test_iq_source.py
import os
import time
import numpy as np
import pytest
from realtime.iq_source import IQSource, FileIQSource, SoapyIQSource


def _make_iq_file(tmp_path, n_samples):
    """Write n_samples complex samples as interleaved int16 .rawiq, return path."""
    path = str(tmp_path / "test.rawiq")
    data = np.empty(2 * n_samples, dtype=np.int16)
    data[0::2] = np.arange(n_samples, dtype=np.int16)        # I ramp
    data[1::2] = -np.arange(n_samples, dtype=np.int16)       # Q ramp
    data.tofile(path)
    return path


def test_file_source_reads_all_samples(tmp_path):
    path = _make_iq_file(tmp_path, 1000)
    src = FileIQSource(path, sample_rate=48000.0, chunk_samples=256, throttle=False)
    total = 0
    while True:
        chunk = src.read_chunk()
        if chunk is None:
            break
        assert chunk.dtype == np.complex64
        total += len(chunk)
    src.close()
    assert total == 1000


def test_file_source_chunk_size(tmp_path):
    path = _make_iq_file(tmp_path, 1000)
    src = FileIQSource(path, sample_rate=48000.0, chunk_samples=256, throttle=False)
    first = src.read_chunk()
    assert len(first) == 256
    src.close()


def test_file_source_throttle_pacing(tmp_path):
    # 4 chunks of 12000 samples at 48000 Hz = 0.25s each = 1.0s total
    path = _make_iq_file(tmp_path, 48000)
    src = FileIQSource(path, sample_rate=48000.0, chunk_samples=12000, throttle=True)
    t0 = time.perf_counter()
    while src.read_chunk() is not None:
        pass
    elapsed = time.perf_counter() - t0
    src.close()
    # Should take ~1.0s; allow generous lower bound to avoid flakiness
    assert elapsed >= 0.7


def test_starve_factor_slows_pacing(tmp_path):
    path = _make_iq_file(tmp_path, 24000)
    src = FileIQSource(path, sample_rate=48000.0, chunk_samples=12000,
                       throttle=True, starve_factor=2.0)
    t0 = time.perf_counter()
    while src.read_chunk() is not None:
        pass
    elapsed = time.perf_counter() - t0
    src.close()
    # 0.5s of data at 2x starve = ~1.0s
    assert elapsed >= 0.7


def test_soapy_source_not_implemented():
    with pytest.raises(NotImplementedError):
        SoapyIQSource(sample_rate=2.4e6)
