import numpy as np
import pytest
from realtime.wideband_source import (
    WidebandIQSource, FileWidebandSource, SoapyWidebandSource,
)


def _make_iq_file(tmp_path, n):
    path = str(tmp_path / "wb.rawiq")
    data = np.empty(2 * n, dtype=np.int16)
    data[0::2] = np.arange(n, dtype=np.int16)
    data[1::2] = -np.arange(n, dtype=np.int16)
    data.tofile(path)
    return path


def test_reads_all_samples(tmp_path):
    path = _make_iq_file(tmp_path, 5000)
    src = FileWidebandSource(path, sample_rate=5e6, chunk_samples=1024, throttle=False)
    total = 0
    while True:
        c = src.read_chunk()
        if c is None:
            break
        assert c.dtype == np.complex64
        total += len(c)
    src.close()
    assert total == 5000


def test_chunk_size_and_center(tmp_path):
    path = _make_iq_file(tmp_path, 5000)
    src = FileWidebandSource(path, sample_rate=5e6, center_hz=435e6,
                             chunk_samples=1024, throttle=False)
    assert src.center_hz == 435e6
    assert len(src.read_chunk()) == 1024
    src.close()


def test_soapy_placeholder_raises():
    with pytest.raises(NotImplementedError):
        SoapyWidebandSource(sample_rate=60e6)
