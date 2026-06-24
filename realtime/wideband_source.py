import time
import numpy as np


class WidebandIQSource:
    """Abstract wideband (60-70MHz-class) IQ source. One-shot capture, no tune."""
    sample_rate: float
    center_hz: float

    def read_chunk(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FileWidebandSource(WidebandIQSource):
    """Read a wideband IQ file (interleaved int16) in chunks.
    throttle=False reads at full speed (offline correctness, default).
    center_hz is the absolute RF center of the captured band (metadata for
    absolute-frequency labeling downstream); it does not alter the samples.
    header_bytes skips a fixed-size file header before the int16 IQ payload
    (e.g. 112 for USRP/BVSP captures); 0 for a headerless .rawiq."""

    def __init__(self, path: str, sample_rate: float, center_hz: float = 0.0,
                 chunk_samples: int = 2_000_000, throttle: bool = False,
                 header_bytes: int = 0):
        self.path = path
        self.sample_rate = float(sample_rate)
        self.center_hz = float(center_hz)
        self.chunk_samples = int(chunk_samples)
        self.throttle = throttle
        self.header_bytes = int(header_bytes)
        self._fh = open(path, "rb")
        if self.header_bytes:
            self._fh.seek(self.header_bytes)

    def read_chunk(self) -> np.ndarray | None:
        raw = np.frombuffer(self._fh.read(self.chunk_samples * 4), dtype=np.int16)
        if len(raw) < 2:
            return None
        n = len(raw) // 2
        iq = (raw[0:2 * n:2].astype(np.float32) +
              1j * raw[1:2 * n:2].astype(np.float32)) / 32768.0
        iq = iq.astype(np.complex64)
        if self.throttle:
            time.sleep(n / self.sample_rate)
        return iq

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()


class SoapyWidebandSource(WidebandIQSource):
    """Real wideband SDR (SoapySDR). Placeholder — not implemented this phase."""

    def __init__(self, sample_rate: float, **kwargs):
        raise NotImplementedError("SoapyWidebandSource is a placeholder for hardware phase")
