import time
import numpy as np


class IQSource:
    """Abstract continuous IQ source. read_chunk returns complex64 blocks."""
    sample_rate: float

    def read_chunk(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FileIQSource(IQSource):
    """Read a .rawiq file (interleaved int16) in chunks, optionally throttled to
    sample_rate to emulate live SDR pacing.

    throttle=False: read at full speed (fast regression tests).
    starve_factor>1.0: sleep longer than real-time to reproduce sample drops
    downstream (the source itself never drops; it just emits slower)."""

    def __init__(self, path: str, sample_rate: float, chunk_samples: int = 65536,
                 throttle: bool = True, starve_factor: float = 1.0):
        self.path = path
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.throttle = throttle
        self.starve_factor = starve_factor
        self._fh = open(path, "rb")

    def read_chunk(self) -> np.ndarray | None:
        # Each complex sample = 2 int16 = 4 bytes
        raw = np.frombuffer(self._fh.read(self.chunk_samples * 4), dtype=np.int16)
        if len(raw) < 2:
            return None
        n = len(raw) // 2
        iq = (raw[0:2 * n:2].astype(np.float32) +
              1j * raw[1:2 * n:2].astype(np.float32)) / 32768.0
        iq = iq.astype(np.complex64)
        if self.throttle:
            time.sleep((n / self.sample_rate) * self.starve_factor)
        return iq

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()


class SoapyIQSource(IQSource):
    """Real SDR via SoapySDR. Placeholder — not implemented this phase."""

    def __init__(self, sample_rate: float, **kwargs):
        raise NotImplementedError("SoapyIQSource is a placeholder for hardware phase")
