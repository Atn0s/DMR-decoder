import threading
import numpy as np


class RingBuffer:
    """Single-producer single-consumer ring buffer for complex64 samples.
    Thread-safe. On overflow, oldest data is dropped (mirrors SDR driver behavior)
    and overflow_count is incremented."""

    def __init__(self, capacity_samples: int):
        self._cap = capacity_samples
        self._buf = np.zeros(capacity_samples, dtype=np.complex64)
        self._write_pos = 0          # absolute count of samples written
        self._read_pos = 0           # absolute count of samples consumed
        self._overflow = 0
        self._lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> int:
        chunk = chunk.astype(np.complex64, copy=False)
        n = len(chunk)
        dropped = 0
        with self._lock:
            # If incoming exceeds capacity, keep only the newest cap samples
            if n >= self._cap:
                chunk = chunk[-self._cap:]
                dropped += n - self._cap
                n = self._cap
            # Make room: if unread + new exceeds capacity, advance read_pos
            unread = self._write_pos - self._read_pos
            free = self._cap - unread
            if n > free:
                evict = n - free
                self._read_pos += evict
                dropped += evict
            start = self._write_pos % self._cap
            end = start + n
            if end <= self._cap:
                self._buf[start:end] = chunk
            else:
                first = self._cap - start
                self._buf[start:] = chunk[:first]
                self._buf[:n - first] = chunk[first:]
            self._write_pos += n
            self._overflow += dropped
        return dropped

    def read_window(self, window_samples: int, step_samples: int) -> np.ndarray | None:
        with self._lock:
            unread = self._write_pos - self._read_pos
            if unread < window_samples:
                return None
            start = self._read_pos % self._cap
            end = start + window_samples
            if end <= self._cap:
                out = self._buf[start:end].copy()
            else:
                first = self._cap - start
                out = np.empty(window_samples, dtype=np.complex64)
                out[:first] = self._buf[start:]
                out[first:] = self._buf[:window_samples - first]
            self._read_pos += step_samples
            return out

    def available(self) -> int:
        with self._lock:
            return self._write_pos - self._read_pos

    @property
    def overflow_count(self) -> int:
        return self._overflow
