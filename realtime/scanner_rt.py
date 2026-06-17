import threading
import numpy as np
from multiprocessing import Pool

from realtime.ring_buffer import RingBuffer
from realtime.detector import Detector
from realtime.aggregator import SessionAggregator, CallRecord
from realtime.worker import decode_window


class RealtimeScanner:
    """Orchestrates acquisition thread + detector + worker pool + aggregator.
    core/ and scanner.py decode logic are reused unchanged via decode_window."""

    def __init__(self, source, num_workers: int = 4, window_sec: float = 1.0,
                 step_sec: float = 0.9, ring_capacity_sec: float = 3.0,
                 use_pool: bool = True):
        self.source = source
        self.num_workers = num_workers
        self.fs = source.sample_rate
        self.window_samples = int(window_sec * self.fs)
        self.step_samples = int(step_sec * self.fs)
        self.ring = RingBuffer(int(ring_capacity_sec * self.fs))
        self.detector = Detector(sample_rate=self.fs)
        self.aggregator = SessionAggregator()
        self.use_pool = use_pool
        self._acq_done = threading.Event()

    def _acquire(self):
        while True:
            chunk = self.source.read_chunk()
            if chunk is None:
                break
            dropped = self.ring.write(chunk)
            if dropped > 0:
                print(f"[WARN] ring overflow: dropped {dropped} samples "
                      f"(total {self.ring.overflow_count})")
        self._acq_done.set()

    def _dispatch(self, tasks, pool):
        if not tasks:
            return []
        if self.use_pool and pool is not None:
            args = [(iq, fo, wid, self.fs) for (iq, fo, wid) in tasks]
            return pool.starmap(decode_window, args)
        return [decode_window(iq, fo, wid, self.fs) for (iq, fo, wid) in tasks]

    def run(self, on_call=None, max_windows: int | None = None) -> list[CallRecord]:
        acq = threading.Thread(target=self._acquire, daemon=True)
        acq.start()

        all_closed: list[CallRecord] = []
        window_id = 0
        pool = Pool(self.num_workers) if self.use_pool else None
        try:
            while True:
                win = self.ring.read_window(self.window_samples, self.step_samples)
                if win is None:
                    if self._acq_done.is_set() and self.ring.available() < self.window_samples:
                        break
                    self._acq_done.wait(timeout=0.05)
                    continue

                tasks = self.detector.process_window(win, window_id)
                results = self._dispatch(tasks, pool)
                for pdu_list in results:
                    for pdu in pdu_list:
                        self.aggregator.feed(pdu)

                closed = self.aggregator.expire(window_id, self.detector.closed_channels())
                for rec in closed:
                    all_closed.append(rec)
                    if on_call:
                        on_call(rec)

                window_id += 1
                if max_windows is not None and window_id >= max_windows:
                    break
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # Flush remaining active calls as timeout-closed
        final = self.aggregator.expire(window_id + self.aggregator.timeout_windows, [])
        for rec in final:
            all_closed.append(rec)
            if on_call:
                on_call(rec)
        self.source.close()
        return all_closed
