import numpy as np

from realtime.channelizer import PolyphaseChannelizer
from realtime.detector import Detector
from realtime.aggregator import SessionAggregator, CallRecord
from realtime.worker import decode_window


class WidebandScanner:
    """Two-stage wideband scanner: channelize one-shot wideband capture into N
    overlapping sub-bands, then run the existing per-band decode pipeline on each
    sub-band, feeding a SHARED aggregator keyed on absolute RF frequency.

    Offline correctness path: the whole capture is read and channelized, then each
    sub-band is windowed and decoded. Decode core (scanner._decode_loop via
    worker.decode_window) is reused unchanged."""

    def __init__(self, source, num_subbands: int = 32, taps_per_phase: int = 12,
                 oversample: int = 2, window_sec: float = 1.0, step_sec: float = 0.9,
                 energy_floor_db: float = 2.0):
        self.source = source
        self.fs = source.sample_rate
        self.center_hz = getattr(source, "center_hz", 0.0)
        self.channelizer = PolyphaseChannelizer(
            self.fs, num_subbands=num_subbands, taps_per_phase=taps_per_phase,
            oversample=oversample)
        self.subband_rate = self.channelizer.subband_rate
        self.centers = self.channelizer.subband_centers()
        self.window_samples = int(window_sec * self.subband_rate)
        self.step_samples = int(step_sec * self.subband_rate)
        self.energy_floor_db = energy_floor_db
        self.aggregator = SessionAggregator()
        # one detector per sub-band (each holds its own frequency state table)
        self._detectors = [Detector(sample_rate=self.subband_rate)
                           for _ in range(num_subbands)]

    def _read_all(self) -> np.ndarray:
        chunks = []
        while True:
            c = self.source.read_chunk()
            if c is None:
                break
            chunks.append(c)
        self.source.close()
        if not chunks:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(chunks)

    def _active_subbands(self, subbands: np.ndarray) -> list:
        # Energy gate: keep sub-bands whose mean power exceeds the 25th-percentile
        # noise-floor estimate by energy_floor_db.  Using percentile(25) rather than
        # min so the floor degrades gracefully as more sub-bands become active —
        # at least 25% of sub-bands must be quiet for the floor to stay anchored.
        power = np.mean(np.abs(subbands) ** 2, axis=1) + 1e-12
        power_db = 10 * np.log10(power)
        floor = np.percentile(power_db, 25)
        return [i for i in range(len(power_db))
                if power_db[i] >= floor + self.energy_floor_db]

    def _flush_active_calls(self, window_id: int) -> list:
        """Close every still-active call via expire(), mirroring scanner_rt idiom."""
        active = self.aggregator.active_calls()
        if not active:
            return []
        flush_window = (max((c.last_window for c in active), default=window_id)
                        + self.aggregator.timeout_windows)
        return self.aggregator.expire(flush_window, [])

    def run(self, on_call=None, max_windows=None) -> list:
        wide = self._read_all()
        if len(wide) == 0:
            return []
        subbands = self.channelizer.process(wide)        # (N, n_out)
        active = self._active_subbands(subbands)

        all_closed: list = []
        n_out = subbands.shape[1]
        n_windows = max(0, (n_out - self.window_samples) // self.step_samples + 1)
        if max_windows is not None:
            n_windows = min(n_windows, max_windows)

        for wid in range(n_windows):
            start = wid * self.step_samples
            stop = start + self.window_samples
            for i in active:
                win = subbands[i, start:stop]
                tasks = self._detectors[i].process_window(win, wid)
                for (iq, fo_rel, w) in tasks:
                    # No fo_rel guard: 2x oversampling means overlap-region signals
                    # legitimately appear in adjacent sub-bands with fo_rel up to
                    # ±full_bw.  The SessionAggregator merges same-RF detections from
                    # both sub-bands into one CallRecord via fo_bucket_hz keying.
                    pdus = decode_window(iq, fo_rel, w, self.subband_rate)
                    rf = self.center_hz + float(self.centers[i]) + fo_rel
                    for pdu in pdus:
                        pdu["_rf_hz"] = rf
                        self.aggregator.feed(pdu)
                closed = self.aggregator.expire(wid, self._detectors[i].closed_channels())
                for rec in closed:
                    all_closed.append(rec)
                    if on_call:
                        on_call(rec)

        # flush remaining active calls as timeout-closed
        for rec in self._flush_active_calls(n_windows):
            all_closed.append(rec)
            if on_call:
                on_call(rec)

        return all_closed
