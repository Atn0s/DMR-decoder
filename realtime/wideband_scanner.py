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
        # Owning-sub-band half-width: fs/N/2 = subband_rate/(2*oversample).
        # A uniform polyphase filterbank tiles the spectrum so that every frequency
        # belongs to exactly one "owner" — the sub-band whose center is nearest.
        # The primary (owning) region of sub-band i is the ±half-width interval
        # around its center.  Oversampled capture extends each sub-band's CAPTURED
        # bandwidth beyond this half-width into the neighbour's primary region, so
        # a real signal at fo_rel in [±half_width, ±full_bw) is also visible as an
        # alias in the adjacent sub-band.  We skip those alias detections here; the
        # signal is decoded by its true owner (the adjacent sub-band) where fo_rel
        # is within ±half_width.  This is NOT a dead zone: every real frequency has
        # exactly one owner and is decoded exactly once with full SNR margin.
        self._owning_halfwidth_hz = self.subband_rate / (2 * oversample)

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
                    # Owning-sub-band guard (nearest-center / primary-region rule).
                    # Each sub-band owns the ±_owning_halfwidth_hz interval around
                    # its center.  With oversampling, the captured bandwidth is wider,
                    # so a real signal can also be detected in an adjacent sub-band
                    # (alias) with |fo_rel| > _owning_halfwidth_hz.  Those alias
                    # detections estimate a slightly different fo_rel there, so
                    # _rf_hz resolves to a WRONG absolute RF that the aggregator
                    # cannot merge with the true call — they survive as phantoms.
                    # Skipping detections outside the primary region eliminates the
                    # phantoms.  This is NOT a dead zone: the real signal is in the
                    # PRIMARY region of the adjacent sub-band (its true owner), where
                    # |fo_rel| < _owning_halfwidth_hz and it IS decoded normally.
                    # Primary regions of a uniform filterbank tile the band with no
                    # gaps, so every real signal is decoded exactly once.
                    if abs(fo_rel) > self._owning_halfwidth_hz:
                        continue
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
