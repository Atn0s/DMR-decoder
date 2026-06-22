import threading
from multiprocessing import Pool

from realtime.ring_buffer import RingBuffer
from realtime.detector import Detector
from realtime.aggregator import SessionAggregator, CallRecord
from realtime.worker import decode_window
from realtime.iq_source import FileIQSource
from realtime.wideband_source import FileWidebandSource
from realtime.wideband_scanner import WidebandScanner


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

    def _flush_active_calls(self, window_id: int) -> list[CallRecord]:
        """Unconditionally close every still-active call as timeout-closed and
        remove it from the aggregator's active set; return the closed records.

        The aggregator is a completed/reviewed module, so we work through its
        public surface only. The single public path that both closes and removes
        active calls is expire(). To guarantee EVERY active call is closed
        regardless of the window the loop ended on, we call expire() with a window
        id chosen to exceed every active call's last_window by at least
        timeout_windows. This is computed from the active calls themselves, so the
        flush is self-contained and not coupled to the loop's window_id.
        """
        active = self.aggregator.active_calls()
        if not active:
            return []
        flush_window = (max((c.last_window for c in active), default=window_id)
                        + self.aggregator.timeout_windows)
        return self.aggregator.expire(flush_window, [])

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

            # Flush remaining active calls as timeout-closed
            final = self._flush_active_calls(window_id)
            for rec in final:
                all_closed.append(rec)
                if on_call:
                    on_call(rec)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            self.source.close()

        return all_closed


def run_wideband_cli(args) -> list:
    """Run a wideband channelizer scan from parsed CLI args. Returns CallRecords."""
    src = FileWidebandSource(args.path, sample_rate=args.fs,
                             center_hz=getattr(args, "center", 0.0),
                             chunk_samples=int(args.fs), throttle=False)
    scanner = WidebandScanner(src, num_subbands=args.nsub,
                              oversample=args.oversample)

    def on_call(c):
        print(f"[CALL] RF={c.fo_hz/1e6:.4f}MHz SRC={c.src} DST={c.dst} "
              f"FLCO={c.flco} closed_by={c.closed_by} "
              f"windows={c.start_window}-{c.end_window}")

    calls = scanner.run(on_call=on_call)
    print(f"=== total wideband calls: {len(calls)} ===")
    return calls


def _detect_sample_rate(path: str) -> float | None:
    """Reuse scanner.detect_sample_rate to infer fs from filename (e.g. _78125.rawiq)."""
    import scanner
    fs = scanner.detect_sample_rate(path)
    return float(fs) if fs else None


def main():
    """CLI entry point: stream an offline .rawiq file through the realtime pipeline.

    Usage:
        python -m realtime.scanner_rt <file.rawiq> [--fs HZ] [--workers N]
                                      [--throttle] [--pool]
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="DMR realtime scanner — stream an IQ file through the realtime pipeline")
    parser.add_argument("path", help="path to a .rawiq file (interleaved int16 IQ)")
    parser.add_argument("--fs", type=float, default=None,
                        help="sample rate in Hz (default: inferred from filename, "
                             "e.g. dmr_1_78125.rawiq -> 78125)")
    parser.add_argument("--workers", type=int, default=2,
                        help="number of decode workers (default: 2)")
    parser.add_argument("--throttle", action="store_true",
                        help="throttle file reads to real-time pacing (simulate live SDR)")
    parser.add_argument("--pool", action="store_true",
                        help="use a multiprocessing worker pool (default: serial)")
    parser.add_argument("--chunk", type=int, default=None,
                        help="acquisition chunk size in samples (default: ~1s of data)")
    parser.add_argument("--wideband", action="store_true",
                        help="run wideband channelizer scan (PFB front-end)")
    parser.add_argument("--center", type=float, default=0.0,
                        help="absolute RF center of the captured band, Hz")
    parser.add_argument("--nsub", type=int, default=32,
                        help="number of channelizer sub-bands (default 32)")
    parser.add_argument("--oversample", type=int, default=2,
                        help="channelizer oversample factor (default 2)")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        parser.error(f"file not found: {args.path}")

    if args.wideband:
        if args.fs is None:
            fs = _detect_sample_rate(args.path)
            if fs is None:
                parser.error("could not infer sample rate; pass --fs HZ")
            args.fs = fs
        print(f"=== Wideband channelizer scan: {args.path} "
              f"(fs={args.fs/1e6:.3f} MHz, center={args.center/1e6:.3f} MHz, "
              f"nsub={args.nsub}, oversample={args.oversample}) ===")
        run_wideband_cli(args)
        return

    fs = args.fs or _detect_sample_rate(args.path)
    if fs is None:
        parser.error("could not infer sample rate from filename; pass --fs HZ")

    chunk = args.chunk or int(fs)
    print(f"=== Realtime scan: {args.path} (fs={fs/1e6:.4f} MHz, "
          f"workers={args.workers}, throttle={args.throttle}, pool={args.pool}) ===")

    src = FileIQSource(args.path, sample_rate=fs, chunk_samples=chunk,
                       throttle=args.throttle)
    rt = RealtimeScanner(src, num_workers=args.workers, use_pool=args.pool)

    def on_call(c: CallRecord):
        fo_str = f"fo={c.fo_hz/1e3:+.1f}kHz " if c.fo_hz else ""
        print(f"[CALL] {fo_str}SRC={c.src} DST={c.dst} FLCO={c.flco} "
              f"closed_by={c.closed_by} windows={c.start_window}-{c.end_window}")

    calls = rt.run(on_call=on_call)
    print(f"=== total calls: {len(calls)} ===")


if __name__ == "__main__":
    main()
