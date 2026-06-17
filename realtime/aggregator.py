from dataclasses import dataclass, field

CALL_TIMEOUT_WINDOWS = 5   # close a call after this many windows with no update


@dataclass
class CallRecord:
    fo_hz: float
    src: int
    dst: int
    flco: str
    start_window: int
    end_window: int | None = None
    voice_raw: list = field(default_factory=list)
    closed_by: str = ""
    last_window: int = 0


class SessionAggregator:
    """Merge fragmented PDUs from workers into call records.
    Merge key: (fo_bucket, src, dst).
    Dedup boundaries:
      - cross-window same signalling (LC/CSBK/Terminator): recorded once
      - voice frames: accumulated in time order (NOT deduped) into voice_raw"""

    def __init__(self, fo_bucket_hz: float = 5000.0,
                 timeout_windows: int = CALL_TIMEOUT_WINDOWS):
        self.fo_bucket_hz = fo_bucket_hz
        self.timeout_windows = timeout_windows
        self._calls: dict[tuple, CallRecord] = {}
        self._pending_closed: list[CallRecord] = []

    def _key(self, pdu: dict) -> tuple:
        bucket = round(pdu.get("_fo_hz", 0.0) / self.fo_bucket_hz) * self.fo_bucket_hz
        return (bucket, pdu["src"], pdu["dst"])

    def feed(self, pdu: dict) -> None:
        key = self._key(pdu)
        wid = pdu.get("_window_id", 0)
        ptype = pdu["type"]

        rec = self._calls.get(key)

        if ptype == "TERMINATOR":
            # Only close an EXISTING active call. A TERMINATOR with no matching
            # active call is a duplicate from an overlapping window — ignore it
            # to prevent a phantom CallRecord being fabricated and closed.
            if rec is None:
                return
            rec.end_window = wid
            rec.closed_by = "terminator"
            self._pending_closed.append(rec)
            del self._calls[key]
            return

        # LC_HEADER / LATE_ENTRY / CSBK: open-or-hit the call record
        if rec is None:
            rec = CallRecord(
                fo_hz=key[0], src=pdu["src"], dst=pdu["dst"],
                flco=pdu.get("flco", ""), start_window=wid, last_window=wid)
            self._calls[key] = rec
        rec.last_window = max(rec.last_window, wid)

        if ptype == "LATE_ENTRY":
            # voice/embedded fragments accumulate (different time segments)
            rec.voice_raw.append(pdu.get("raw_bits", b""))

    def expire(self, current_window: int, closed_fos: list[float]) -> list[CallRecord]:
        closed = list(self._pending_closed)
        self._pending_closed = []

        closed_buckets = {round(fo / self.fo_bucket_hz) * self.fo_bucket_hz
                          for fo in closed_fos}

        for key, rec in list(self._calls.items()):
            bucket = key[0]
            if bucket in closed_buckets:
                rec.end_window = current_window
                rec.closed_by = "detector"
                closed.append(rec)
                del self._calls[key]
            elif current_window - rec.last_window >= self.timeout_windows:
                rec.end_window = current_window
                rec.closed_by = "timeout"
                closed.append(rec)
                del self._calls[key]
        return closed

    def active_calls(self) -> list[CallRecord]:
        return list(self._calls.values())
