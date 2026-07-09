from __future__ import annotations


SYMBOL_RATE = 4_800.0


class DMRSessionAssembler:
    """Build lightweight DMR call summaries from decoded signalling PDUs."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._src = 0
        self._dst = 0
        self._flco = ""
        self._fid = ""
        self._call_type = "unknown"
        self._color_code: int | None = None
        self._first_sample: int | None = None
        self._last_sample: int | None = None
        self._signalling_count = 0
        self._late_entry_count = 0
        self._csbk_count = 0

    def feed(self, pdu: dict, sps: int = 10) -> dict | None:
        ptype = pdu.get("type", "")
        if ptype not in {"LC_HEADER", "LATE_ENTRY", "TERMINATOR", "CSBK"}:
            return None

        if ptype == "TERMINATOR":
            if not self._active:
                return None
            self._update(pdu)
            out = self._emit(pdu, sps, closed_by="terminator")
            self.reset()
            return out

        if ptype in {"LC_HEADER", "LATE_ENTRY"}:
            if not self._active:
                self._active = True
                self._first_sample = _sample(pdu)
            self._update(pdu)
            if ptype == "LATE_ENTRY":
                self._late_entry_count += 1
        elif ptype == "CSBK":
            self._csbk_count += 1
            if not self._active and (pdu.get("src", 0) or pdu.get("dst", 0)):
                self._active = True
                self._first_sample = _sample(pdu)
                self._update(pdu)

        self._signalling_count += 1
        return None

    def finalize(self, sps: int = 10) -> dict | None:
        if not self._active:
            return None
        out = self._emit(None, sps, closed_by="end_of_scan")
        self.reset()
        return out

    def _update(self, pdu: dict) -> None:
        sample = _sample(pdu)
        if sample is not None:
            if self._first_sample is None:
                self._first_sample = sample
            self._last_sample = sample

        if pdu.get("src", 0):
            self._src = pdu.get("src", 0)
        if pdu.get("dst", 0):
            self._dst = pdu.get("dst", 0)
        if pdu.get("flco"):
            self._flco = str(pdu.get("flco", ""))
        if pdu.get("fid"):
            self._fid = str(pdu.get("fid", ""))

        extra = pdu.get("extra", {})
        if isinstance(extra, dict):
            if "color_code" in extra:
                self._color_code = extra.get("color_code")
            flc = extra.get("flc", {})
            if isinstance(flc, dict) and flc.get("call_type"):
                self._call_type = str(flc.get("call_type"))

    def _emit(self, last_pdu: dict | None, sps: int, closed_by: str) -> dict:
        first = self._first_sample or 0
        last = self._last_sample if self._last_sample is not None else first
        duration_s = max(0.0, (last - first) / (sps * SYMBOL_RATE))
        extra = {
            "call_type": self._call_type,
            "color_code": self._color_code,
            "start_sample": self._first_sample,
            "end_sample": self._last_sample,
            "duration_s": round(duration_s, 3),
            "signalling_count": self._signalling_count,
            "late_entry_count": self._late_entry_count,
            "csbk_count": self._csbk_count,
            "closed_by": closed_by,
        }
        if isinstance(last_pdu, dict):
            last_extra = last_pdu.get("extra", {})
            if isinstance(last_extra, dict):
                extra["last_data_type_name"] = last_extra.get("data_type_name")
                extra["last_sync_type"] = last_extra.get("sync_type")
        return {
            "protocol": "DMR",
            "type": "DMR_CALL",
            "src": self._src,
            "dst": self._dst,
            "ts": 0,
            "flco": self._flco,
            "fid": self._fid,
            "extra": extra,
            "raw_bits": b"",
        }


def _sample(pdu: dict) -> int | None:
    extra = pdu.get("extra", {})
    if not isinstance(extra, dict):
        return None
    value = extra.get("fs_start")
    return int(value) if isinstance(value, (int, float)) else None

