from __future__ import annotations

from p25.framing import P25FrameInfo
from p25.link_control import LinkControl

SYMBOL_RATE = 4800.0


class P25SessionAssembler:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._nac: int | None = None
        self._src = 0
        self._dst = 0
        self._is_group = False
        self._first_fs: int | None = None
        self._ldu_count = 0

    def feed(
        self,
        frame_info: P25FrameInfo,
        link_control: LinkControl | None,
        fs_start: int,
        sps: int = 10,
    ) -> dict | None:
        if frame_info.is_terminator:
            if not self._active:
                return None
            pdu = self._emit(fs_start, sps)
            self.reset()
            return pdu

        if not self._active and frame_info.duid_name in ("HDU", "LDU1", "LDU2"):
            self._active = True
            self._nac = frame_info.nac
            self._first_fs = fs_start

        if self._active:
            if frame_info.is_voice:
                self._ldu_count += 1
            if link_control is not None:
                self._src = link_control.src
                self._dst = link_control.dst
                self._is_group = link_control.is_group
        return None

    def _emit(self, fs_start: int, sps: int) -> dict:
        first = self._first_fs if self._first_fs is not None else fs_start
        duration_s = (fs_start - first) / (sps * SYMBOL_RATE)
        return {
            "protocol": "P25",
            "type": "P25_CALL",
            "src": self._src,
            "dst": self._dst,
            "ts": 0,
            "flco": "GROUP" if self._is_group else "",
            "fid": "",
            "extra": {
                "nac": self._nac,
                "duration_s": round(duration_s, 3),
                "ldu_count": self._ldu_count,
            },
            "raw_bits": b"",
        }
