from __future__ import annotations

from dataclasses import dataclass

from dpmr.cch import CCHRecord, air_interface_id_to_str


def cch_record_usable(record: CCHRecord) -> bool:
    return record.crc_ok or record.hamming_ok


@dataclass
class DPMRSessionAssembler:
    dst: str = ""
    src: str = ""
    records: dict[int, CCHRecord] | None = None

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = {}

    def _store(self, record: CCHRecord) -> None:
        assert self.records is not None
        current = self.records.get(record.frame_number)
        if current is None or (record.crc_ok and not current.crc_ok):
            self.records[record.frame_number] = record

    def _assemble_pair(self, first: int, second: int) -> str:
        assert self.records is not None
        if first not in self.records or second not in self.records:
            return ""
        value = (
            (self.records[first].id_half << 12)
            | self.records[second].id_half
        ) & 0xFFFFFF
        return air_interface_id_to_str(value)

    def feed(self, cch0: CCHRecord | None, cch1: CCHRecord | None) -> tuple[str, str, str]:
        records = [rec for rec in (cch0, cch1) if rec is not None and rec.crc_ok]
        part = "unknown"
        for record in records:
            self._store(record)
        dst = self._assemble_pair(0, 1)
        src = self._assemble_pair(2, 3)
        if dst:
            self.dst = dst
            part = "dst"
        if src:
            self.src = src
            part = "src"
        return self.src, self.dst, part
