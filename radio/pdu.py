from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


_MISSING = object()
_STANDARD_KEYS = {
    "protocol",
    "type",
    "src",
    "dst",
    "ts",
    "flco",
    "fid",
    "extra",
    "raw_bits",
    "meta",
}


@dataclass
class PDU(Mapping[str, Any]):
    """Structured PDU boundary object with legacy dict-compatible reads."""

    type: str = ""
    protocol: str = "DMR"
    src: int | str = 0
    dst: int | str = 0
    ts: int | None = None
    flco: str = ""
    fid: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    raw_bits: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "PDU") -> "PDU":
        if isinstance(data, PDU):
            return data

        meta = dict(data.get("meta", {}))
        for key, value in data.items():
            if key.startswith("_"):
                meta.setdefault(key[1:], value)

        fields = {
            key: value
            for key, value in data.items()
            if key not in _STANDARD_KEYS and not key.startswith("_")
        }

        extra = data.get("extra", {})
        return cls(
            type=str(data.get("type", "")),
            protocol=str(data.get("protocol", "DMR")),
            src=data.get("src", 0),
            dst=data.get("dst", 0),
            ts=data.get("ts"),
            flco=str(data.get("flco", "")),
            fid=str(data.get("fid", "")),
            extra=dict(extra) if isinstance(extra, dict) else {},
            raw_bits=data.get("raw_bits"),
            meta=meta,
            fields=fields,
        )

    def to_dict(
        self,
        include_raw_bits: bool = True,
        include_meta: bool = False,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "protocol": self.protocol,
            "type": self.type,
            "src": self.src,
            "dst": self.dst,
            "ts": self.ts,
            "flco": self.flco,
            "fid": self.fid,
            "extra": dict(self.extra),
        }
        data.update(self.fields)
        if include_raw_bits and self.raw_bits is not None:
            data["raw_bits"] = self.raw_bits
        for key, value in self.meta.items():
            legacy_key = key if key.startswith("_") else f"_{key}"
            data[legacy_key] = value
        if include_meta:
            data["meta"] = dict(self.meta)
        return data

    def get(self, key: str, default: Any = None) -> Any:
        value = self._lookup(key)
        return default if value is _MISSING else value

    def _lookup(self, key: str) -> Any:
        if key == "protocol":
            return self.protocol
        if key == "type":
            return self.type
        if key == "src":
            return self.src
        if key == "dst":
            return self.dst
        if key == "ts":
            return self.ts
        if key == "flco":
            return self.flco
        if key == "fid":
            return self.fid
        if key == "extra":
            return self.extra
        if key == "raw_bits":
            return self.raw_bits
        if key == "meta":
            return self.meta
        if key.startswith("_"):
            stripped = key[1:]
            if stripped in self.meta:
                return self.meta[stripped]
            if key in self.meta:
                return self.meta[key]
        if key in self.fields:
            return self.fields[key]
        return _MISSING

    def __getitem__(self, key: str) -> Any:
        value = self._lookup(key)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def normalize_pdu(pdu: Mapping[str, Any] | PDU) -> PDU:
    return PDU.from_dict(pdu)


def pdu_get(pdu: Mapping[str, Any] | PDU, key: str, default: Any = None) -> Any:
    return pdu.get(key, default)


def pdu_to_dict(
    pdu: Mapping[str, Any] | PDU,
    include_raw_bits: bool = True,
    include_meta: bool = False,
) -> dict[str, Any]:
    if isinstance(pdu, PDU):
        return pdu.to_dict(
            include_raw_bits=include_raw_bits,
            include_meta=include_meta,
        )
    data = dict(pdu)
    if not include_raw_bits:
        data.pop("raw_bits", None)
    if not include_meta:
        data.pop("meta", None)
    return data


def pdu_to_standard_dict(
    pdu: Mapping[str, Any] | PDU,
    include_raw_bits: bool = True,
    include_meta: bool = False,
) -> dict[str, Any]:
    return normalize_pdu(pdu).to_dict(
        include_raw_bits=include_raw_bits,
        include_meta=include_meta,
    )


def pdus_to_standard_dicts(
    pdus: list[Mapping[str, Any] | PDU],
    include_raw_bits: bool = True,
    include_meta: bool = False,
) -> list[dict[str, Any]]:
    return [
        pdu_to_standard_dict(
            pdu,
            include_raw_bits=include_raw_bits,
            include_meta=include_meta,
        )
        for pdu in pdus
    ]


def set_pdu_meta(pdu: Mapping[str, Any] | PDU, key: str, value: Any) -> None:
    meta_key = key[1:] if key.startswith("_") else key
    if isinstance(pdu, PDU):
        pdu.meta[meta_key] = value
        return
    pdu[f"_{meta_key}"] = value
