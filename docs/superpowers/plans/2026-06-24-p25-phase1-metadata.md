# P25 Phase 1 Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a P25 Phase 1 metadata decode path that reuses the existing IQ/DDC/FM frontend and initially emits stable P25 NAC/DUID PDUs without disturbing the working DMR decoder.

**Architecture:** Keep DMR and P25 protocol logic separate. `core.dsp.frontend()` remains the shared discriminator output; a new `protocols.decode_all(y)` calls the existing DMR decode loop and a new `p25.decoder.decode(y)`. P25 uses frame-start sync (`FS`) as the anchor, so its symbol recovery is implemented in `p25/dsp.py` instead of reusing DMR's center-sync `recover_burst()`.

**Tech Stack:** Python, NumPy, SciPy, bitarray, pytest. No new runtime dependency for the first milestone.

---

## File Structure

Create:
- `p25/__init__.py`: package marker and public exports.
- `p25/constants.py`: P25 symbol mappings, frame sync constants, DUID names, status symbol cadence constants.
- `p25/sync.py`: P25 frame sync detection using NCC against the 48-bit FS rendered as 24 C4FM symbols.
- `p25/dsp.py`: frame-start anchored symbol recovery and four-level bit slicing.
- `p25/nid.py`: NID decode interface returning NAC and DUID. First milestone supports raw extraction and strict schema; BCH repair is added later in `p25/fec.py`.
- `p25/decoder.py`: top-level P25 decode loop that emits `P25_NID` PDUs.
- `protocols.py`: protocol dispatch layer that combines existing DMR decoding with P25 decoding.
- `tests/test_p25_constants.py`
- `tests/test_p25_sync.py`
- `tests/test_p25_dsp.py`
- `tests/test_p25_nid.py`
- `tests/test_p25_decoder.py`
- `tests/test_protocol_dispatch.py`

Modify:
- `scanner.py`: import `protocols` and change `_decode_loop(y)` to delegate to a renamed DMR-only helper plus P25 dispatch.
- `realtime/worker.py`: no direct P25 logic; it continues calling `scanner._decode_loop(y)`, so scanner integration covers realtime.

Deferred:
- `p25/fec.py`, `p25/framing.py`, `p25/lc.py`, `p25/tsbk.py`, `p25/session.py`: implement after the `P25_NID` path is passing and has an IQ sample oracle or synthetic FEC vectors.

---

### Task 1: P25 Constants And Symbol Mapping

**Files:**
- Create: `p25/__init__.py`
- Create: `p25/constants.py`
- Test: `tests/test_p25_constants.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_p25_constants.py
import numpy as np

from p25.constants import (
    DUID_NAMES,
    FRAME_SYNC_BITS,
    FRAME_SYNC_HEX,
    FRAME_SYNC_SYMBOLS,
    dibits_to_symbols,
    symbols_to_dibits,
)


def test_frame_sync_constant_shape():
    assert FRAME_SYNC_HEX == "5575F5FF77FF"
    assert len(FRAME_SYNC_BITS) == 48
    assert FRAME_SYNC_SYMBOLS.shape == (24,)
    assert set(FRAME_SYNC_SYMBOLS.tolist()) <= {-3, -1, 1, 3}


def test_dibit_symbol_round_trip():
    bits = "00011011"
    symbols = dibits_to_symbols(bits)
    assert np.array_equal(symbols, np.array([1, 3, -1, -3]))
    assert symbols_to_dibits(symbols) == bits


def test_duid_names_include_metadata_units():
    assert DUID_NAMES[0x0] == "HDU"
    assert DUID_NAMES[0x5] == "LDU1"
    assert DUID_NAMES[0x7] == "TSBK"
    assert DUID_NAMES[0xA] == "LDU2"
    assert DUID_NAMES[0xF] == "TDULC"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_p25_constants.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'p25'`.

- [ ] **Step 3: Implement constants**

```python
# p25/__init__.py
"""P25 Phase 1 metadata decoding package."""
```

```python
# p25/constants.py
from __future__ import annotations

import numpy as np

FRAME_SYNC_HEX = "5575F5FF77FF"
FRAME_SYNC_BITS = "".join(f"{int(c, 16):04b}" for c in FRAME_SYNC_HEX)

DIBIT_TO_SYMBOL = {
    "00": 1,
    "01": 3,
    "10": -1,
    "11": -3,
}
SYMBOL_TO_DIBIT = {v: k for k, v in DIBIT_TO_SYMBOL.items()}

DUID_NAMES = {
    0x0: "HDU",
    0x3: "TDU",
    0x5: "LDU1",
    0x7: "TSBK",
    0xA: "LDU2",
    0xC: "PDU",
    0xF: "TDULC",
}

FS_BITS = 48
NID_BITS = 64
FS_SYMBOLS = FS_BITS // 2
NID_SYMBOLS = NID_BITS // 2
FS_NID_SYMBOLS = FS_SYMBOLS + NID_SYMBOLS


def dibits_to_symbols(bits: str) -> np.ndarray:
    if len(bits) % 2 != 0:
        raise ValueError("dibit string length must be even")
    return np.array(
        [DIBIT_TO_SYMBOL[bits[i:i + 2]] for i in range(0, len(bits), 2)],
        dtype=float,
    )


def symbols_to_dibits(symbols: np.ndarray) -> str:
    levels = np.array([-3, -1, 1, 3])
    nearest = levels[np.argmin(np.abs(symbols[:, None] - levels[None, :]), axis=1)]
    return "".join(SYMBOL_TO_DIBIT[int(v)] for v in nearest)


FRAME_SYNC_SYMBOLS = dibits_to_symbols(FRAME_SYNC_BITS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_p25_constants.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add p25/__init__.py p25/constants.py tests/test_p25_constants.py
git commit -m "feat: add P25 constants"
```

---

### Task 2: P25 Frame Sync Detection

**Files:**
- Create: `p25/sync.py`
- Test: `tests/test_p25_sync.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_p25_sync.py
import numpy as np

from p25.constants import FRAME_SYNC_SYMBOLS
from p25.sync import P25SyncCandidate, find_frame_sync


def test_find_frame_sync_returns_start_anchor():
    sps = 10
    fs_start = 300
    y = np.random.default_rng(123).normal(0.0, 0.03, 900)
    y[fs_start:fs_start + len(FRAME_SYNC_SYMBOLS) * sps] += np.repeat(
        FRAME_SYNC_SYMBOLS,
        sps,
    )

    hits = find_frame_sync(y, sps=sps, threshold=0.85)

    assert hits
    assert isinstance(hits[0], P25SyncCandidate)
    assert abs(hits[0].fs_start - fs_start) <= 1
    assert hits[0].polarity == 1.0
    assert hits[0].ncc >= 0.85


def test_find_frame_sync_detects_inverted_polarity():
    sps = 10
    fs_start = 200
    y = np.zeros(700)
    y[fs_start:fs_start + len(FRAME_SYNC_SYMBOLS) * sps] = -np.repeat(
        FRAME_SYNC_SYMBOLS,
        sps,
    )

    hits = find_frame_sync(y, sps=sps, threshold=0.85)

    assert hits
    assert abs(hits[0].fs_start - fs_start) <= 1
    assert hits[0].polarity == -1.0


def test_find_frame_sync_returns_empty_for_short_signal():
    y = np.zeros(100)
    assert find_frame_sync(y, sps=10) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_p25_sync.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'p25.sync'`.

- [ ] **Step 3: Implement sync detection**

```python
# p25/sync.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

from p25.constants import FRAME_SYNC_SYMBOLS


@dataclass(frozen=True)
class P25SyncCandidate:
    fs_start: int
    polarity: float
    ncc: float


def find_frame_sync(
    y: np.ndarray,
    sps: int = 10,
    threshold: float = 0.62,
    min_distance_symbols: int = 120,
) -> list[P25SyncCandidate]:
    """Find P25 frame sync and return frame-start anchors.

    Unlike DMR, P25 FS is at the start of a data unit. The returned sample is
    the FS start, not the FS center.
    """
    ref = np.repeat(FRAME_SYNC_SYMBOLS, sps)
    if len(y) < len(ref):
        return []

    c = signal.correlate(y, ref, mode="same")
    e = np.convolve(y ** 2, np.ones(len(ref)), mode="same")
    e = np.where(e <= 0, 1e-9, e)
    ncc = c / np.sqrt(e * np.sum(ref ** 2))
    distance = max(1, min_distance_symbols * sps)

    hits: list[P25SyncCandidate] = []
    pos_peaks, pos_props = signal.find_peaks(ncc, height=threshold, distance=distance)
    neg_peaks, neg_props = signal.find_peaks(-ncc, height=threshold, distance=distance)

    half = len(ref) // 2
    for peak, height in zip(pos_peaks, pos_props["peak_heights"]):
        hits.append(P25SyncCandidate(int(peak - half), 1.0, float(height)))
    for peak, height in zip(neg_peaks, neg_props["peak_heights"]):
        hits.append(P25SyncCandidate(int(peak - half), -1.0, float(height)))

    hits = [h for h in hits if h.fs_start >= 0]
    hits.sort(key=lambda h: h.fs_start)
    return hits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_p25_sync.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add p25/sync.py tests/test_p25_sync.py
git commit -m "feat: add P25 frame sync detection"
```

---

### Task 3: P25 Frame-Start Symbol Recovery

**Files:**
- Create: `p25/dsp.py`
- Test: `tests/test_p25_dsp.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_p25_dsp.py
import numpy as np

from p25.constants import FRAME_SYNC_SYMBOLS
from p25.dsp import recover_symbols_from_fs, slice_symbols_to_bits
from p25.sync import P25SyncCandidate


def test_recover_symbols_from_fs_uses_start_anchor():
    sps = 10
    fs_start = 80
    payload = np.array([1, 3, -1, -3, 1, -1], dtype=float)
    symbols = np.concatenate([FRAME_SYNC_SYMBOLS, payload])
    y = np.zeros(600)
    y[fs_start:fs_start + len(symbols) * sps] = np.repeat(symbols * 1.7 + 0.4, sps)
    candidate = P25SyncCandidate(fs_start=fs_start, polarity=1.0, ncc=0.99)

    recovered = recover_symbols_from_fs(y, candidate, symbol_count=len(symbols), sps=sps)

    assert recovered is not None
    assert np.array_equal(np.round(recovered[-len(payload):]).astype(int), payload.astype(int))


def test_recover_symbols_from_fs_handles_inverted_signal():
    sps = 10
    fs_start = 80
    payload = np.array([3, 1, -1, -3], dtype=float)
    symbols = np.concatenate([FRAME_SYNC_SYMBOLS, payload])
    y = np.zeros(600)
    y[fs_start:fs_start + len(symbols) * sps] = -np.repeat(symbols, sps)
    candidate = P25SyncCandidate(fs_start=fs_start, polarity=-1.0, ncc=1.0)

    recovered = recover_symbols_from_fs(y, candidate, symbol_count=len(symbols), sps=sps)

    assert recovered is not None
    assert np.array_equal(np.round(recovered[-len(payload):]).astype(int), payload.astype(int))


def test_slice_symbols_to_bits_uses_p25_dibit_mapping():
    symbols = np.array([1, 3, -1, -3], dtype=float)
    assert slice_symbols_to_bits(symbols).to01() == "00011011"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_p25_dsp.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'p25.dsp'`.

- [ ] **Step 3: Implement frame-start recovery**

```python
# p25/dsp.py
from __future__ import annotations

import numpy as np
from bitarray import bitarray

from p25.constants import FRAME_SYNC_SYMBOLS, SYMBOL_TO_DIBIT
from p25.sync import P25SyncCandidate


def _interp(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    i = np.floor(pos).astype(int)
    fr = pos - i
    i = np.clip(i, 0, len(arr) - 2)
    return arr[i] * (1 - fr) + arr[i + 1] * fr


def recover_symbols_from_fs(
    y: np.ndarray,
    candidate: P25SyncCandidate,
    symbol_count: int,
    sps: int = 10,
    phase_search: np.ndarray | None = None,
) -> np.ndarray | None:
    """Recover symbols forward from P25 FS start.

    P25 uses a frame-start sync anchor. This function intentionally samples
    from `candidate.fs_start` forward; it does not apply DMR's center-sync
    burst offset.
    """
    if phase_search is None:
        phase_search = np.linspace(-4, 4, 33)

    levels = np.array([-3, -1, 1, 3])
    best: tuple[float, np.ndarray | None] = (1e18, None)
    for phase in phase_search:
        pos = candidate.fs_start + phase + np.arange(symbol_count) * sps
        if pos[0] < 0 or pos[-1] >= len(y) - 1:
            continue

        seg = candidate.polarity * _interp(y, pos)
        fs_seg = seg[:len(FRAME_SYNC_SYMBOLS)]
        a, b = np.linalg.lstsq(
            np.vstack([fs_seg, np.ones(len(fs_seg))]).T,
            FRAME_SYNC_SYMBOLS,
            rcond=None,
        )[0]
        calibrated = a * seg + b
        nearest = levels[
            np.argmin(np.abs(calibrated[:, None] - levels[None, :]), axis=1)
        ]
        resid = float(np.mean((calibrated[:len(FRAME_SYNC_SYMBOLS)] - FRAME_SYNC_SYMBOLS) ** 2))
        resid += 0.05 * float(np.mean((calibrated - nearest) ** 2))
        if resid < best[0]:
            best = (resid, calibrated)
    return best[1]


def slice_symbols_to_bits(symbols: np.ndarray) -> bitarray:
    levels = np.array([-3, -1, 1, 3])
    nearest = levels[np.argmin(np.abs(symbols[:, None] - levels[None, :]), axis=1)]
    bits = bitarray()
    bits.extend("".join(SYMBOL_TO_DIBIT[int(v)] for v in nearest))
    return bits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_p25_dsp.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add p25/dsp.py tests/test_p25_dsp.py
git commit -m "feat: recover P25 symbols from frame sync"
```

---

### Task 4: NID Decode Interface

**Files:**
- Create: `p25/nid.py`
- Test: `tests/test_p25_nid.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_p25_nid.py
from bitarray import bitarray

from p25.nid import P25NID, decode_nid


def make_nid_bits(nac: int, duid: int) -> bitarray:
    bits = bitarray(endian="big")
    bits.extend(f"{nac:012b}{duid:04b}")
    bits.extend("0" * 48)
    return bits


def test_decode_nid_extracts_nac_and_duid_schema():
    nid = decode_nid(make_nid_bits(0x293, 0x5))

    assert isinstance(nid, P25NID)
    assert nid.nac == 0x293
    assert nid.duid == 0x5
    assert nid.duid_name == "LDU1"
    assert nid.corrected is False
    assert nid.valid_bch is None


def test_decode_nid_rejects_wrong_length():
    short_bits = bitarray("0" * 63)
    try:
        decode_nid(short_bits)
    except ValueError as exc:
        assert "64 bits" in str(exc)
    else:
        raise AssertionError("decode_nid should reject non-64-bit input")


def test_decode_nid_names_unknown_duid():
    nid = decode_nid(make_nid_bits(0x123, 0x2))
    assert nid.duid_name == "UNKNOWN_0x2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_p25_nid.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'p25.nid'`.

- [ ] **Step 3: Implement NID schema and raw extraction**

```python
# p25/nid.py
from __future__ import annotations

from dataclasses import dataclass

from bitarray import bitarray
from bitarray.util import ba2int

from p25.constants import DUID_NAMES, NID_BITS


@dataclass(frozen=True)
class P25NID:
    nac: int
    duid: int
    duid_name: str
    valid_bch: bool | None
    corrected: bool
    raw_bits: bitarray


def decode_nid(bits: bitarray) -> P25NID:
    """Decode P25 NID shape.

    First milestone extracts the protected 16 information bits directly.
    BCH validation/repair is intentionally represented by `valid_bch=None`
    until `p25.fec` is implemented with vectors.
    """
    if len(bits) != NID_BITS:
        raise ValueError("P25 NID must be exactly 64 bits")
    nac = ba2int(bits[0:12])
    duid = ba2int(bits[12:16])
    return P25NID(
        nac=nac,
        duid=duid,
        duid_name=DUID_NAMES.get(duid, f"UNKNOWN_0x{duid:X}"),
        valid_bch=None,
        corrected=False,
        raw_bits=bits.copy(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_p25_nid.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add p25/nid.py tests/test_p25_nid.py
git commit -m "feat: add P25 NID decode interface"
```

---

### Task 5: P25 NID Decoder End-To-End On Synthetic Discriminator Output

**Files:**
- Create: `p25/decoder.py`
- Test: `tests/test_p25_decoder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_p25_decoder.py
import numpy as np
from bitarray import bitarray

from p25.constants import FRAME_SYNC_SYMBOLS, dibits_to_symbols
from p25.decoder import decode


def nid_symbols(nac: int, duid: int) -> np.ndarray:
    bits = f"{nac:012b}{duid:04b}" + "0" * 48
    return dibits_to_symbols(bits)


def test_decode_emits_p25_nid_pdu_from_synthetic_y():
    sps = 10
    fs_start = 120
    symbols = np.concatenate([FRAME_SYNC_SYMBOLS, nid_symbols(0x293, 0x7)])
    y = np.random.default_rng(456).normal(0.0, 0.02, 900)
    y[fs_start:fs_start + len(symbols) * sps] += np.repeat(symbols, sps)

    pdus = decode(y, sps=sps, sync_threshold=0.85)

    assert len(pdus) == 1
    pdu = pdus[0]
    assert pdu["protocol"] == "P25"
    assert pdu["type"] == "P25_NID"
    assert pdu["src"] == 0
    assert pdu["dst"] == 0
    assert pdu["flco"] == "TSBK"
    assert pdu["fid"] == ""
    assert pdu["extra"]["nac"] == 0x293
    assert pdu["extra"]["duid"] == 0x7
    assert pdu["extra"]["duid_name"] == "TSBK"
    assert "raw_bits" in pdu
    assert isinstance(pdu["raw_bits"], bytes)


def test_decode_returns_empty_when_no_frame_sync():
    y = np.zeros(1000)
    assert decode(y, sps=10, sync_threshold=0.85) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_p25_decoder.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'p25.decoder'`.

- [ ] **Step 3: Implement top-level P25 NID decode**

```python
# p25/decoder.py
from __future__ import annotations

import numpy as np

from p25.constants import FS_NID_SYMBOLS, FS_SYMBOLS, NID_SYMBOLS
from p25.dsp import recover_symbols_from_fs, slice_symbols_to_bits
from p25.nid import decode_nid
from p25.sync import find_frame_sync


def decode(
    y: np.ndarray,
    sps: int = 10,
    sync_threshold: float = 0.62,
) -> list[dict]:
    results: list[dict] = []
    for candidate in find_frame_sync(y, sps=sps, threshold=sync_threshold):
        symbols = recover_symbols_from_fs(
            y,
            candidate,
            symbol_count=FS_NID_SYMBOLS,
            sps=sps,
        )
        if symbols is None:
            continue
        bits = slice_symbols_to_bits(symbols)
        nid_bits = bits[FS_SYMBOLS * 2:(FS_SYMBOLS + NID_SYMBOLS) * 2]
        try:
            nid = decode_nid(nid_bits)
        except ValueError:
            continue
        results.append(
            {
                "protocol": "P25",
                "type": "P25_NID",
                "src": 0,
                "dst": 0,
                "ts": 0,
                "flco": nid.duid_name,
                "fid": "",
                "extra": {
                    "nac": nid.nac,
                    "duid": nid.duid,
                    "duid_name": nid.duid_name,
                    "valid_bch": nid.valid_bch,
                    "corrected": nid.corrected,
                    "fs_start": candidate.fs_start,
                    "sync_ncc": candidate.ncc,
                },
                "raw_bits": bits.tobytes(),
            }
        )
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_p25_decoder.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add p25/decoder.py tests/test_p25_decoder.py
git commit -m "feat: decode P25 NID frames"
```

---

### Task 6: Protocol Dispatch Without Disturbing DMR

**Files:**
- Create: `protocols.py`
- Modify: `scanner.py`
- Test: `tests/test_protocol_dispatch.py`
- Regression: `tests/test_decoder.py`, `tests/test_realtime_e2e.py`

- [ ] **Step 1: Write the failing protocol dispatch tests**

```python
# tests/test_protocol_dispatch.py
import numpy as np

import protocols


def test_decode_all_combines_dmr_and_p25(monkeypatch):
    def fake_dmr(y):
        return [{"protocol": "DMR", "type": "LC_HEADER", "src": 1, "dst": 2}]

    def fake_p25(y):
        return [{"protocol": "P25", "type": "P25_NID", "src": 0, "dst": 0}]

    monkeypatch.setattr(protocols, "decode_dmr", fake_dmr)
    monkeypatch.setattr(protocols, "decode_p25", fake_p25)

    result = protocols.decode_all(np.zeros(1000))

    assert [p["protocol"] for p in result] == ["DMR", "P25"]


def test_decode_dmr_adds_protocol_key(monkeypatch):
    def fake_loop(y):
        return [{"type": "CSBK", "src": 10, "dst": 20}]

    monkeypatch.setattr(protocols, "_dmr_decode_loop", fake_loop)

    result = protocols.decode_dmr(np.zeros(1000))

    assert result[0]["protocol"] == "DMR"
    assert result[0]["type"] == "CSBK"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_protocol_dispatch.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'protocols'`.

- [ ] **Step 3: Refactor scanner DMR loop and add dispatcher**

In `scanner.py`, rename the existing `_decode_loop` implementation to `_decode_dmr_loop` and replace `_decode_loop` with:

```python
def _decode_dmr_loop(y: np.ndarray) -> list[dict]:
    """Existing DMR-only decode loop.

    This is the old _decode_loop body. Keep all DMR behavior unchanged.
    """
    positions = find_sync_positions(y)
    results = []
    seen_bursts: set[tuple] = set()

    for center, polarity, sync_type in positions:
        dedup_key = (round(center / 50), sync_type)
        if dedup_key in seen_bursts:
            continue
        seen_bursts.add(dedup_key)

        if "VOICE" in sync_type:
            ph = _lock_voice_phase(y, center, polarity, sync_type)
            collector = LateEntryCollector()
            for j in range(6):
                ba = _recover_stepped_burst(y, center, j, ph, polarity)
                if ba is None:
                    break
                pdu = collector.feed(ba, sync_type)
                if pdu is not None:
                    results.append(dict(pdu))
                    break
        else:
            symbols = recover_burst(y, center, polarity, sync_type)
            if symbols is None:
                continue
            pdu = decode_burst(symbols, sync_type)
            if pdu is not None:
                results.append(dict(pdu))

    return results


def _decode_loop(y: np.ndarray) -> list[dict]:
    import protocols

    return protocols.decode_all(y)
```

Create `protocols.py`:

```python
from __future__ import annotations

import numpy as np

import scanner
from p25.decoder import decode as decode_p25


def _dmr_decode_loop(y: np.ndarray) -> list[dict]:
    return scanner._decode_dmr_loop(y)


def decode_dmr(y: np.ndarray) -> list[dict]:
    pdus = _dmr_decode_loop(y)
    for pdu in pdus:
        pdu.setdefault("protocol", "DMR")
    return pdus


def decode_all(y: np.ndarray) -> list[dict]:
    results: list[dict] = []
    results.extend(decode_dmr(y))
    results.extend(decode_p25(y))
    return results
```

- [ ] **Step 4: Run dispatch tests**

Run: `pytest tests/test_protocol_dispatch.py -v`

Expected: PASS.

- [ ] **Step 5: Run DMR regression tests**

Run: `pytest tests/test_decoder.py tests/test_realtime_e2e.py -v`

Expected: PASS. If sample data is absent, existing tests that depend on files should return without failing as they do today.

- [ ] **Step 6: Commit**

```bash
git add protocols.py scanner.py tests/test_protocol_dispatch.py
git commit -m "feat: dispatch DMR and P25 decoders"
```

---

### Task 7: Offline Scanner Output Supports Protocol Field

**Files:**
- Modify: `scanner.py`
- Test: `tests/test_protocol_dispatch.py`

- [ ] **Step 1: Add failing output-format test**

Append to `tests/test_protocol_dispatch.py`:

```python
def test_print_results_accepts_p25_nid(capsys):
    import scanner

    scanner._print_results([
        {
            "protocol": "P25",
            "type": "P25_NID",
            "src": 0,
            "dst": 0,
            "flco": "LDU1",
            "fid": "",
            "extra": {"nac": 0x293, "duid": 0x5},
        }
    ])

    out = capsys.readouterr().out
    assert "P25_NID" in out
    assert "PROTO=P25" in out
    assert "NAC=0x293" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_protocol_dispatch.py::test_print_results_accepts_p25_nid -v`

Expected: FAIL because `_print_results` does not print protocol or NAC.

- [ ] **Step 3: Update `_print_results`**

Replace `scanner._print_results` with:

```python
def _print_results(pdus: list[dict]) -> None:
    for p in pdus:
        fo_str = f" (fo={p['_fo_hz']/1e3:+.1f}kHz)" if "_fo_hz" in p else ""
        proto = p.get("protocol", "DMR")
        extra = p.get("extra", {})
        nac_str = f" NAC=0x{extra['nac']:03X}" if proto == "P25" and "nac" in extra else ""
        print(
            f"[{p['type']:<12}] PROTO={proto} SRC={p['src']} DST={p['dst']} "
            f"FLCO={p['flco']} FID={p.get('fid','')}{nac_str}{fo_str}"
        )
```

- [ ] **Step 4: Run output-format test**

Run: `pytest tests/test_protocol_dispatch.py::test_print_results_accepts_p25_nid -v`

Expected: PASS.

- [ ] **Step 5: Run scanner and dispatcher regression tests**

Run: `pytest tests/test_protocol_dispatch.py tests/test_scanner.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scanner.py tests/test_protocol_dispatch.py
git commit -m "feat: print protocol metadata in scanner results"
```

---

### Task 8: Full Test Run And Follow-Up Notes

**Files:**
- Modify: `docs/superpowers/plans/2026-06-24-p25-phase1-metadata.md` only if execution reveals a correction needed in this plan.

- [ ] **Step 1: Run focused P25 tests**

Run:

```bash
pytest tests/test_p25_constants.py tests/test_p25_sync.py tests/test_p25_dsp.py tests/test_p25_nid.py tests/test_p25_decoder.py tests/test_protocol_dispatch.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Manually inspect scanner behavior on available DMR sample**

Run:

```bash
python scanner.py data/dmr_1_78125.rawiq
```

Expected: Existing DMR output still appears. Each printed line includes `PROTO=DMR`.

- [ ] **Step 4: Document next milestone in final handoff**

Report these exact next milestone options:

```text
Next P25 milestone:
1. Add BCH(63,16,23) validation/repair for NID.
2. Add known-vector tests for BCH before using it in real IQ decode.
3. Extend p25.decoder from P25_NID to DUID-specific framing for TSBK and LDU1 LC.
```

No commit is required for this task unless test execution forces a plan correction.

---

## Self-Review

Spec coverage:
- Shared DSP frontend and scanner/realtime integration are covered by Tasks 6 and 7.
- P25 start-of-frame synchronization is covered by Tasks 2 and 3.
- NAC/DUID identification is covered by Tasks 4 and 5.
- LDU1 LC, TSBK, full FEC, and session aggregation are explicitly deferred because the first working milestone is P25 identification through NID. They should be implemented after BCH vectors are available.

Placeholder scan:
- No `TBD`, `TODO`, or placeholder-only implementation steps remain.

Type consistency:
- `P25SyncCandidate.fs_start`, `polarity`, and `ncc` are introduced in Task 2 and used consistently in Tasks 3 and 5.
- `P25NID.nac`, `duid`, `duid_name`, `valid_bch`, and `corrected` are introduced in Task 4 and used consistently in Task 5.
- PDU schema uses `protocol`, `type`, `src`, `dst`, `ts`, `flco`, `fid`, `extra`, and `raw_bits`, matching the existing scanner expectations while adding protocol metadata.
