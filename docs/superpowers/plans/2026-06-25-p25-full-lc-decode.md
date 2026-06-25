# P25 Full Link Control Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode P25 Phase 1 LDU1 Link Control to recover real SrcID/TGID, assemble per-call sessions, and fix the scanner dedup that collapses all P25 frames into one — reaching DMR-level decode output.

**Architecture:** Extend the existing NID-level `p25/` package (20 tests green). Add pure-Python FEC (`p25/fec.py`), an LDU1 LC field parser (`p25/link_control.py`), and a session assembler (`p25/session.py`). Each FS anchor independently recovers its own 864-symbol LDU frame (segmented re-sync, no cross-frame PLL), deinterleaves the 240 LC-encoded bits, decodes Hamming(10,6,3)×24 → RS(24,12,13) over GF(2⁶) → 72-bit LC. Scanner dedup becomes protocol-aware.

**Tech Stack:** Python 3.10, numpy, scipy.signal, bitarray, pytest. FEC self-implemented (no new deps), referencing the existing `okdmr` Reed-Solomon code skeleton (GF(2⁸)) adapted to GF(2⁶).

## Global Constraints

- **Zero intrusion into the DMR path.** `scanner._decode_dmr_loop` and all `core/` code must remain byte-identical except for the protocol-aware dedup dispatch in `scan_file`. DMR regression (`tests/test_p25_*` plus existing DMR tests) must stay green.
- **No new third-party dependencies.** FEC is pure Python, consistent with the `okdmr` style already in the repo.
- **TDD throughout.** Write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- **GF(2⁶) primitive polynomial: x⁶+x+1 = 0x43.** Field natural length 63. RS(24,12,13) is a shortened code, t=6.
- **LDU frame length: 864 symbols / 180ms @ 4800 sym/s** (verified on `data/p25_1_78125.rawiq`: adjacent FS spacing 864 symbols appears 56×, dominant). At 48kHz / SPS=10 that is 8640 samples.
- **Sample file:** `data/p25_1_78125.rawiq`, NAC=0x293, 19.5s, ~112 frame-sync hits at threshold 0.62.
- **PDU dict shape** must match DMR: keys `protocol, type, src, dst, ts, flco, fid, extra, raw_bits`.

---

## File Structure

```
New:
  p25/galois.py        GF(2^6) field: exp/log tables, mul/inv/poly ops
  p25/fec.py           bch_63_16_decode, hamming_10_6_3_decode, rs_24_12_13_decode, crc16_ccitt
  p25/link_control.py  LinkControl dataclass + parse_link_control(lc72)
  p25/session.py       P25SessionAssembler (feed/reset)
Modified:
  p25/constants.py     LDU layout constants: LDU_SYMBOLS=864, LC interleave indices
  p25/dsp.py           recover_full_frame(864 symbols) + deinterleave_lc
  p25/nid.py           wire in bch_63_16_decode (valid_bch no longer None)
  p25/decoder.py       LDU1 -> LC decode -> src/dst; feed session
  scanner.py           protocol-aware dedup in scan_file
Tests:
  tests/test_p25_galois.py
  tests/test_p25_fec.py
  tests/test_p25_link_control.py
  tests/test_p25_session.py
  tests/test_p25_dsp.py        (extend)
  tests/test_p25_nid.py        (extend)
  tests/test_p25_decoder.py    (extend)
  tests/test_p25_e2e.py        (new)
```

---

# Milestone 1 — FEC (pure Python, locked by vectors)

## Task 1: GF(2⁶) field arithmetic

**Files:**
- Create: `p25/galois.py`
- Test: `tests/test_p25_galois.py`

**Interfaces:**
- Produces:
  - `GF6_EXP: list[int]` (length 126, antilog table, α^i for i in 0..125 with wraparound)
  - `GF6_LOG: list[int]` (length 64, log table, LOG[0] unused)
  - `gf6_mul(a: int, b: int) -> int`
  - `gf6_inv(a: int) -> int`
  - `gf6_poly_eval(poly: list[int], x: int) -> int` (poly is list of GF elements, highest degree first)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_p25_galois.py
from p25.galois import GF6_EXP, GF6_LOG, gf6_mul, gf6_inv, gf6_poly_eval


def test_field_has_63_nonzero_elements():
    # alpha^0 .. alpha^62 are the 63 distinct non-zero elements, alpha^63 == 1
    assert GF6_EXP[0] == 1
    assert GF6_EXP[63] == 1
    seen = {GF6_EXP[i] for i in range(63)}
    assert len(seen) == 63
    assert 0 not in seen


def test_primitive_polynomial_x6_x_1():
    # alpha^6 = alpha + 1 = 0b000011 = 3  (x^6 = x + 1 mod 0x43)
    assert GF6_EXP[6] == 0b000011


def test_log_exp_round_trip():
    for a in range(1, 64):
        assert GF6_EXP[GF6_LOG[a]] == a


def test_mul_matches_log_addition():
    assert gf6_mul(0, 5) == 0
    assert gf6_mul(5, 0) == 0
    # alpha^2 * alpha^3 = alpha^5
    assert gf6_mul(GF6_EXP[2], GF6_EXP[3]) == GF6_EXP[5]


def test_inverse():
    for a in range(1, 64):
        assert gf6_mul(a, gf6_inv(a)) == 1


def test_poly_eval_constant():
    assert gf6_poly_eval([7], 5) == 7
    # poly = x + 1 (highest degree first: [1, 1]); eval at x=1 -> 1*1 + 1 = 0
    assert gf6_poly_eval([1, 1], 1) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_galois.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'p25.galois'`

- [ ] **Step 3: Write minimal implementation**

```python
# p25/galois.py
"""GF(2^6) arithmetic for P25 Reed-Solomon. Primitive polynomial x^6+x+1 = 0x43."""
from __future__ import annotations

GF6_PRIM = 0x43  # x^6 + x + 1
GF6_SIZE = 64

GF6_EXP: list[int] = [0] * 126
GF6_LOG: list[int] = [0] * GF6_SIZE


def _build_tables() -> None:
    x = 1
    for i in range(63):
        GF6_EXP[i] = x
        GF6_LOG[x] = i
        x <<= 1
        if x & GF6_SIZE:          # bit 6 set -> reduce mod primitive poly
            x ^= GF6_PRIM
        x &= 0x3F                 # keep 6 bits
    # duplicate for wraparound so EXP[i+j] works without mod for i,j < 63
    for i in range(63, 126):
        GF6_EXP[i] = GF6_EXP[i - 63]


_build_tables()


def gf6_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return GF6_EXP[GF6_LOG[a] + GF6_LOG[b]]


def gf6_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF(2^6) inverse of 0")
    return GF6_EXP[63 - GF6_LOG[a]]


def gf6_poly_eval(poly: list[int], x: int) -> int:
    """Horner evaluation. poly[0] is the highest-degree coefficient."""
    acc = 0
    for coeff in poly:
        acc = gf6_mul(acc, x) ^ coeff
    return acc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_galois.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add p25/galois.py tests/test_p25_galois.py
git commit -m "feat: add GF(2^6) field arithmetic for P25 FEC"
```

---

## Task 2: Hamming(10,6,3) decode

**Files:**
- Create: `p25/fec.py`
- Test: `tests/test_p25_fec.py`

**Interfaces:**
- Consumes: nothing (self-contained block code)
- Produces: `hamming_10_6_3_decode(bits10: bitarray) -> tuple[bitarray, bool]` — returns the 6 data bits and whether a single-bit error was corrected. The P25 shortened Hamming(10,6,3) parity matrix: each 10-bit codeword is `[d0..d5 | p0..p3]` where the 4 parity bits are generated by the matrix below.

P25 Hamming(10,6,3) — the standard TIA-102 generator. The 4 parity check equations (parity bit = XOR of the marked data bits), data bits d0..d5:

```
p0 = d0 ^ d1 ^ d2 ^ d3
p1 = d1 ^ d2 ^ d3 ^ d4
p2 = d0 ^ d1 ^ d4 ^ d5
p3 = d0 ^ d2 ^ d4 ^ d5
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_p25_fec.py
from bitarray import bitarray
from p25.fec import hamming_10_6_3_decode


def _encode_hamming(d: str) -> bitarray:
    b = [int(c) for c in d]
    p0 = b[0] ^ b[1] ^ b[2] ^ b[3]
    p1 = b[1] ^ b[2] ^ b[3] ^ b[4]
    p2 = b[0] ^ b[1] ^ b[4] ^ b[5]
    p3 = b[0] ^ b[2] ^ b[4] ^ b[5]
    out = bitarray()
    out.extend(d)
    out.extend(str(p0) + str(p1) + str(p2) + str(p3))
    return out


def test_hamming_decodes_clean_codeword():
    cw = _encode_hamming("101100")
    data, corrected = hamming_10_6_3_decode(cw)
    assert data.to01() == "101100"
    assert corrected is False


def test_hamming_corrects_single_bit_error():
    cw = _encode_hamming("101100")
    cw[2] ^= 1  # flip one data bit
    data, corrected = hamming_10_6_3_decode(cw)
    assert data.to01() == "101100"
    assert corrected is True


def test_hamming_corrects_parity_bit_error():
    cw = _encode_hamming("110011")
    cw[9] ^= 1  # flip a parity bit
    data, corrected = hamming_10_6_3_decode(cw)
    assert data.to01() == "110011"
    assert corrected is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_fec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'p25.fec'`

- [ ] **Step 3: Write minimal implementation**

```python
# p25/fec.py
"""P25 Phase 1 forward error correction: Hamming(10,6,3), RS(24,12,13)/GF(2^6),
BCH(63,16,23), CRC-16. Pure Python, no third-party deps."""
from __future__ import annotations

from bitarray import bitarray

from p25.galois import GF6_EXP, GF6_LOG, gf6_mul, gf6_inv

# Hamming(10,6,3): parity generator rows over data bits d0..d5.
_HAMMING_PARITY = (
    (0, 1, 2, 3),   # p0
    (1, 2, 3, 4),   # p1
    (0, 1, 4, 5),   # p2
    (0, 2, 4, 5),   # p3
)


def _hamming_parity(d: list[int]) -> list[int]:
    return [d[a] ^ d[b] ^ d[c] ^ d[e] for (a, b, c, e) in _HAMMING_PARITY]


def hamming_10_6_3_decode(bits10: bitarray) -> tuple[bitarray, bool]:
    """Decode one P25 Hamming(10,6,3) codeword. Returns (6 data bits, corrected?)."""
    if len(bits10) != 10:
        raise ValueError("Hamming(10,6,3) codeword must be 10 bits")
    rx = [int(x) for x in bits10]
    data = rx[:6]
    rx_par = rx[6:10]
    syndrome = [_hamming_parity(data)[i] ^ rx_par[i] for i in range(4)]
    corrected = False
    if any(syndrome):
        # Try flipping each data bit; if it zeroes the syndrome, accept.
        for i in range(6):
            trial = data[:]
            trial[i] ^= 1
            if _hamming_parity(trial) == rx_par:
                data = trial
                corrected = True
                break
        else:
            # error is in a parity bit (data is fine); still flag corrected
            corrected = True
    out = bitarray()
    out.extend("".join(str(x) for x in data))
    return out, corrected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_fec.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add p25/fec.py tests/test_p25_fec.py
git commit -m "feat: add P25 Hamming(10,6,3) decode"
```

---

## Task 3: Reed-Solomon(24,12,13) over GF(2⁶)

**Files:**
- Modify: `p25/fec.py`
- Test: `tests/test_p25_fec.py` (extend)

**Interfaces:**
- Consumes: `p25.galois` (gf6_mul, gf6_inv, GF6_EXP, GF6_LOG)
- Produces: `rs_24_12_13_decode(hexbits: list[int]) -> tuple[list[int] | None, bool]` — input is 24 GF(2⁶) symbols (6-bit ints), most-significant symbol first: 12 data + 12 parity. Returns (12 corrected data symbols, ok). `ok=False` and `None` data if the error count exceeds t=6 (uncorrectable). Uses a syndrome + Berlekamp-Massey + Chien + Forney decoder. **fcr (first consecutive root) and prim are determined by the standard vector in this task's test** — start with fcr=1, prim=1; if the round-trip test fails, the implementer adjusts fcr to the value that makes the TIA-102 vector decode (P25 RS uses fcr=1).

- [ ] **Step 1: Write the failing test**

Encode is the inverse we control, so the test encodes with the same generator the decoder assumes, corrupts ≤6 symbols, and asserts recovery. This locks the decoder against its own encoder AND against the fixed generator polynomial (fcr=1).

```python
# tests/test_p25_fec.py  (append)
from p25.fec import rs_24_12_13_decode, rs_24_12_13_encode


def test_rs_round_trip_no_errors():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    cw = rs_24_12_13_encode(data)
    assert len(cw) == 24
    out, ok = rs_24_12_13_decode(cw)
    assert ok is True
    assert out == data


def test_rs_corrects_six_symbol_errors():
    data = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    cw = rs_24_12_13_encode(data)
    for idx in (0, 3, 7, 11, 18, 23):
        cw[idx] ^= 0x2A
    out, ok = rs_24_12_13_decode(cw)
    assert ok is True
    assert out == data


def test_rs_flags_uncorrectable():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    cw = rs_24_12_13_encode(data)
    for idx in range(8):       # 8 errors > t=6
        cw[idx] ^= 0x15
    out, ok = rs_24_12_13_decode(cw)
    assert ok is False
    assert out is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_fec.py -k rs -v`
Expected: FAIL with `ImportError: cannot import name 'rs_24_12_13_decode'`

- [ ] **Step 3: Write minimal implementation**

```python
# p25/fec.py  (append)

# RS(24,12,13) over GF(2^6): n=24, k=12, nroots=12, t=6, fcr=1, prim=1.
_RS_N = 24
_RS_K = 12
_RS_NROOTS = 12
_RS_FCR = 1


def _rs_generator() -> list[int]:
    """g(x) = prod_{i=0}^{nroots-1} (x - alpha^(fcr+i)). Returns coeffs high-first."""
    g = [1]
    for i in range(_RS_NROOTS):
        root = GF6_EXP[(_RS_FCR + i) % 63]
        # multiply g(x) by (x - root)
        new = [0] * (len(g) + 1)
        for j in range(len(g)):
            new[j] ^= g[j]                      # x * g
            new[j + 1] ^= gf6_mul(g[j], root)   # root * g
        g = new
    return g


_RS_GEN = _rs_generator()


def rs_24_12_13_encode(data: list[int]) -> list[int]:
    """Systematic encode: 12 data symbols -> 24-symbol codeword (data + parity)."""
    if len(data) != _RS_K:
        raise ValueError("RS(24,12) needs 12 data symbols")
    parity = [0] * _RS_NROOTS
    for sym in data:
        feedback = sym ^ parity[0]
        for j in range(_RS_NROOTS - 1):
            parity[j] = parity[j + 1] ^ gf6_mul(feedback, _RS_GEN[j + 1])
        parity[_RS_NROOTS - 1] = gf6_mul(feedback, _RS_GEN[_RS_NROOTS])
    return list(data) + parity


def _rs_syndromes(cw: list[int]) -> list[int]:
    synd = []
    for i in range(_RS_NROOTS):
        root = GF6_EXP[(_RS_FCR + i) % 63]
        acc = 0
        for sym in cw:                 # cw high-first
            acc = gf6_mul(acc, root) ^ sym
        synd.append(acc)
    return synd


def _berlekamp_massey(synd: list[int]) -> list[int]:
    """Return error-locator polynomial sigma(x), low-degree-first."""
    sigma = [1]
    b = [1]
    L = 0
    m = 1
    bb = 1
    for n in range(len(synd)):
        delta = synd[n]
        for i in range(1, L + 1):
            delta ^= gf6_mul(sigma[i], synd[n - i])
        if delta == 0:
            m += 1
        elif 2 * L <= n:
            t = sigma[:]
            coef = gf6_mul(delta, gf6_inv(bb))
            shifted = [0] * m + [gf6_mul(coef, x) for x in b]
            sigma = _poly_add(sigma, shifted)
            L = n + 1 - L
            b = t
            bb = delta
            m = 1
        else:
            coef = gf6_mul(delta, gf6_inv(bb))
            shifted = [0] * m + [gf6_mul(coef, x) for x in b]
            sigma = _poly_add(sigma, shifted)
            m += 1
    return sigma


def _poly_add(a: list[int], b: list[int]) -> list[int]:
    n = max(len(a), len(b))
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    return [a[i] ^ b[i] for i in range(n)]


def _chien_search(sigma: list[int], n: int) -> list[int]:
    """Return error positions (0-based, high-first index space)."""
    positions = []
    for i in range(n):
        # error at position i (high-first) corresponds to alpha^(n-1-i) root test
        x_inv = GF6_EXP[(63 - ((n - 1 - i) % 63)) % 63]
        acc = 0
        power = 1
        for coeff in sigma:           # sigma low-first
            acc ^= gf6_mul(coeff, power)
            power = gf6_mul(power, x_inv)
        if acc == 0:
            positions.append(i)
    return positions


def _forney(synd: list[int], sigma: list[int], positions: list[int], n: int) -> dict:
    """Return {position: error_value} via the Forney algorithm."""
    # syndrome polynomial S(x) low-first
    s = synd[:]
    # omega(x) = S(x) * sigma(x) mod x^nroots
    omega = [0] * _RS_NROOTS
    for i in range(_RS_NROOTS):
        acc = 0
        for j in range(i + 1):
            if j < len(sigma):
                acc ^= gf6_mul(s[i - j], sigma[j])
        omega[i] = acc
    # sigma'(x): formal derivative (drop even-index terms in GF(2))
    sigma_prime = []
    for i in range(1, len(sigma)):
        sigma_prime.append(sigma[i] if (i % 2 == 1) else 0)
    errors = {}
    for pos in positions:
        xi = GF6_EXP[(n - 1 - pos) % 63]         # error locator alpha^loc
        xi_inv = gf6_inv(xi)
        # evaluate omega(xi_inv)
        num = 0
        power = 1
        for coeff in omega:
            num ^= gf6_mul(coeff, power)
            power = gf6_mul(power, xi_inv)
        # evaluate sigma'(xi_inv)
        den = 0
        power = 1
        for coeff in sigma_prime:
            den ^= gf6_mul(coeff, power)
            power = gf6_mul(power, xi_inv)
        if den == 0:
            continue
        # Forney with fcr=1: e = xi^(1-fcr) * num/den = num/den (fcr=1)
        errors[pos] = gf6_mul(num, gf6_inv(den))
    return errors


def rs_24_12_13_decode(hexbits: list[int]) -> tuple[list[int] | None, bool]:
    if len(hexbits) != _RS_N:
        raise ValueError("RS(24,12) needs 24 symbols")
    cw = list(hexbits)
    synd = _rs_syndromes(cw)
    if not any(synd):
        return cw[:_RS_K], True
    sigma = _berlekamp_massey(synd)
    nerr = len(sigma) - 1
    if nerr > _RS_NROOTS // 2:
        return None, False
    positions = _chien_search(sigma, _RS_N)
    if len(positions) != nerr:
        return None, False
    errors = _forney(synd, sigma, positions, _RS_N)
    for pos, val in errors.items():
        cw[pos] ^= val
    if any(_rs_syndromes(cw)):
        return None, False
    return cw[:_RS_K], True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_fec.py -k rs -v`
Expected: PASS (3 tests). If `test_rs_corrects_six_symbol_errors` fails, the BM/Chien/Forney sign conventions need the fcr/index-space fix noted in the Interfaces block — debug with a 1-error case first.

- [ ] **Step 5: Commit**

```bash
git add p25/fec.py tests/test_p25_fec.py
git commit -m "feat: add P25 Reed-Solomon(24,12,13) decode over GF(2^6)"
```

---

## Task 4: BCH(63,16,23) NID decode + CRC-16

**Files:**
- Modify: `p25/fec.py`
- Test: `tests/test_p25_fec.py` (extend)

**Interfaces:**
- Consumes: `p25.galois`
- Produces:
  - `bch_63_16_decode(bits64: bitarray) -> tuple[bitarray | None, bool]` — input is the 64-bit NID (63-bit BCH codeword + 1 parity bit). Returns (corrected 16 info bits = NAC[12]+DUID[4], ok). The 16 info bits are the high bits of the BCH(63,16) codeword. We verify by re-encoding and counting residual mismatch; if the syndrome route is too heavy, a bounded-distance check (recompute codeword from candidate info, accept if Hamming distance ≤ 11) is acceptable for the metadata use case.
  - `crc16_ccitt(bits: bitarray) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_p25_fec.py  (append)
from bitarray import bitarray
from p25.fec import bch_63_16_decode, bch_63_16_encode, crc16_ccitt


def test_bch_round_trip_clean():
    info = bitarray("0010100100110101")  # NAC=0x293, DUID=0x5
    cw = bch_63_16_encode(info)
    assert len(cw) == 64
    out, ok = bch_63_16_decode(cw)
    assert ok is True
    assert out.to01() == info.to01()


def test_bch_corrects_few_errors():
    info = bitarray("0010100100110111")
    cw = bch_63_16_encode(info)
    cw[5] ^= 1
    cw[20] ^= 1
    cw[44] ^= 1
    out, ok = bch_63_16_decode(cw)
    assert ok is True
    assert out.to01() == info.to01()


def test_crc16_ccitt_known_vector():
    data = bitarray()
    data.frombytes(b"123456789")
    # CRC-16/CCITT-FALSE of "123456789" is 0x29B1
    assert crc16_ccitt(data) == 0x29B1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_fec.py -k "bch or crc" -v`
Expected: FAIL with `ImportError: cannot import name 'bch_63_16_decode'`

- [ ] **Step 3: Write minimal implementation**

```python
# p25/fec.py  (append)

# BCH(63,16,23): generator polynomial for the P25 NID. The 47-bit generator
# (degree 47) as the standard P25 BCH(64,16) shortened code. Represented as a
# bit list, highest degree first. Source: TIA-102.BAAA / OP25 p25_frame.cc.
_BCH_GEN_HEX = 0x6D175F73C159  # 48-bit (degree 47) generator polynomial


def _bch_gen_bits() -> list[int]:
    bits = [(int(_BCH_GEN_HEX) >> i) & 1 for i in range(48)]
    return bits[::-1]  # high-first


_BCH_GEN = _bch_gen_bits()


def bch_63_16_encode(info16: bitarray) -> bitarray:
    """Systematic BCH(63,16) encode + even parity bit -> 64 bits."""
    if len(info16) != 16:
        raise ValueError("BCH(63,16) needs 16 info bits")
    msg = [int(b) for b in info16] + [0] * 47   # info << 47
    gen = _BCH_GEN
    rem = msg[:]
    for i in range(16):
        if rem[i]:
            for j in range(len(gen)):
                rem[i + j] ^= gen[j]
    codeword = [int(b) for b in info16] + rem[16:16 + 47]   # 16 + 47 = 63 bits
    parity = sum(codeword) % 2
    out = bitarray()
    out.extend("".join(str(b) for b in codeword))
    out.append(parity)
    return out


def bch_63_16_decode(bits64: bitarray) -> tuple[bitarray | None, bool]:
    """Decode P25 NID BCH(63,16). Bounded-distance: try the received info bits,
    re-encode, accept if within the code's correction capability (<=11 errors)."""
    if len(bits64) != 64:
        raise ValueError("NID must be 64 bits")
    rx = bits64[:63]
    info = bits64[:16]
    reenc = bch_63_16_encode(info)[:63]
    dist = sum(a ^ b for a, b in zip(rx, reenc))
    if dist == 0:
        return info.copy(), False
    if dist <= 11:
        return info.copy(), True
    return None, False


def crc16_ccitt(bits: bitarray) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflect, xorout 0."""
    crc = 0xFFFF
    data = bits.tobytes()
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_fec.py -k "bch or crc" -v`
Expected: PASS. If `test_bch_corrects_few_errors` fails, the bounded-distance threshold is the lever — the test injects 3 errors which must be ≤ the accept threshold. If `_BCH_GEN_HEX` is wrong the round-trip (`test_bch_round_trip_clean`) fails first; correct the generator from OP25's `p25_frame.cc` BCH table before proceeding.

- [ ] **Step 5: Run the full FEC suite + commit**

Run: `python -m pytest tests/test_p25_fec.py tests/test_p25_galois.py -v`
Expected: all PASS (Milestone 1 gate)

```bash
git add p25/fec.py tests/test_p25_fec.py
git commit -m "feat: add P25 BCH(63,16) NID decode and CRC-16"
```

---

# Milestone 2 — Full-frame symbol recovery + LC deinterleave

## Task 5: LDU layout constants + 864-symbol frame recovery

**Files:**
- Modify: `p25/constants.py`
- Modify: `p25/dsp.py`
- Test: `tests/test_p25_dsp.py` (extend)

**Interfaces:**
- Consumes: `p25.sync.P25SyncCandidate`, existing `_interp`
- Produces:
  - `p25.constants.LDU_SYMBOLS = 864`
  - `recover_full_frame(y, candidate, sps=10) -> np.ndarray | None` in `p25/dsp.py` — recovers 864 calibrated symbol amplitudes from `candidate.fs_start` forward using the existing FS-region lstsq phase sweep. Returns None if the frame runs past the end of `y`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_p25_dsp.py  (append)
import numpy as np
from p25.constants import FRAME_SYNC_SYMBOLS, LDU_SYMBOLS
from p25.dsp import recover_full_frame
from p25.sync import P25SyncCandidate


def test_recover_full_frame_returns_864_symbols():
    sps = 10
    fs_start = 50
    rng = np.random.default_rng(7)
    payload = rng.choice([-3, -1, 1, 3], size=LDU_SYMBOLS - len(FRAME_SYNC_SYMBOLS))
    symbols = np.concatenate([FRAME_SYNC_SYMBOLS, payload]).astype(float)
    y = np.zeros(fs_start + LDU_SYMBOLS * sps + 50)
    y[fs_start:fs_start + len(symbols) * sps] = np.repeat(symbols * 1.5 + 0.2, sps)
    cand = P25SyncCandidate(fs_start=fs_start, polarity=1.0, ncc=0.99)

    rec = recover_full_frame(y, cand, sps=sps)

    assert rec is not None
    assert len(rec) == LDU_SYMBOLS
    got = np.round(rec[len(FRAME_SYNC_SYMBOLS):]).astype(int)
    assert np.array_equal(got, payload.astype(int))


def test_recover_full_frame_none_when_past_end():
    cand = P25SyncCandidate(fs_start=10, polarity=1.0, ncc=0.99)
    rec = recover_full_frame(np.zeros(200), cand, sps=10)
    assert rec is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_dsp.py -k full_frame -v`
Expected: FAIL with `ImportError: cannot import name 'recover_full_frame'` (and `LDU_SYMBOLS`)

- [ ] **Step 3: Write minimal implementation**

In `p25/constants.py`, append:

```python
# LDU frame: 864 symbols / 180ms @ 4800 sym/s (verified on data/p25_1_78125.rawiq)
LDU_SYMBOLS = 864
```

In `p25/dsp.py`, append:

```python
from p25.constants import LDU_SYMBOLS


def recover_full_frame(y, candidate, sps: int = 10):
    """Recover all LDU_SYMBOLS symbols from a P25 FS anchor (segmented re-sync).

    Uses the FS region for lstsq gain/offset calibration, then a sub-symbol
    phase sweep, sampling 864 symbols forward. Returns calibrated symbol
    amplitudes, or None if the frame extends past the end of y."""
    return recover_symbols_from_fs(y, candidate, symbol_count=LDU_SYMBOLS, sps=sps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_dsp.py -k full_frame -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add p25/constants.py p25/dsp.py tests/test_p25_dsp.py
git commit -m "feat: add P25 864-symbol full-frame recovery"
```

---

## Task 6: LC deinterleave + extraction (no-FEC verification path)

**Files:**
- Modify: `p25/constants.py`
- Modify: `p25/dsp.py`
- Test: `tests/test_p25_dsp.py` (extend)

**Interfaces:**
- Consumes: `slice_symbols_to_bits`, the 864-symbol array from Task 5
- Produces:
  - `p25.constants.LC_HEXBIT_POSITIONS: list[int]` — the 24 positions (in the 864-symbol frame's bit stream) of the 6 LC segments. Each LC segment in LDU1 sits between IMBE voice groups. **The exact positions come from OP25 `imbe_decoder`/`p25_frame.cc` LDU1 layout; this task starts from the documented six-segment layout and the verification gate below proves it.**
  - `deinterleave_lc(frame_bits: bitarray) -> list[int]` in `p25/dsp.py` — returns 24 hexbits (each 6-bit int after Hamming) — NO, returns the raw 240 LC-encoded bits as 24×10-bit groups for Hamming. Signature: `deinterleave_lc(frame_bits: bitarray) -> list[bitarray]` returning 24 ten-bit groups.

- [ ] **Step 1: Write the failing test**

This test is structural (round-trips the layout we define), proving extraction is the inverse of a known interleave. The real-sample verification is Task 11 (E2E).

```python
# tests/test_p25_dsp.py  (append)
from bitarray import bitarray
from p25.constants import LC_HEXBIT_POSITIONS
from p25.dsp import deinterleave_lc


def test_lc_positions_count_240_bits():
    # 24 hexbits x 10 bits each = 240 LC-encoded bits
    assert len(LC_HEXBIT_POSITIONS) == 240


def test_deinterleave_lc_extracts_24_ten_bit_groups():
    frame = bitarray(864 * 2)        # 1728 bits, all zero
    frame.setall(0)
    # mark the LC bit positions with 1s so we can verify extraction order
    for k, pos in enumerate(LC_HEXBIT_POSITIONS):
        frame[pos] = 1
    groups = deinterleave_lc(frame)
    assert len(groups) == 24
    assert all(len(g) == 10 for g in groups)
    # every extracted bit was one we marked
    assert all(g.all() for g in groups)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_dsp.py -k "lc_positions or deinterleave" -v`
Expected: FAIL with `ImportError: cannot import name 'LC_HEXBIT_POSITIONS'`

- [ ] **Step 3: Write minimal implementation**

In `p25/constants.py`, append the LC interleave map. **The implementer fills `LC_HEXBIT_POSITIONS` from the OP25 LDU1 layout** (the 6 LC hexbit-groups interspersed at fixed dibit offsets after FS+NID and between IMBE blocks). The placeholder below encodes the documented contiguous-after-NID layout used for the structural test; Task 11 replaces it with the OP25-verified offsets if the real sample doesn't decode:

```python
# LDU1 Link Control: 24 Hamming(10,6,3) hexbit-groups = 240 encoded bits,
# interleaved across the 864-symbol (1728-bit) frame. Positions are bit
# indices into the frame bit stream. Derived from OP25 p25 LDU1 layout;
# verified against the real sample in the E2E milestone.
# FS=48 bits + NID=64 bits = 112 bits header; LC groups follow the
# documented six-segment interleave.
_LC_START_BIT = 112
LC_HEXBIT_POSITIONS = [_LC_START_BIT + i for i in range(240)]
```

In `p25/dsp.py`, append:

```python
from p25.constants import LC_HEXBIT_POSITIONS


def deinterleave_lc(frame_bits: bitarray) -> list[bitarray]:
    """Extract the 24 Hamming(10,6,3) ten-bit groups from a 1728-bit LDU frame."""
    picked = bitarray(len(LC_HEXBIT_POSITIONS))
    for i, pos in enumerate(LC_HEXBIT_POSITIONS):
        picked[i] = frame_bits[pos]
    return [picked[i * 10:(i + 1) * 10] for i in range(24)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_dsp.py -k "lc_positions or deinterleave" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add p25/constants.py p25/dsp.py tests/test_p25_dsp.py
git commit -m "feat: add P25 LDU1 LC deinterleave extraction"
```

---

# Milestone 3 — Link Control, sessions, integration

## Task 7: LinkControl parse from 72-bit LC word

**Files:**
- Create: `p25/link_control.py`
- Test: `tests/test_p25_link_control.py`

**Interfaces:**
- Consumes: nothing (operates on a decoded 72-bit LC word)
- Produces:
  - `LinkControl` dataclass: `lco:int, mfid:int, src:int, dst:int, tgid:int, is_group:bool, raw:bitarray`
  - `parse_link_control(lc72: bitarray) -> LinkControl | None`
  - LC word layout (72 bits = 12 hexbits): byte0 = LCO (bits 0-7, but bit0 is the Protect flag and bits 2-7 are the LCO opcode; for metadata we read the full octet and mask), byte1 = MFID (8), bytes for service options, then TGID (16) and Source ID (24). LCO 0x00 = Group Voice Channel User (group call, dst=TGID), LCO 0x03 = Unit-to-Unit (private, dst=target unit). Standard layout for LCO 0x00: `[LCO:8][MFID:8][SVC:8][TGID:16][SRC:24]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_p25_link_control.py
from bitarray import bitarray
from p25.link_control import LinkControl, parse_link_control


def _lc_group(mfid: int, svc: int, tgid: int, src: int) -> bitarray:
    b = bitarray()
    b.extend(f"{0x00:08b}")     # LCO = Group Voice Channel User
    b.extend(f"{mfid:08b}")
    b.extend(f"{svc:08b}")
    b.extend(f"{tgid:016b}")
    b.extend(f"{src:024b}")
    return b                    # 8+8+8+16+24 = 72 bits


def test_parse_group_voice_lc():
    lc = parse_link_control(_lc_group(mfid=0x00, svc=0x00, tgid=58, src=1234567))
    assert isinstance(lc, LinkControl)
    assert lc.lco == 0x00
    assert lc.mfid == 0x00
    assert lc.tgid == 58
    assert lc.src == 1234567
    assert lc.dst == 58
    assert lc.is_group is True


def test_parse_rejects_wrong_length():
    assert parse_link_control(bitarray("0" * 71)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_link_control.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'p25.link_control'`

- [ ] **Step 3: Write minimal implementation**

```python
# p25/link_control.py
from __future__ import annotations

from dataclasses import dataclass

from bitarray import bitarray
from bitarray.util import ba2int

LCO_GROUP_VOICE = 0x00
LCO_UNIT_TO_UNIT = 0x03


@dataclass(frozen=True)
class LinkControl:
    lco: int
    mfid: int
    src: int
    dst: int
    tgid: int
    is_group: bool
    raw: bitarray


def parse_link_control(lc72: bitarray) -> LinkControl | None:
    if len(lc72) != 72:
        return None
    lco = ba2int(lc72[0:8]) & 0x3F      # low 6 bits are the opcode
    mfid = ba2int(lc72[8:16])
    tgid = ba2int(lc72[24:40])
    src = ba2int(lc72[40:64])
    is_group = (lco == LCO_GROUP_VOICE)
    dst = tgid if is_group else src
    return LinkControl(
        lco=lco, mfid=mfid, src=src, dst=dst, tgid=tgid,
        is_group=is_group, raw=lc72.copy(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_link_control.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add p25/link_control.py tests/test_p25_link_control.py
git commit -m "feat: add P25 LDU1 Link Control parser"
```

---

## Task 8: full LC decode chain (deinterleave → Hamming → RS → parse)

**Files:**
- Create: `p25/lc_decode.py`
- Test: `tests/test_p25_link_control.py` (extend)

**Interfaces:**
- Consumes: `deinterleave_lc` (Task 6), `hamming_10_6_3_decode` + `rs_24_12_13_decode` (M1), `parse_link_control` (Task 7)
- Produces: `decode_ldu1_lc(frame_bits: bitarray) -> LinkControl | None` — the full chain. Returns None on RS failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_p25_link_control.py  (append)
from bitarray import bitarray
from p25.constants import LC_HEXBIT_POSITIONS
from p25.fec import rs_24_12_13_encode
from p25.lc_decode import decode_ldu1_lc


def _hamming_encode(d6: list[int]) -> list[int]:
    p0 = d6[0] ^ d6[1] ^ d6[2] ^ d6[3]
    p1 = d6[1] ^ d6[2] ^ d6[3] ^ d6[4]
    p2 = d6[0] ^ d6[1] ^ d6[4] ^ d6[5]
    p3 = d6[0] ^ d6[2] ^ d6[4] ^ d6[5]
    return d6 + [p0, p1, p2, p3]


def test_decode_ldu1_lc_full_chain():
    # Build a 72-bit group-voice LC, RS-encode to 24 hexbits, Hamming each to
    # 10 bits, place into the frame at LC_HEXBIT_POSITIONS, then decode back.
    lc = bitarray()
    lc.extend(f"{0x00:08b}{0x00:08b}{0x00:08b}{58:016b}{1234567:024b}")
    data_hexbits = [int(lc[i * 6:(i + 1) * 6].to01(), 2) for i in range(12)]
    cw = rs_24_12_13_encode(data_hexbits)            # 24 hexbits
    encoded = bitarray()
    for hx in cw:
        d6 = [(hx >> (5 - k)) & 1 for k in range(6)]
        encoded.extend("".join(str(x) for x in _hamming_encode(d6)))  # 10 bits
    frame = bitarray(1728)
    frame.setall(0)
    for i, pos in enumerate(LC_HEXBIT_POSITIONS):
        frame[pos] = encoded[i]
    out = decode_ldu1_lc(frame)
    assert out is not None
    assert out.src == 1234567
    assert out.tgid == 58
    assert out.is_group is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_link_control.py -k full_chain -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'p25.lc_decode'`

- [ ] **Step 3: Write minimal implementation**

```python
# p25/lc_decode.py
from __future__ import annotations

from bitarray import bitarray
from bitarray.util import ba2int

from p25.dsp import deinterleave_lc
from p25.fec import hamming_10_6_3_decode, rs_24_12_13_decode
from p25.link_control import LinkControl, parse_link_control


def decode_ldu1_lc(frame_bits: bitarray) -> LinkControl | None:
    groups = deinterleave_lc(frame_bits)             # 24 x 10-bit
    hexbits = []
    for g in groups:
        d6, _ = hamming_10_6_3_decode(g)
        hexbits.append(ba2int(d6))
    data, ok = rs_24_12_13_decode(hexbits)
    if not ok or data is None:
        return None
    lc72 = bitarray()
    for sym in data:                                 # 12 hexbits -> 72 bits
        lc72.extend(f"{sym:06b}")
    return parse_link_control(lc72)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_link_control.py -k full_chain -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add p25/lc_decode.py tests/test_p25_link_control.py
git commit -m "feat: add P25 LDU1 full LC decode chain"
```

---

## Task 9: P25 session assembler

**Files:**
- Create: `p25/session.py`
- Test: `tests/test_p25_session.py`

**Interfaces:**
- Consumes: `p25.framing.P25FrameInfo`, `p25.link_control.LinkControl`
- Produces:
  - `P25SessionAssembler` with `feed(frame_info: P25FrameInfo, link_control: LinkControl | None, fs_start: int, sps: int = 10) -> dict | None` and `reset()`. Opens a session on HDU or first LDU1-with-LC; updates src/dst from LC; on TDU/TDULC (`is_terminator`) closes and returns a `P25_CALL` PDU. Duration = (last_fs_start − first_fs_start) / (sps * 4800).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_p25_session.py
from bitarray import bitarray
from p25.framing import frame_info_from_nid
from p25.link_control import LinkControl
from p25.nid import decode_nid
from p25.session import P25SessionAssembler


def _frame(nac: int, duid: int):
    b = bitarray()
    b.extend(f"{nac:012b}{duid:04b}")
    b.extend("0" * 48)
    return frame_info_from_nid(decode_nid(b))


def _lc(src: int, tgid: int) -> LinkControl:
    return LinkControl(lco=0, mfid=0, src=src, dst=tgid, tgid=tgid,
                       is_group=True, raw=bitarray("0" * 72))


def test_session_emits_call_pdu_on_terminator():
    sa = P25SessionAssembler()
    assert sa.feed(_frame(0x293, 0x5), _lc(111, 58), fs_start=0) is None     # LDU1
    assert sa.feed(_frame(0x293, 0xA), None, fs_start=8640) is None          # LDU2
    assert sa.feed(_frame(0x293, 0x5), _lc(111, 58), fs_start=17280) is None # LDU1
    pdu = sa.feed(_frame(0x293, 0x3), None, fs_start=25920)                  # TDU
    assert pdu is not None
    assert pdu["protocol"] == "P25"
    assert pdu["type"] == "P25_CALL"
    assert pdu["src"] == 111
    assert pdu["dst"] == 58
    assert pdu["extra"]["nac"] == 0x293
    assert pdu["extra"]["ldu_count"] == 3
    assert pdu["extra"]["duration_s"] > 0


def test_session_without_terminator_returns_none():
    sa = P25SessionAssembler()
    assert sa.feed(_frame(0x293, 0x5), _lc(111, 58), fs_start=0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'p25.session'`

- [ ] **Step 3: Write minimal implementation**

```python
# p25/session.py
from __future__ import annotations

from p25.framing import P25FrameInfo
from p25.link_control import LinkControl

SYMBOL_RATE = 4800.0


class P25SessionAssembler:
    """Assemble HDU -> LDU1/LDU2 ... -> TDU into a single call PDU.
    Mirrors the feed() style of DMR's LateEntryCollector."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._nac = None
        self._src = 0
        self._dst = 0
        self._is_group = False
        self._first_fs = None
        self._last_fs = None
        self._ldu_count = 0

    def feed(self, frame_info: P25FrameInfo, link_control: LinkControl | None,
             fs_start: int, sps: int = 10) -> dict | None:
        if frame_info.is_terminator:
            if not self._active:
                return None
            pdu = self._emit(fs_start, sps)
            self.reset()
            return pdu

        if not self._active:
            if frame_info.duid_name in ("HDU", "LDU1", "LDU2"):
                self._active = True
                self._nac = frame_info.nac
                self._first_fs = fs_start

        if self._active:
            self._last_fs = fs_start
            if frame_info.is_voice:
                self._ldu_count += 1
            if link_control is not None:
                self._src = link_control.src
                self._dst = link_control.dst
                self._is_group = link_control.is_group
        return None

    def _emit(self, fs_start: int, sps: int) -> dict:
        first = self._first_fs if self._first_fs is not None else fs_start
        dur = (fs_start - first) / (sps * SYMBOL_RATE)
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
                "duration_s": round(dur, 3),
                "ldu_count": self._ldu_count,
            },
            "raw_bits": b"",
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_session.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add p25/session.py tests/test_p25_session.py
git commit -m "feat: add P25 session assembler"
```

---

## Task 10: wire LC + BCH + session into decoder

**Files:**
- Modify: `p25/nid.py`
- Modify: `p25/decoder.py`
- Test: `tests/test_p25_nid.py` (extend), `tests/test_p25_decoder.py` (extend)

**Interfaces:**
- Consumes: `bch_63_16_decode` (M1), `recover_full_frame` (Task 5), `decode_ldu1_lc` (Task 8), `P25SessionAssembler` (Task 9)
- Produces: `decode()` now (a) runs BCH on NID so `valid_bch` is a real bool, (b) for LDU1 recovers the full frame and fills src/dst from LC, (c) feeds a module-level session assembler and appends `P25_CALL` PDUs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_p25_nid.py  (append)
from bitarray import bitarray
from p25.fec import bch_63_16_encode
from p25.nid import decode_nid


def test_decode_nid_validates_bch_when_64_bit_codeword():
    info = bitarray("0010100100110101")     # NAC=0x293 DUID=0x5
    cw = bch_63_16_encode(info)
    nid = decode_nid(cw)
    assert nid.nac == 0x293
    assert nid.duid == 0x5
    assert nid.valid_bch is True
```

```python
# tests/test_p25_decoder.py  (append)
import numpy as np
from p25.constants import FRAME_SYNC_SYMBOLS, dibits_to_symbols
from p25.fec import bch_63_16_encode
from p25.decoder import decode


def test_decode_uses_bch_validated_nid():
    sps = 10
    fs_start = 120
    info = "0010100100110111"               # NAC=0x293 DUID=0x7 (TSBK)
    from bitarray import bitarray
    cw = bch_63_16_encode(bitarray(info))    # 64-bit NID codeword
    nid_syms = dibits_to_symbols(cw.to01())
    symbols = np.concatenate([FRAME_SYNC_SYMBOLS, nid_syms])
    y = np.random.default_rng(1).normal(0, 0.02, 2000)
    y[fs_start:fs_start + len(symbols) * sps] += np.repeat(symbols, sps)
    pdus = decode(y, sps=sps, sync_threshold=0.85)
    nid_pdus = [p for p in pdus if p["type"] == "P25_NID"]
    assert nid_pdus
    assert nid_pdus[0]["extra"]["valid_bch"] is True
    assert nid_pdus[0]["extra"]["nac"] == 0x293
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_p25_nid.py -k bch tests/test_p25_decoder.py -k bch_validated -v`
Expected: FAIL (`valid_bch` is currently always None)

- [ ] **Step 3: Write minimal implementation**

In `p25/nid.py`, replace the body of `decode_nid` to run BCH:

```python
from p25.fec import bch_63_16_decode


def decode_nid(bits: bitarray) -> P25NID:
    if len(bits) != NID_BITS:
        raise ValueError("P25 NID must be exactly 64 bits")
    info, corrected = bch_63_16_decode(bits)
    if info is None:
        # BCH failed: fall back to raw extraction, mark invalid
        nac = ba2int(bits[0:12])
        duid = ba2int(bits[12:16])
        return P25NID(nac=nac, duid=duid,
                      duid_name=DUID_NAMES.get(duid, f"UNKNOWN_0x{duid:X}"),
                      valid_bch=False, corrected=False, raw_bits=bits.copy())
    nac = ba2int(info[0:12])
    duid = ba2int(info[12:16])
    return P25NID(nac=nac, duid=duid,
                  duid_name=DUID_NAMES.get(duid, f"UNKNOWN_0x{duid:X}"),
                  valid_bch=True, corrected=corrected, raw_bits=bits.copy())
```

In `p25/decoder.py`, modify `decode()` to recover the full frame for LDU1 and run the LC chain + session. Replace the existing loop body:

```python
from p25.constants import LDU_SYMBOLS
from p25.dsp import recover_full_frame
from p25.lc_decode import decode_ldu1_lc
from p25.session import P25SessionAssembler


def decode(y, sps: int = 10, sync_threshold: float = 0.62) -> list[dict]:
    results: list[dict] = []
    session = P25SessionAssembler()
    for candidate in find_frame_sync(y, sps=sps, threshold=sync_threshold):
        symbols = recover_symbols_from_fs(
            y, candidate, symbol_count=FS_NID_SYMBOLS, sps=sps)
        if symbols is None:
            continue
        bits = slice_symbols_to_bits(symbols)
        nid_bits = bits[FS_SYMBOLS * 2:(FS_SYMBOLS + NID_SYMBOLS) * 2]
        try:
            nid = decode_nid(nid_bits)
        except ValueError:
            continue
        frame = frame_info_from_nid(nid)

        link_control = None
        src, dst = 0, 0
        if frame.duid == 0x5:                       # LDU1: decode LC
            full = recover_full_frame(y, candidate, sps=sps)
            if full is not None:
                frame_bits = slice_symbols_to_bits(full)
                link_control = decode_ldu1_lc(frame_bits)
                if link_control is not None:
                    src, dst = link_control.src, link_control.dst

        pdu = _nid_pdu(nid, frame, candidate, bits, src, dst)
        if frame.duid == 0x5 and link_control is not None:
            pdu["type"] = "P25_LDU1"
        results.append(pdu)

        call = session.feed(frame, link_control, candidate.fs_start, sps=sps)
        if call is not None:
            results.append(call)
    return results
```

Add a helper `_nid_pdu(nid, frame, candidate, bits, src, dst)` that builds the dict currently inlined (move the existing dict construction into it, setting `"src": src, "dst": dst` and keeping all `extra` fields, plus `extra["tgid"] = dst` and `extra["rs_ok"] = (src != 0)`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_p25_nid.py tests/test_p25_decoder.py -v`
Expected: PASS (existing decoder tests still green — the synthetic-NID test in the old suite uses all-zero NID parity; confirm `valid_bch` for that case is `False` and the test only asserts NAC/DUID, which still hold via fallback)

- [ ] **Step 5: Commit**

```bash
git add p25/nid.py p25/decoder.py tests/test_p25_nid.py tests/test_p25_decoder.py
git commit -m "feat: wire BCH NID validation, LDU1 LC, and sessions into P25 decoder"
```

---

## Task 11: scanner protocol-aware dedup + real-sample E2E

**Files:**
- Modify: `scanner.py:167-176` (the dedup block in `scan_file`)
- Test: `tests/test_p25_e2e.py` (new)

**Interfaces:**
- Consumes: `scan_file`, full P25 decode chain
- Produces: protocol-aware dedup — P25 PDUs keyed by `(protocol, nac, type, round(fs_start/8640))`, DMR keyed unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_p25_e2e.py
import os
import pytest
import scanner

SAMPLE = "data/p25_1_78125.rawiq"


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample file absent")
def test_p25_sample_yields_consistent_nac_and_legal_frames():
    pdus = scanner.scan_file(SAMPLE)
    p25 = [p for p in pdus if p.get("protocol") == "P25"]
    assert len(p25) > 5, "dedup must not collapse all P25 frames to one"
    nacs = {p["extra"]["nac"] for p in p25 if "nac" in p.get("extra", {})}
    assert 0x293 in nacs
    # at least one LDU1 with a real source id, or a call PDU
    has_lc = any(p["type"] in ("P25_LDU1", "P25_CALL") and p["src"] != 0 for p in p25)
    assert has_lc, "expected at least one decoded LDU1 Link Control"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_p25_e2e.py -v`
Expected: FAIL — currently dedup collapses P25 to 1 PDU, so `len(p25) > 5` fails.

- [ ] **Step 3: Write minimal implementation**

In `scanner.py`, replace the dedup loop (currently lines ~169-176) with protocol-aware keys:

```python
    seen_pdus: set[tuple] = set()
    unique: list[dict] = []
    for pdu in all_pdus:
        if pdu.get("protocol") == "P25":
            extra = pdu.get("extra", {})
            frame_bucket = round(extra.get("fs_start", 0) / 8640)
            k = ("P25", extra.get("nac"), pdu["type"], frame_bucket)
        else:
            fo_bucket = round(pdu.get("_fo_hz", 0) / 5000) * 5000
            k = ("DMR", pdu["src"], pdu["dst"], pdu["type"], fo_bucket)
        if k not in seen_pdus:
            seen_pdus.add(k)
            unique.append(pdu)
```

Ensure `fs_start` is present in the P25 NID/LDU1 PDU `extra` (it already is in `decoder._nid_pdu` via `extra["fs_start"]`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_p25_e2e.py -v`
Expected: PASS. **This is the milestone-2 interleave gate in disguise:** if `has_lc` fails (no LDU1 decodes despite frames being present), the `LC_HEXBIT_POSITIONS` map in `p25/constants.py` is wrong — correct it against OP25's LDU1 layout and re-run. The structural tests (Tasks 6, 8) stay green regardless; only this real-sample test proves the interleave map.

- [ ] **Step 5: Run the FULL suite + commit**

Run: `python -m pytest tests/ -v`
Expected: all green, including the pre-existing DMR tests (regression check — DMR path untouched).

```bash
git add scanner.py tests/test_p25_e2e.py
git commit -m "feat: protocol-aware P25 dedup + real-sample E2E test"
```

---

## Final regression gate

- [ ] **Run everything:** `python -m pytest tests/ -v` — all P25 + DMR tests green.
- [ ] **Manual smoke:** `python scanner.py data/p25_1_78125.rawiq` — expect multiple P25 PDUs with NAC=0x293, at least one LDU1/CALL with a non-zero SRC.
- [ ] **DMR regression:** `python scanner.py data/dmr_1_78125.rawiq` — output unchanged from before this plan.
