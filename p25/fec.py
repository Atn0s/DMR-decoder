"""P25 Phase 1 forward error correction helpers."""
from __future__ import annotations

from bitarray import bitarray
from bitarray.util import ba2int, int2ba
import numpy as np

from p25.galois import GF6_EXP, GF6_LOG, gf6_inv, gf6_mul

_HAMMING_PARITY = (
    (0, 1, 2, 5),
    (0, 1, 3, 5),
    (0, 2, 3, 4),
    (1, 2, 3, 4),
)


def _hamming_parity(d: list[int]) -> list[int]:
    return [d[a] ^ d[b] ^ d[c] ^ d[e] for (a, b, c, e) in _HAMMING_PARITY]


def hamming_10_6_3_decode(bits10: bitarray) -> tuple[bitarray, bool]:
    if len(bits10) != 10:
        raise ValueError("Hamming(10,6,3) codeword must be 10 bits")
    rx = [int(x) for x in bits10]
    data = rx[:6]
    best_dist = 11
    best_data: list[int] | None = None
    for val in range(64):
        trial = [(val >> (5 - i)) & 1 for i in range(6)]
        codeword = trial + _hamming_parity(trial)
        dist = sum(a ^ b for a, b in zip(rx, codeword))
        if dist < best_dist:
            best_dist = dist
            best_data = trial
    corrected = best_dist == 1
    if best_dist <= 1 and best_data is not None:
        data = best_data
    out = bitarray(endian="big")
    out.extend("".join(str(x) for x in data))
    return out, corrected


_RS_FULL_N = 63


def _rs_generator_index(nroots: int) -> list[int]:
    gg = [0] * (nroots + 1)
    gg[0] = 2
    gg[1] = 1
    for i in range(2, nroots + 1):
        gg[i] = 1
        for j in range(i - 1, 0, -1):
            if gg[j] != 0:
                gg[j] = gg[j - 1] ^ GF6_EXP[(GF6_LOG[gg[j]] + i) % _RS_FULL_N]
            else:
                gg[j] = gg[j - 1]
        gg[0] = GF6_EXP[(GF6_LOG[gg[0]] + i) % _RS_FULL_N]
    return [-1 if x == 0 else GF6_LOG[x] for x in gg]


_RS24_NROOTS = 12
_RS24_K = 12
_RS24_FULL_K = 51
_RS24_T = 6
_RS24_GEN_INDEX = _rs_generator_index(_RS24_NROOTS)

_RS24_16_NROOTS = 8
_RS24_16_K = 16
_RS24_16_FULL_K = 55
_RS24_16_T = 4
_RS24_16_GEN_INDEX = _rs_generator_index(_RS24_16_NROOTS)

_RS36_NROOTS = 16
_RS36_K = 20
_RS36_FULL_K = 47
_RS36_T = 8
_RS36_GEN_INDEX = _rs_generator_index(_RS36_NROOTS)


def rs_24_12_13_encode(data: list[int]) -> list[int]:
    if len(data) != _RS24_K:
        raise ValueError("RS(24,12) needs 12 data symbols")
    parity = _rs63_encode_shortened(data, _RS24_FULL_K, _RS24_NROOTS, _RS24_GEN_INDEX)
    return list(data) + parity


def rs_36_20_17_encode(data: list[int]) -> list[int]:
    if len(data) != _RS36_K:
        raise ValueError("RS(36,20) needs 20 data symbols")
    parity = _rs63_encode_shortened(data, _RS36_FULL_K, _RS36_NROOTS, _RS36_GEN_INDEX)
    return list(data) + parity


def rs_24_16_9_encode(data: list[int]) -> list[int]:
    if len(data) != _RS24_16_K:
        raise ValueError("RS(24,16) needs 16 data symbols")
    parity = _rs63_encode_shortened(
        data,
        _RS24_16_FULL_K,
        _RS24_16_NROOTS,
        _RS24_16_GEN_INDEX,
    )
    return list(data) + parity


def _rs63_encode_shortened(
    data: list[int],
    full_k: int,
    nroots: int,
    gen_index: list[int],
) -> list[int]:
    full_data = list(data) + [0] * (full_k - len(data))
    parity = [0] * nroots
    for i in range(full_k - 1, -1, -1):
        feedback = -1 if (full_data[i] ^ parity[nroots - 1]) == 0 else GF6_LOG[
            full_data[i] ^ parity[nroots - 1]
        ]
        if feedback != -1:
            for j in range(nroots - 1, 0, -1):
                if gen_index[j] != -1:
                    parity[j] = parity[j - 1] ^ GF6_EXP[(gen_index[j] + feedback) % _RS_FULL_N]
                else:
                    parity[j] = parity[j - 1]
            parity[0] = GF6_EXP[(gen_index[0] + feedback) % _RS_FULL_N]
        else:
            for j in range(nroots - 1, 0, -1):
                parity[j] = parity[j - 1]
            parity[0] = 0
    return parity


def rs_24_12_13_decode(hexbits: list[int]) -> tuple[list[int] | None, bool]:
    if len(hexbits) != 24:
        raise ValueError("RS(24,12) needs 24 symbols")
    data = list(hexbits[:_RS24_K])
    parity = list(hexbits[_RS24_K:])
    recd = parity + data + [0] * (_RS_FULL_N - 24)
    ok = _rs63_decode_in_place(recd, _RS24_NROOTS, _RS24_T)
    if not ok:
        return None, False
    return recd[12:24], True


def rs_36_20_17_decode(hexbits: list[int]) -> tuple[list[int] | None, bool]:
    if len(hexbits) != 36:
        raise ValueError("RS(36,20) needs 36 symbols")
    data = list(hexbits[:_RS36_K])
    parity = list(hexbits[_RS36_K:])
    recd = parity + data + [0] * (_RS_FULL_N - 36)
    ok = _rs63_decode_in_place(recd, _RS36_NROOTS, _RS36_T)
    if not ok:
        return None, False
    return recd[16:36], True


def rs_24_16_9_decode(hexbits: list[int]) -> tuple[list[int] | None, bool]:
    if len(hexbits) != 24:
        raise ValueError("RS(24,16) needs 24 symbols")
    data = list(hexbits[:_RS24_16_K])
    parity = list(hexbits[_RS24_16_K:])
    recd = parity + data + [0] * (_RS_FULL_N - 24)
    ok = _rs63_decode_in_place(recd, _RS24_16_NROOTS, _RS24_16_T)
    if not ok:
        return None, False
    return recd[8:24], True


def _idx(x: int) -> int:
    return -1 if x == 0 else GF6_LOG[x]


def _rs63_decode_in_place(recd_poly: list[int], nroots: int, t: int) -> bool:
    recd = [_idx(x) for x in recd_poly]
    s = [0] * (nroots + 1)
    syn_error = False
    for i in range(1, nroots + 1):
        accum = 0
        for j in range(_RS_FULL_N):
            if recd[j] != -1:
                accum ^= GF6_EXP[(recd[j] + i * j) % _RS_FULL_N]
        if accum != 0:
            syn_error = True
        s[i] = _idx(accum)
    if not syn_error:
        return True

    elp = [[0] * nroots for _ in range(nroots + 2)]
    d = [0] * (nroots + 2)
    l = [0] * (nroots + 2)
    u_lu = [0] * (nroots + 2)
    d[0] = 0
    d[1] = s[1]
    elp[0][0] = 0
    elp[1][0] = 1
    for i in range(1, nroots):
        elp[0][i] = -1
        elp[1][i] = 0
    l[0] = l[1] = 0
    u_lu[0] = -1
    u_lu[1] = 0
    u = 0
    while True:
        u += 1
        if d[u] == -1:
            l[u + 1] = l[u]
            for i in range(l[u] + 1):
                elp[u + 1][i] = elp[u][i]
                elp[u][i] = _idx(elp[u][i])
        else:
            q = u - 1
            while d[q] == -1 and q > 0:
                q -= 1
            if q > 0:
                j = q
                while j > 0:
                    j -= 1
                    if d[j] != -1 and u_lu[q] < u_lu[j]:
                        q = j
            l[u + 1] = max(l[u], l[q] + u - q)
            for i in range(nroots):
                elp[u + 1][i] = 0
            for i in range(l[q] + 1):
                if elp[q][i] != -1:
                    elp[u + 1][i + u - q] = GF6_EXP[(d[u] + _RS_FULL_N - d[q] + elp[q][i]) % _RS_FULL_N]
            for i in range(l[u] + 1):
                elp[u + 1][i] ^= elp[u][i]
                elp[u][i] = _idx(elp[u][i])
        u_lu[u + 1] = u - l[u + 1]
        if u < nroots:
            if s[u + 1] != -1:
                d[u + 1] = GF6_EXP[s[u + 1]]
            else:
                d[u + 1] = 0
            for i in range(1, l[u + 1] + 1):
                if s[u + 1 - i] != -1 and elp[u + 1][i] != 0:
                    d[u + 1] ^= GF6_EXP[(s[u + 1 - i] + _idx(elp[u + 1][i])) % _RS_FULL_N]
            d[u + 1] = _idx(d[u + 1])
        if not (u < nroots and l[u + 1] <= t):
            break

    u += 1
    if l[u] > t:
        return False
    for i in range(l[u] + 1):
        elp[u][i] = _idx(elp[u][i])

    root = [0] * t
    loc = [0] * t
    reg = [0] * (t + 1)
    for i in range(1, l[u] + 1):
        reg[i] = elp[u][i]
    count = 0
    for i in range(1, _RS_FULL_N + 1):
        q = 1
        for j in range(1, l[u] + 1):
            if reg[j] != -1:
                reg[j] = (reg[j] + j) % _RS_FULL_N
                q ^= GF6_EXP[reg[j]]
        if q == 0:
            if count >= t:
                return False
            root[count] = i
            loc[count] = _RS_FULL_N - i
            count += 1
    if count != l[u]:
        return False

    z = [0] * (t + 1)
    err = [0] * _RS_FULL_N
    for i in range(1, l[u] + 1):
        if s[i] != -1 and elp[u][i] != -1:
            z[i] = GF6_EXP[s[i]] ^ GF6_EXP[elp[u][i]]
        elif s[i] != -1:
            z[i] = GF6_EXP[s[i]]
        elif elp[u][i] != -1:
            z[i] = GF6_EXP[elp[u][i]]
        else:
            z[i] = 0
        for j in range(1, i):
            if s[j] != -1 and elp[u][i - j] != -1:
                z[i] ^= GF6_EXP[(elp[u][i - j] + s[j]) % _RS_FULL_N]
        z[i] = _idx(z[i])

    for i in range(_RS_FULL_N):
        recd_poly[i] = GF6_EXP[recd[i]] if recd[i] != -1 else 0
    for i in range(l[u]):
        err[loc[i]] = 1
        for j in range(1, l[u] + 1):
            if z[j] != -1:
                err[loc[i]] ^= GF6_EXP[(z[j] + j * root[i]) % _RS_FULL_N]
        if err[loc[i]] != 0:
            err[loc[i]] = _idx(err[loc[i]])
            q = 0
            for j in range(l[u]):
                if j != i:
                    q += _idx(1 ^ GF6_EXP[(loc[j] + root[i]) % _RS_FULL_N])
            q %= _RS_FULL_N
            err[loc[i]] = GF6_EXP[(err[loc[i]] - q + _RS_FULL_N) % _RS_FULL_N]
            recd_poly[loc[i]] ^= err[loc[i]]
    return True


_POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
_BCH_CODEWORDS: np.ndarray | None = None
_BCH_SYNDROME_MATRIX: np.ndarray | None = None
_BCH_PARITY_SOLVER: tuple[np.ndarray, list[int]] | None = None
_BCH_GF_EXP = np.array(
    [
        1, 2, 4, 8, 16, 32, 3, 6, 12, 24, 48, 35, 5, 10, 20, 40,
        19, 38, 15, 30, 60, 59, 53, 41, 17, 34, 7, 14, 28, 56, 51, 37,
        9, 18, 36, 11, 22, 44, 27, 54, 47, 29, 58, 55, 45, 25, 50, 39,
        13, 26, 52, 43, 21, 42, 23, 46, 31, 62, 63, 61, 57, 49, 33,
    ],
    dtype=np.uint8,
)


def _bch_duid_parity(info16: bitarray) -> int:
    d0 = (int(info16[12]) << 1) | int(info16[13])
    d1 = (int(info16[14]) << 1) | int(info16[15])
    return 1 if (d0, d1) in ((1, 1), (2, 2)) else 0


def _bch_syndrome_matrix() -> np.ndarray:
    global _BCH_SYNDROME_MATRIX
    if _BCH_SYNDROME_MATRIX is not None:
        return _BCH_SYNDROME_MATRIX
    h = np.zeros((22 * 6, 63), dtype=np.uint8)
    for air_pos in range(63):
        op25_pos = 62 - air_pos
        for syn in range(1, 23):
            value = int(_BCH_GF_EXP[(syn * op25_pos) % 63])
            for bit in range(6):
                h[(syn - 1) * 6 + bit, air_pos] = (value >> bit) & 1
    _BCH_SYNDROME_MATRIX = h
    return _BCH_SYNDROME_MATRIX


def _bch_parity_solver() -> tuple[np.ndarray, list[int]]:
    global _BCH_PARITY_SOLVER
    if _BCH_PARITY_SOLVER is not None:
        return _BCH_PARITY_SOLVER

    h = _bch_syndrome_matrix()
    a = h[:, 16:63].copy()
    rows, cols = a.shape
    transform = np.eye(rows, dtype=np.uint8)
    pivot_rows: list[int] = []
    r = 0
    for c in range(cols):
        candidates = np.flatnonzero(a[r:, c])
        if len(candidates) == 0:
            continue
        p = int(candidates[0] + r)
        if p != r:
            a[[r, p]] = a[[p, r]]
            transform[[r, p]] = transform[[p, r]]
        for rr in range(rows):
            if rr != r and a[rr, c]:
                a[rr] ^= a[r]
                transform[rr] ^= transform[r]
        pivot_rows.append(r)
        r += 1
        if r == cols:
            break
    if len(pivot_rows) != cols:
        raise RuntimeError("P25 BCH parity matrix is rank deficient")
    _BCH_PARITY_SOLVER = (transform, pivot_rows)
    return _BCH_PARITY_SOLVER


def bch_63_16_encode(info16: bitarray) -> bitarray:
    if len(info16) != 16:
        raise ValueError("BCH(63,16) needs 16 info bits")
    h = _bch_syndrome_matrix()
    transform, pivot_rows = _bch_parity_solver()
    data = np.array([int(b) for b in info16], dtype=np.uint8)
    rhs = (h[:, :16] @ data) & 1
    reduced_rhs = (transform @ rhs) & 1
    parity_bits = reduced_rhs[pivot_rows]
    codeword = list(data) + [int(x) for x in parity_bits]
    parity = _bch_duid_parity(info16)
    out = bitarray(endian="big")
    out.extend("".join(str(b) for b in codeword))
    out.append(parity)
    return out


def bch_63_16_decode(bits64: bitarray) -> tuple[bitarray | None, bool]:
    if len(bits64) != 64:
        raise ValueError("NID must be 64 bits")
    received = ba2int(bits64)
    codewords = _bch_codewords()
    xor = np.bitwise_xor(codewords, np.uint64(received))
    dists = _POPCOUNT8[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
    best = int(np.argmin(dists))
    dist = int(dists[best])
    if dist <= 11:
        return int2ba(best, length=16, endian="big"), dist != 0
    return None, False


def _bch_codewords() -> np.ndarray:
    global _BCH_CODEWORDS
    if _BCH_CODEWORDS is None:
        vals = [ba2int(bch_63_16_encode(int2ba(i, length=16, endian="big"))) for i in range(1 << 16)]
        _BCH_CODEWORDS = np.array(vals, dtype=np.uint64)
    return _BCH_CODEWORDS


_GOLAY_POLY = 0xAE3


def _golay_23(data12: int) -> int:
    cw = data12 & 0xFFF
    c = cw
    for _ in range(12):
        if cw & 1:
            cw ^= _GOLAY_POLY
        cw >>= 1
    return (cw << 12) | c


def golay_24_6_encode(data6: int) -> bitarray:
    data12 = (data6 & 0x3F) << 6
    codeword = _golay_23(data12)
    if codeword.bit_count() & 1:
        codeword ^= 0x800000
    out = bitarray(endian="big")
    for i in range(12, 24):
        out.append((codeword >> i) & 1)
    return out


def golay_24_6_decode(data6: bitarray, parity12: bitarray) -> tuple[int, bool]:
    if len(data6) != 6 or len(parity12) != 12:
        raise ValueError("Golay(24,6) needs 6 data bits and 12 parity bits")
    rx_data = ba2int(data6)
    rx = _golay_24_6_word(rx_data, parity12)
    best_dist = 25
    best_data = rx_data
    for data in range(64):
        cw = _golay_24_6_word(data, golay_24_6_encode(data))
        dist = (rx ^ cw).bit_count()
        if dist < best_dist:
            best_dist = dist
            best_data = data
    if best_dist <= 3:
        return best_data, best_dist != 0
    return rx_data, False


def _golay_24_6_word(data6: int, parity12: bitarray) -> int:
    word = 0
    for i in range(11, -1, -1):
        word = (word << 1) | int(parity12[i])
    word = (word << 6) | (data6 & 0x3F)
    return word << 6


def crc16_ccitt(bits: bitarray) -> int:
    crc = 0xFFFF
    for byte in bits.tobytes():
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc
