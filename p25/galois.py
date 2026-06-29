"""GF(2^6) arithmetic for P25 Reed-Solomon."""
from __future__ import annotations

GF6_PRIM = 0x43
GF6_SIZE = 64

GF6_EXP: list[int] = [0] * 126
GF6_LOG: list[int] = [0] * GF6_SIZE


def _build_tables() -> None:
    x = 1
    for i in range(63):
        GF6_EXP[i] = x
        GF6_LOG[x] = i
        x <<= 1
        if x & GF6_SIZE:
            x ^= GF6_PRIM
        x &= 0x3F
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
    acc = 0
    for coeff in poly:
        acc = gf6_mul(acc, x) ^ coeff
    return acc
