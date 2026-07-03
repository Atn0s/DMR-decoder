from __future__ import annotations

import os
import re

import numpy as np


def default_iq_scale(dtype: str | np.dtype) -> float:
    """Return the signed full-scale divisor for an interleaved IQ dtype."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return float(2 ** (8 * dtype.itemsize - 1))
    return 1.0


def read_rawiq(
    filename: str,
    dtype: str | np.dtype = np.int16,
    scale: float | None = None,
) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.dtype(dtype))
    i_data, q_data = data[0::2], data[1::2]
    n = min(len(i_data), len(q_data))
    scale = default_iq_scale(dtype) if scale is None else float(scale)
    if scale == 0:
        raise ValueError("IQ scale must be non-zero")
    return (i_data[:n] + 1j * q_data[:n]) / scale


def detect_sample_rate(path: str) -> int | None:
    """Extract sample rate from filenames like dmr_1_78125.rawiq."""
    basename = os.path.basename(path)
    match = re.search(r"_(\d{4,9})\.rawiq", basename)
    if match:
        return int(match.group(1))
    mhz_match = re.search(r"(\d+(?:\.\d+)?)\s*[mM][hH][zZ]\.rawiq", basename)
    if mhz_match:
        return int(float(mhz_match.group(1)) * 1_000_000)
    return None
