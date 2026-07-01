from __future__ import annotations

import os
import re

import numpy as np


def read_rawiq(filename: str) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.int16)
    i_data, q_data = data[0::2], data[1::2]
    n = min(len(i_data), len(q_data))
    return (i_data[:n] + 1j * q_data[:n]) / 32768.0


def detect_sample_rate(path: str) -> int | None:
    """Extract sample rate from filenames like dmr_1_78125.rawiq."""
    match = re.search(r"_(\d{4,7})\.rawiq", os.path.basename(path))
    return int(match.group(1)) if match else None

