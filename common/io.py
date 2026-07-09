from __future__ import annotations

import os
import re
import wave

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
    if _is_wav_iq(filename):
        return _read_wav_iq(filename)

    data = np.fromfile(filename, dtype=np.dtype(dtype))
    i_data, q_data = data[0::2], data[1::2]
    n = min(len(i_data), len(q_data))
    scale = default_iq_scale(dtype) if scale is None else float(scale)
    if scale == 0:
        raise ValueError("IQ scale must be non-zero")
    return (i_data[:n] + 1j * q_data[:n]) / scale


def _is_wav_iq(filename: str) -> bool:
    if filename.lower().endswith((".wav", ".wave")):
        return True
    try:
        with open(filename, "rb") as f:
            header = f.read(12)
    except OSError:
        return False
    return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def _read_wav_iq(filename: str) -> np.ndarray:
    """Read stereo PCM WAV as complex IQ using channel 0=I and channel 1=Q."""
    with wave.open(filename, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        if channels < 2:
            raise ValueError("WAV IQ input must have at least two channels")
        if sample_width not in (1, 2, 4):
            raise ValueError(f"unsupported WAV IQ sample width: {sample_width} bytes")
        raw = wav.readframes(frames)

    if sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    else:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0

    samples = data.reshape(-1, channels)
    return samples[:, 0] + 1j * samples[:, 1]


def detect_sample_rate(path: str) -> int | None:
    """Extract sample rate from filenames like dmr_1_78125.rawiq."""
    basename = os.path.basename(path)
    match = re.search(r"_(\d{4,9})\.rawiq", basename)
    if match:
        return int(match.group(1))
    mhz_match = re.search(r"(\d+(?:\.\d+)?)\s*[mM][hH][zZ]\.rawiq", basename)
    if mhz_match:
        return int(float(mhz_match.group(1)) * 1_000_000)
    if _is_wav_iq(path):
        try:
            with wave.open(path, "rb") as wav:
                return int(wav.getframerate())
        except (OSError, wave.Error):
            return None
    return None
