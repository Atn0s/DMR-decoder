import numpy as np
import scipy.signal as signal
from math import gcd

from core.burst_type import Fs_dec
from radio import registry
from radio.pdu import set_pdu_meta


def _decimation_factors(source_sample_rate: float, target: float = Fs_dec
                        ) -> tuple[int, int]:
    """Derive resample_poly(up, down) so source*up/down approx target.
    Reduces the ratio by gcd to keep filter length manageable."""
    up = int(round(target))
    down = int(round(source_sample_rate))
    g = gcd(up, down)
    return up // g, down // g


def decode_window(window_iq: np.ndarray, fo_hz: float, window_id: int,
                  source_sample_rate: float) -> list[dict]:
    """Decode one wideband IQ window at a given frequency offset.
    DDC(fo) -> resample to 48kHz -> frontend -> protocol registry.
    Pure function (no shared state) so it can run in a multiprocessing.Pool.
    Each returned PDU is tagged with _fo_hz and _window_id.
    Exceptions are swallowed -> returns [] so one bad window can't kill the pool."""
    try:
        n = np.arange(len(window_iq))
        shifted = window_iq * np.exp(
            -1j * 2 * np.pi * fo_hz * n / source_sample_rate
        ).astype(np.complex64)
        up, down = _decimation_factors(source_sample_rate)
        iq_dec = signal.resample_poly(shifted, up, down)
        if len(iq_dec) < 512:
            return []
        pdus = registry.decode_iq(iq_dec, sample_rate=Fs_dec)
        for pdu in pdus:
            set_pdu_meta(pdu, "fo_hz", fo_hz)
            set_pdu_meta(pdu, "window_id", window_id)
        return pdus
    except Exception:
        return []
