from __future__ import annotations

from math import gcd

import numpy as np
import scipy.signal as signal

from common.config import DEFAULT_RADIO_CONFIG, RadioConfig
import protocols
from radio.pdu import set_pdu_meta


def psd_blind_search(
    iq: np.ndarray,
    sample_rate: float,
    radio_config: RadioConfig = DEFAULT_RADIO_CONFIG,
) -> list[float]:
    """Find signal candidates in wideband IQ via Welch PSD peak detection."""
    f, psd = signal.welch(
        iq,
        fs=sample_rate,
        nperseg=radio_config.psd_nperseg,
        return_onesided=False,
    )
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-12)
    nf = np.median(psd_db)
    peaks, _ = signal.find_peaks(
        psd_db,
        height=nf + radio_config.psd_peak_threshold_db,
        distance=radio_config.psd_peak_min_distance_bins,
    )
    return [float(f[p]) for p in peaks]


def resample_factors(source_sample_rate: float, target_sample_rate: float) -> tuple[int, int]:
    up = int(round(target_sample_rate))
    down = int(round(source_sample_rate))
    g = gcd(up, down)
    return up // g, down // g


def process_candidate(
    iq: np.ndarray,
    fo: float,
    source_sample_rate: float,
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
    radio_config: RadioConfig = DEFAULT_RADIO_CONFIG,
) -> list[dict]:
    """DDC + resample + protocol decode for one wideband frequency candidate."""
    target_sample_rate = radio_config.target_sample_rate_hz
    sample_rate_tolerance = radio_config.sample_rate_tolerance_hz
    t = np.arange(len(iq)) / source_sample_rate
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * fo * t)

    if abs(source_sample_rate - target_sample_rate) < sample_rate_tolerance:
        iq_dec = iq_shifted
    else:
        up, down = resample_factors(source_sample_rate, target_sample_rate)
        iq_dec = signal.resample_poly(iq_shifted, up, down)

    results = protocols.decode_iq(
        iq_dec,
        protocol_names=protocol_names,
        sample_rate=target_sample_rate,
    )
    for pdu in results:
        set_pdu_meta(pdu, "fo_hz", fo)
    return results


def process_baseband(
    iq: np.ndarray,
    source_sample_rate: float,
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
    radio_config: RadioConfig = DEFAULT_RADIO_CONFIG,
) -> list[dict]:
    """Resample a centered baseband IQ stream if needed, then run protocol decode."""
    target_sample_rate = radio_config.target_sample_rate_hz
    sample_rate_tolerance = radio_config.sample_rate_tolerance_hz
    if abs(source_sample_rate - target_sample_rate) < sample_rate_tolerance:
        iq_dec = iq
    else:
        up, down = resample_factors(source_sample_rate, target_sample_rate)
        iq_dec = signal.resample_poly(iq, up, down)
    return protocols.decode_iq(
        iq_dec,
        protocol_names=protocol_names,
        sample_rate=target_sample_rate,
    )


def process_narrowband(
    iq: np.ndarray,
    source_sample_rate: float,
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
    radio_config: RadioConfig = DEFAULT_RADIO_CONFIG,
) -> list[dict]:
    """Compatibility wrapper for the old centered narrowband path."""
    return process_baseband(iq, source_sample_rate, protocol_names, radio_config)


def scan_iq(
    iq: np.ndarray,
    sample_rate: float,
    freq_list: list[float] | None = None,
    blind_search: bool = False,
    protocol_names: list[str] | tuple[str, ...] | set[str] | None = None,
    radio_config: RadioConfig = DEFAULT_RADIO_CONFIG,
) -> list[dict]:
    """Run the offline IQ orchestration path and return postprocessed unique PDUs."""
    if sample_rate is None:
        raise ValueError("sample_rate is required; pass --fs or use a filename with sample rate metadata")

    enabled_protocols = protocols.normalize_protocol_names(protocol_names)
    fs_in = float(sample_rate)

    if freq_list is not None:
        all_pdus = []
        for fo in freq_list:
            all_pdus.extend(
                process_candidate(
                    iq,
                    fo,
                    fs_in,
                    enabled_protocols,
                    radio_config,
                )
            )
    elif blind_search:
        all_pdus = []
        for fo in psd_blind_search(iq, fs_in, radio_config):
            all_pdus.extend(
                process_candidate(
                    iq,
                    fo,
                    fs_in,
                    enabled_protocols,
                    radio_config,
                )
            )
    else:
        all_pdus = process_baseband(
            iq,
            fs_in,
            enabled_protocols,
            radio_config,
        )

    all_pdus = protocols.postprocess_pdus(all_pdus, enabled_protocols)
    return protocols.deduplicate_pdus(all_pdus)
