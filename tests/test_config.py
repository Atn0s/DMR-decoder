from common.config import DEFAULT_RADIO_CONFIG, DEFAULT_REALTIME_CONFIG
from dmr.config import DEFAULT_DMR_CONFIG
from dmr.constants import (
    Fs_dec,
    NCC_THRESHOLD_DATA,
    NCC_THRESHOLD_VOICE,
)
from dmr.engine import BURST_STRIDE
from dpmr.config import DEFAULT_DPMR_CONFIG
from p25.config import DEFAULT_P25_CONFIG
from radio.pipeline import resample_factors
from radio import registry

import scanner


def test_radio_config_uses_dynamic_resampling_contract():
    assert DEFAULT_RADIO_CONFIG.target_sample_rate_hz == Fs_dec
    assert DEFAULT_RADIO_CONFIG.sample_rate_tolerance_hz == 1.0
    assert DEFAULT_RADIO_CONFIG.psd_peak_threshold_db == scanner.PSD_PEAK_THRESHOLD_DB
    assert DEFAULT_RADIO_CONFIG.target_sample_rate_hz == scanner.Fs_dec
    assert resample_factors(2_500_000.0, DEFAULT_RADIO_CONFIG.target_sample_rate_hz) == (12, 625)


def test_protocol_specs_expose_default_configs():
    assert registry.spec_for_protocol("dmr").config is DEFAULT_DMR_CONFIG
    assert registry.spec_for_protocol("p25").config is DEFAULT_P25_CONFIG
    assert registry.spec_for_protocol("dpmr").config is DEFAULT_DPMR_CONFIG


def test_protocol_config_defaults_keep_current_symbol_rates():
    assert DEFAULT_DMR_CONFIG.symbol_rate_hz == 4_800.0
    assert DEFAULT_P25_CONFIG.symbol_rate_hz == 4_800.0
    assert DEFAULT_DPMR_CONFIG.symbol_rate_hz == 2_400.0
    assert DEFAULT_DMR_CONFIG.target_sample_rate_hz == 48_000.0
    assert DEFAULT_P25_CONFIG.target_sample_rate_hz == 48_000.0
    assert DEFAULT_DPMR_CONFIG.target_sample_rate_hz == 48_000.0


def test_dmr_config_matches_legacy_decode_parameters():
    assert DEFAULT_DMR_CONFIG.sync_threshold_voice == NCC_THRESHOLD_VOICE
    assert DEFAULT_DMR_CONFIG.sync_threshold_data == NCC_THRESHOLD_DATA
    assert DEFAULT_DMR_CONFIG.sync_peak_distance_samples == 800
    assert DEFAULT_DMR_CONFIG.voice_burst_stride_samples == BURST_STRIDE
    assert DEFAULT_DMR_CONFIG.voice_burst_count == 6
    assert DEFAULT_DMR_CONFIG.burst_dedup_window_samples == 50
    assert DEFAULT_DMR_CONFIG.dedup_frequency_bucket_hz == 5_000.0


def test_protocol_configs_capture_frontend_and_dedup_defaults():
    assert DEFAULT_DMR_CONFIG.frontend_min_samples == 512
    assert DEFAULT_DMR_CONFIG.frontend_psd_nperseg == 4096
    assert DEFAULT_P25_CONFIG.frontend_min_samples == 512
    assert DEFAULT_P25_CONFIG.frontend_psd_nperseg == 4096
    assert DEFAULT_P25_CONFIG.sync_min_distance_symbols == 120
    assert DEFAULT_P25_CONFIG.stable_nac_min_count == 5
    assert DEFAULT_P25_CONFIG.stable_nac_min_ratio == 0.4
    assert DEFAULT_P25_CONFIG.dedup_frame_bucket_samples == 8_640
    assert DEFAULT_DPMR_CONFIG.frontend_min_samples == 512
    assert DEFAULT_DPMR_CONFIG.frontend_psd_nperseg == 4096
    assert DEFAULT_DPMR_CONFIG.nominal_deviation_hz == 1_050.0
    assert DEFAULT_DPMR_CONFIG.sync_max_symbol_errors == 0
    assert DEFAULT_DPMR_CONFIG.sync_min_distance_samples == 1_200
    assert DEFAULT_DPMR_CONFIG.sync_dedup_window_symbols == 3
    assert DEFAULT_DPMR_CONFIG.sync_error_phase_min == -12.0
    assert DEFAULT_DPMR_CONFIG.sync_error_phase_max == 12.0
    assert DEFAULT_DPMR_CONFIG.sync_error_phase_steps == 13
    assert DEFAULT_DPMR_CONFIG.sps_search_min == 20.0
    assert DEFAULT_DPMR_CONFIG.sps_search_max == 20.0
    assert DEFAULT_DPMR_CONFIG.sps_search_steps == 1
    assert DEFAULT_DPMR_CONFIG.phase_search_min == -12.0
    assert DEFAULT_DPMR_CONFIG.phase_search_max == 12.0
    assert DEFAULT_DPMR_CONFIG.phase_search_steps == 25
    assert DEFAULT_DPMR_CONFIG.sample_windows == (0,)
    assert DEFAULT_DPMR_CONFIG.decision_ambiguous_threshold == 0.35
    assert DEFAULT_DPMR_CONFIG.header_sync_candidate_limit == 50
    assert DEFAULT_DPMR_CONFIG.header_symbol_candidate_limit == 16
    assert DEFAULT_DPMR_CONFIG.voice_sync_candidate_limit == 100
    assert DEFAULT_DPMR_CONFIG.voice_symbol_candidate_limit == 8
    assert DEFAULT_DPMR_CONFIG.dedup_frame_bucket_samples == 3_840
    assert DEFAULT_DPMR_CONFIG.stable_color_min_repeats == 2


def test_realtime_config_captures_current_defaults():
    assert DEFAULT_REALTIME_CONFIG.active_threshold_db == 15.0
    assert DEFAULT_REALTIME_CONFIG.channel_grid_hz == 12_500.0
    assert DEFAULT_REALTIME_CONFIG.close_hysteresis_windows == 3
    assert DEFAULT_REALTIME_CONFIG.call_timeout_windows == 5
    assert DEFAULT_REALTIME_CONFIG.fo_bucket_hz == 5_000.0
