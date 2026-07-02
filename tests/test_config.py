from common.config import DEFAULT_RADIO_CONFIG, DEFAULT_REALTIME_CONFIG
from dmr.config import DEFAULT_DMR_CONFIG
from dmr.constants import (
    DOWN_FACTOR,
    Fs_dec,
    Fs_wide,
    NCC_THRESHOLD_DATA,
    NCC_THRESHOLD_VOICE,
    UP_FACTOR,
)
from dmr.engine import BURST_STRIDE
from dpmr.config import DEFAULT_DPMR_CONFIG
from p25.config import DEFAULT_P25_CONFIG

import protocols
import scanner


def test_radio_config_matches_legacy_scanner_constants():
    assert DEFAULT_RADIO_CONFIG.target_sample_rate_hz == Fs_dec
    assert DEFAULT_RADIO_CONFIG.sample_rate_tolerance_hz == 1.0
    assert DEFAULT_RADIO_CONFIG.wideband_sample_rate_hz == Fs_wide
    assert DEFAULT_RADIO_CONFIG.wideband_resample_up == UP_FACTOR
    assert DEFAULT_RADIO_CONFIG.wideband_resample_down == DOWN_FACTOR
    assert DEFAULT_RADIO_CONFIG.psd_peak_threshold_db == scanner.PSD_PEAK_THRESHOLD_DB
    assert DEFAULT_RADIO_CONFIG.target_sample_rate_hz == scanner.Fs_dec
    assert DEFAULT_RADIO_CONFIG.wideband_sample_rate_hz == scanner.Fs_wide


def test_protocol_specs_expose_default_configs():
    assert protocols.spec_for_protocol("dmr").config is DEFAULT_DMR_CONFIG
    assert protocols.spec_for_protocol("p25").config is DEFAULT_P25_CONFIG
    assert protocols.spec_for_protocol("dpmr").config is DEFAULT_DPMR_CONFIG


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
    assert DEFAULT_DMR_CONFIG.burst_dedup_window_samples == 50
    assert DEFAULT_DMR_CONFIG.dedup_frequency_bucket_hz == 5_000.0


def test_protocol_configs_capture_frontend_and_dedup_defaults():
    assert DEFAULT_DMR_CONFIG.frontend_min_samples == 512
    assert DEFAULT_DMR_CONFIG.frontend_psd_nperseg == 4096
    assert DEFAULT_P25_CONFIG.frontend_min_samples == 512
    assert DEFAULT_P25_CONFIG.frontend_psd_nperseg == 4096
    assert DEFAULT_P25_CONFIG.dedup_frame_bucket_samples == 8_640
    assert DEFAULT_DPMR_CONFIG.frontend_min_samples == 512
    assert DEFAULT_DPMR_CONFIG.frontend_psd_nperseg == 4096
    assert DEFAULT_DPMR_CONFIG.dedup_frame_bucket_samples == 3_840
    assert DEFAULT_DPMR_CONFIG.stable_color_min_repeats == 2


def test_realtime_config_captures_current_defaults():
    assert DEFAULT_REALTIME_CONFIG.active_threshold_db == 15.0
    assert DEFAULT_REALTIME_CONFIG.channel_grid_hz == 12_500.0
    assert DEFAULT_REALTIME_CONFIG.close_hysteresis_windows == 3
    assert DEFAULT_REALTIME_CONFIG.call_timeout_windows == 5
    assert DEFAULT_REALTIME_CONFIG.fo_bucket_hz == 5_000.0
