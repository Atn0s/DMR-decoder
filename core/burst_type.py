"""Backward-compatible DMR constants facade.

New code should import from dmr.constants.
"""

from dmr.constants import (  # noqa: F401
    DEV_NOMINAL,
    DOWN_FACTOR,
    Fs_dec,
    Fs_wide,
    NCC_THRESHOLD_DATA,
    NCC_THRESHOLD_VOICE,
    SPS,
    SYNC_TEMPLATES,
    UP_FACTOR,
    VLC_RS_MASK,
    SlotDataType,
    _hex_to_symbols,
)
