"""Backward-compatible DMR DSP facade.

New code should import from dmr.dsp.
"""

from dmr.dsp import (  # noqa: F401
    _interp,
    adaptive_slice_bits,
    find_sync_positions,
    frontend,
    lc_front_end_compat,
    read_rawiq,
    recover_burst,
)
