"""Backward-compatible DMR decoder facade.

New code should import from dmr.decoder.
"""

from dmr.decoder import LateEntryCollector, decode_burst  # noqa: F401
