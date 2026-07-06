"""Backward-compatible DMR link-layer facade.

New code should import from dmr.link_layer.
"""

from dmr.link_layer import LateEntryCollector, decode_burst  # noqa: F401
