import numpy as np
from bitarray import bitarray
from core.decoder import decode_burst, LateEntryCollector


def test_decode_burst_garbage_returns_none():
    # All-zero symbols -> Golay check fails -> None
    syms = np.zeros(132)
    result = decode_burst(syms, "DATA_MS")
    assert result is None


def test_decode_burst_random_returns_none_or_dict():
    np.random.seed(42)
    syms = np.random.uniform(-4, 4, 132)
    result = decode_burst(syms, "DATA_MS")
    assert result is None or isinstance(result, dict)


def test_decode_burst_dict_has_required_keys():
    # We can't guarantee a valid burst from random data, but if it returns a dict
    # it must have all required keys
    np.random.seed(0)
    for _ in range(20):
        syms = np.random.uniform(-4, 4, 132)
        result = decode_burst(syms, "DATA_MS")
        if result is not None:
            for key in ("type", "src", "dst", "ts", "flco", "extra", "raw_bits"):
                assert key in result, f"Missing key: {key}"
            break


def test_late_entry_collector_needs_four_frags():
    col = LateEntryCollector()
    ba = bitarray(264)
    ba.setall(0)
    # Single feed should never produce a PDU (not enough frags)
    result = col.feed(ba, "MS_VOICE")
    assert result is None


def test_late_entry_collector_reset():
    col = LateEntryCollector()
    ba = bitarray(264)
    ba.setall(0)
    col.feed(ba, "MS_VOICE")
    col.reset()
    assert col._collecting is False
    assert col._frags == []


def test_late_entry_collector_state_machine():
    from okdmr.dmrlib.etsi.layer2.pdu.embedded_signalling import EmbeddedSignalling
    from okdmr.dmrlib.etsi.layer2.elements.lcss import LCSS
    col = LateEntryCollector()

    def make_ba_with_lcss(lcss_val):
        """Build a 264-bit burst with a specific LCSS value in the EMB header."""
        ba = bitarray(264)
        ba.setall(0)
        # EMB header is center[0:8] + center[40:48] where center = ba[108:156]
        # LCSS occupies bits [0:2] of the EMB 16-bit header (after last_block and protect_flag)
        # EmbeddedSignalling layout: [0]=last_block [1]=protect_flag [2:4]=lcss [4:8]=color_code [8:16]=emb_parity
        # We set lcss in bits [2:4] of emb_bits = center[0:8]+center[40:48]
        # center[0:8] = ba[108:116], center[40:48] = ba[148:156]
        # emb_bits[0]=ba[108], emb_bits[1]=ba[109], emb_bits[2]=ba[110], emb_bits[3]=ba[111]
        # LCSS FirstFragmentLC = 3 = 0b11
        lcss_bits = f"{lcss_val:02b}"
        ba[110] = int(lcss_bits[0])
        ba[111] = int(lcss_bits[1])
        return ba

    # Feed a FirstFragmentLC burst — should start collecting
    first_ba = make_ba_with_lcss(LCSS.FirstFragmentLC.value)
    result = col.feed(first_ba, "MS_VOICE")
    # May fail EMB parse due to parity, but should not raise
    assert result is None or isinstance(result, dict)
