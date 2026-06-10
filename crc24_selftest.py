"""Brute-force the DMR Voice LC Header CRC-24 parameters against a REAL frame.

Why: ok-dmrlib 0.8.0 ships only CRC16, and the exact CRC-24 (poly/init/xor) for
the FLC isn't something to trust from memory. This finds the param set that makes
a known-good frame verify, so dmr_pipeline_v2 can lock it in.

Usage:
  1. In dmr_pipeline_v2, temporarily print the 72-bit FLC and the 24-bit rx_crc
     for a frame you believe is clean (Golay_ok=True, IDs look sane).
  2. Paste them below as FLC_BITS (list of 0/1, length 72) and RX_CRC (int).
  3. Run: python3 crc24_selftest.py
"""

# ---- paste a real captured frame here ----
FLC_BITS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
RX_CRC = 0x902DAD   # int from ba2int(decoded[72:96])
# ------------------------------------------


def crc24_msb(bits, poly, init, xorout=0):
    reg = init
    for b in bits:
        reg ^= (b & 1) << 23
        reg &= 0xFFFFFF
        top = reg & 0x800000
        reg = (reg << 1) & 0xFFFFFF
        if top:
            reg ^= poly
    return reg ^ xorout


def self_test_engine():
    msg = b'123456789'
    bits = []
    for by in msg:
        for k in range(7, -1, -1):
            bits.append((by >> k) & 1)
    got = crc24_msb(bits, 0x864CFB, 0xB704CE)
    ok = (got == 0x21CF02)
    print("engine self-test (CRC-24/OpenPGP): 0x%06X expect 0x21CF02 -> %s" % (got, ok))
    return ok


def brute_force(flc_bits, rx_crc):
    # candidate generators commonly cited for DMR / 24-bit CRCs
    polys = [0x1864CFB & 0xFFFFFF, 0x864CFB, 0x800063, 0x5D6DCB, 0x328B63]
    inits = [0x000000, 0xFFFFFF, 0xB704CE]
    masks = [0x969696, 0x000000, 0xFFFFFF, 0x999999]
    found = []
    for poly in polys:
        for init in inits:
            calc = crc24_msb(flc_bits, poly, init)
            for mask in masks:
                if (calc ^ mask) == rx_crc:
                    found.append((poly, init, mask, "calc^mask"))
                if calc == (rx_crc ^ mask):
                    found.append((poly, init, mask, "rx^mask"))
    return found


if __name__ == '__main__':
    self_test_engine()
    if not FLC_BITS or RX_CRC is None:
        print("\nPaste FLC_BITS (len 72) and RX_CRC from a real frame, then re-run.")
    elif len(FLC_BITS) != 72:
        print("\nFLC_BITS must be length 72, got %d" % len(FLC_BITS))
    else:
        hits = brute_force(FLC_BITS, RX_CRC)
        if hits:
            print("\nMATCHING PARAM SETS (poly, init, mask, convention):")
            for h in hits:
                print("  poly=0x%06X init=0x%06X mask=0x%06X (%s)" % h)
            print("\n-> set CRC24_POLY/INIT/XOR in dmr_pipeline_v2 accordingly.")
        else:
            print("\nNo param set matched. Likely the frame has symbol errors")
            print("(check Stage 3 histogram) OR the bit ordering of the 96-bit")
            print("BPTC output differs. Try MSB/LSB-reversing rx_crc or FLC bytes.")
