import numpy as np
from enum import IntEnum

Fs_wide  = 2_500_000.0
Fs_dec   = 48_000.0
SPS      = 10
UP_FACTOR   = 12
DOWN_FACTOR = 625
NCC_THRESHOLD_VOICE = 0.68
NCC_THRESHOLD_DATA  = 0.55
DEV_NOMINAL = 1944.0
VLC_RS_MASK = bytes([0x96, 0x96, 0x96])

class SlotDataType(IntEnum):
    PI_HEADER        = 0
    VOICE_LC_HEADER  = 1
    TERMINATOR_WITH_LC = 2
    CSBK             = 3
    MBC_HEADER       = 4
    MBCC             = 5
    DATA_HEADER      = 6
    RATE_HALF        = 7
    RATE_34          = 8
    IDLE             = 9
    RATE_1           = 10

def _hex_to_symbols(hex_str: str) -> np.ndarray:
    bin_str = "".join(f"{int(c,16):04b}" for c in hex_str)
    tbl = {'01':3,'00':1,'10':-1,'11':-3}
    return np.array([tbl[bin_str[i:i+2]] for i in range(0,len(bin_str),2)])

SYNC_TEMPLATES = {
    "BS_VOICE": _hex_to_symbols("755FD7DF75F7"),
    "MS_VOICE": _hex_to_symbols("7F7D5DD57DFD"),
    "DATA_BS":  _hex_to_symbols("DFF57D75DF5D"),
    "DATA_MS":  _hex_to_symbols("D5D7F77FD757"),
}
