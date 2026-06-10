import os
import numpy as np
import scipy.signal as signal
from bitarray import bitarray
from bitarray.util import ba2int

from okdmr.dmrlib.etsi.fec.golay_20_8_7 import Golay2087
from okdmr.dmrlib.etsi.fec.bptc_196_96 import BPTC19696
from okdmr.dmrlib.etsi.fec.vbptc_128_72 import VBPTC12873
from okdmr.dmrlib.etsi.layer2.pdu.embedded_signalling import EmbeddedSignalling
from okdmr.dmrlib.etsi.layer2.elements.lcss import LCSS

Fs_wide = 2500000.0
Fs_dec = 48000.0
SPS = 10
UP_FACTOR = 12
DOWN_FACTOR = 625

def read_rawiq(filename):
    data = np.fromfile(filename, dtype=np.int16)
    I = data[0::2]
    Q = data[1::2]
    length = min(len(I), len(Q))
    return (I[:length] + 1j * Q[:length]) / 32768.0

def hex_to_symbols(hex_str):
    bin_str = "".join(f"{int(c, 16):04b}" for c in hex_str)
    symbols = []
    for i in range(0, len(bin_str), 2):
        dibit = bin_str[i:i+2]
        if dibit == '01': symbols.append(3)
        elif dibit == '00': symbols.append(1)
        elif dibit == '10': symbols.append(-1)
        elif dibit == '11': symbols.append(-3)
    return np.array(symbols)

def slice_symbol_to_bits(val):
    if val > 2.0: return [0, 1]
    elif val > 0.0: return [0, 0]
    elif val > -2.0: return [1, 0]
    else: return [1, 1]

def reverse_bits_in_bytes(ba):
    """ 对 bitarray 进行字节内的比特反转 """
    by = ba.tobytes()
    rev = bitarray()
    for b in by:
        rev.frombytes(bytes([int(f'{b:08b}'[::-1], 2)]))
    return rev

def main():
    target_file = "synthesized_wideband_2.5MHz.rawiq"
    if not os.path.exists(target_file):
        print("找不到文件！")
        return
        
    iq = read_rawiq(target_file)
    
    # ---- 1. 获取黄金信令标准 (解调开头 332292 处的 Voice LC Header) ----
    f_offset = 150000.0
    t = np.arange(len(iq)) / Fs_wide
    iq_shifted = iq * np.exp(-1j * 2 * np.pi * f_offset * t)
    iq_dec = signal.resample_poly(iq_shifted, UP_FACTOR, DOWN_FACTOR)
    
    y_demod = np.angle(iq_dec[1:] * np.conj(iq_dec[:-1]))
    # 局部频偏自动修正
    y_demod_centered = y_demod - np.mean(y_demod[330000:])
    y_scaled = y_demod_centered * (3.0 / (2.0 * np.pi * 1944.0 / Fs_dec))
    y_clipped = np.clip(y_scaled, -4.0, 4.0)
    
    # 提取第 0 帧并译码，得到黄金 72-bit (9字节) 数组
    header_indices = 332292 - 655 + np.arange(132) * 10
    header_bits = []
    for idx in header_indices:
        header_bits.extend(slice_symbol_to_bits(y_clipped[idx]))
    header_ba = bitarray(header_bits)
    info_bits = header_ba[0:98] + header_ba[166:264]
    
    gold_decoded = BPTC19696.repair_if_necessary(info_bits, deinterleaved=False)
    gold_lc_pdu = gold_decoded[0:72]
    
    # ---- 2. 状态机动态对齐提取：获取真正无错对齐的 4 个 32-bit 碎片 ----
    # 搜索 Voice SYNC 负峰，精确定位第一帧语音 Burst A 的位置
    t_symbols_voice = hex_to_symbols("7F7D5DD57DFD")
    t_wave_voice = np.repeat(t_symbols_voice, SPS)
    corr_linear = signal.correlate(y_clipped, t_wave_voice, mode='same')
    y_sq = y_clipped ** 2
    y_energy = np.convolve(y_sq, np.ones(len(t_wave_voice)), mode='same')
    y_energy = np.where(y_energy == 0, 1e-10, y_energy)
    ncc = corr_linear / np.sqrt(y_energy * np.sum(t_wave_voice ** 2))
    peaks, _ = signal.find_peaks(-ncc, height=0.68, distance=800)
    
    if len(peaks) == 0:
        print("未在该频点上实现同步。")
        return
        
    anchor_peak = peaks[0]
    
    fragments = []
    is_collecting = False
    
    # 状态机：只有当遇到 First 时，才真正启动 4 帧碎片的收集
    for j in range(12):
        center_burst = anchor_peak + 2880 * (j + 1)
        sample_indices = center_burst - 115 + np.arange(24) * 10
        sample_indices = np.clip(sample_indices, 0, len(y_clipped) - 1).astype(int)
        
        burst_bits = []
        for idx in sample_indices:
            burst_bits.extend(slice_symbol_to_bits(y_clipped[idx]))
        burst_ba = bitarray(burst_bits)
        emb_bits = burst_ba[0:8] + burst_ba[40:48]
        signalling_bits = burst_ba[8:40]
        
        emb_pdu = EmbeddedSignalling.from_bits(emb_bits)
        lcss_status = emb_pdu.link_control_start_stop
        
        if not is_collecting:
            if lcss_status == LCSS.FirstFragmentLC:
                is_collecting = True
                fragments.append(signalling_bits)
        else:
            fragments.append(signalling_bits)
            if len(fragments) == 4:
                break

    if len(fragments) < 4:
        print("未能在时间轴上收集齐 4 帧嵌入信令！")
        return

    # 拼接得到我们提取出并对准的 128 位原始数据
    extracted_128 = fragments[0] + fragments[1] + fragments[2] + fragments[3]

    # ---- 3. 比特级打印比对 ----
    print("="*80)
    print("                      DMR 时域二进制比特级调试比对")
    print("="*80)
    print(f"1. 黄金信令 72-bit (Gold LC PDU)   : \n   {gold_lc_pdu.to01()}")
    print(f"   长度: {len(gold_lc_pdu)} bits")
    print("-"*80)
    print(f"2. 中途提取的 128-bit 原始控制块  : \n   {extracted_128.to01()}")
    print(f"   长度: {len(extracted_128)} bits")
    print(f"   其中：")
    print(f"     时隙 B 碎片 (32-bit): {fragments[0].to01()}")
    print(f"     时隙 C 碎片 (32-bit): {fragments[1].to01()}")
    print(f"     时隙 D 碎片 (32-bit): {fragments[2].to01()}")
    print(f"     时隙 E 碎片 (32-bit): {fragments[3].to01()}")
    print("="*80)

    # ---- 4. 暴力破译（对齐后的数据） ----
    print("正在对对准后的数据进行比特序高精度暴力破译...")
    transforms = {
        "Raw (不反转)": lambda b: b,
        "Bit-Reverse (整包全反转)": lambda b: b[::-1]
    }
    
    found = False
    for name_1, t1 in transforms.items():
        for name_2, t2 in transforms.items():
            b1 = t1(fragments[0])
            b2 = t2(fragments[1])
            b3 = t2(fragments[2])
            b4 = t2(fragments[3])
            
            test_128 = b1 + b2 + b3 + b4
            
            try:
                # 剔除 5 位校验和进行 72 位解帧
                lc_pdu = VBPTC12873.deinterleave_data_bits(test_128, include_cs5=False)
                test_bytes = lc_pdu.tobytes()
                
                if test_bytes == gold_lc_pdu.tobytes():
                    print(f"\n[🎉 破译成功！找到完美匹配特征算法]：")
                    print(f"  ├─ 碎片 1 变换方式: {name_1}")
                    print(f"  ├─ 碎片 2~4 变换方式: {name_2}")
                    print(f"  └─ 译码结果字节   : {test_bytes.hex()}")
                    
                    destination_id = (test_bytes[3] << 16) | (test_bytes[4] << 8) | test_bytes[5]
                    source_id = (test_bytes[6] << 16) | (test_bytes[7] << 8) | test_bytes[8]
                    print(f"     =========================================")
                    print(f"     主叫对讲机 ID (Source ID)     : {source_id}")
                    print(f"     被叫对讲机 ID (Destination ID): {destination_id}")
                    print(f"     =========================================")
                    found = True
                    break
            except Exception:
                continue
        if found:
            break
            
    if not found:
        print("\n[未找到匹配]：请检查是否由于滑窗对齐或时隙提取存在偏差。")

if __name__ == '__main__':
    main()