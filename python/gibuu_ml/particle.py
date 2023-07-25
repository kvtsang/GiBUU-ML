PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = 0, 0x400, 0x7ff

def encode_id(gibuu_id, charge, is_real):

    #  --------------------------------
    # |   11    | 10 9 8 7 | 6 5 ... 0 | : bit
    #  --------------------------------
    # | is_real |   Q + 8  | GiBUU ID  | : content
    #  --------------------------------

    bits = gibuu_id \
        + ((charge+8) << 7) \
        + is_real * (1 << 11)
            
    return bits

def decode_id(bits):
    gibuu_id = bits & 0x7f
    charge = ((bits >> 7) & 0xf) - 8
    is_real = (bits >> 11) & 0x1

    return gibuu_id, charge, is_real

def mask_real_bit(bits):
    return bits & 0x7FF
