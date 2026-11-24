# B1: SNUNet-CD (c16), Open-CD default 40k schedule, 256x256 crops, batch 8.
_base_ = [
    '../snunet/snunet_c16_256x256_40k_levircd.py',
    './_taktpixel_data.py',
]
