# B2: ChangeFormer (MiT-b0), Open-CD default 40k schedule, 256x256 crops, batch 8.
_base_ = [
    '../changeformer/changeformer_mit-b0_256x256_40k_levircd.py',
    './_taktpixel_data.py',
]
