# Pilot H-aug: B2 ChangeFormer (MiT-b0) with the SAME photometric distortion applied to A and B
# (Open-CD default samples it independently per image). Everything else identical to
# changeformer_b0_taktpixel.py (40k, batch 8, same base pipeline order and parameters).
_base_ = ['./changeformer_b0_taktpixel.py']

custom_imports = dict(imports=['pdcd.opencd_ext'], allow_failed_imports=False)

crop_size = (256, 256)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='PairedPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
