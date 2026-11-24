# B2: ChangeFormer (MiT-b0), Open-CD default 40k schedule, 256x256 crops, batch 8.
_base_ = ['../changeformer/changeformer_mit-b0_256x256_40k_levircd.py']

# ---- Taktpixel overrides (inline: mmengine forbids duplicate keys across multiple _base_ files) ----
# data_root is a symlink created by setup.sh inside third_party/open-cd/data/.
data_root = 'data/opencd_taktpixel'

# Full-resolution evaluation (tiles are 256x256): drop the base 1024 resize so thin
# defects are not resampled; masks stay at native resolution.
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=test_pipeline))

randomness = dict(seed=0, deterministic=False)
