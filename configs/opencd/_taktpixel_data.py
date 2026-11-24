# Dataset overrides shared by all Taktpixel configs (merged on top of the model config).
# data_root is a symlink created by setup.sh inside third_party/open-cd/data/.
data_root = 'data/opencd_taktpixel'

# Full-resolution evaluation: no resize, so thin defects are not blurred away.
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=test_pipeline))

randomness = dict(seed=0, deterministic=False)
