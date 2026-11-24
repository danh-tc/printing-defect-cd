"""Open-CD extensions, loaded in configs via
    custom_imports = dict(imports=['pdcd.opencd_ext'], allow_failed_imports=False)
(train_opencd.sh / opencd_predict.py put <repo>/src on PYTHONPATH).
"""
from __future__ import annotations

import numpy as np
from opencd.datasets.transforms.transforms import MultiImgPhotoMetricDistortion
from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PairedPhotoMetricDistortion(MultiImgPhotoMetricDistortion):
    """Same photometric distortion (identical random draws) for every image of the pair.

    Open-CD's MultiImgPhotoMetricDistortion samples brightness/contrast/saturation/hue
    independently per image, which injects a global master-vs-target colour difference
    labelled 'no change' (H-aug in PROGRESS.md). Here the numpy RNG state is replayed so
    A and B receive the same parameters: global appearance still varies, A-B consistency
    is preserved.
    """

    def transform(self, results: dict) -> dict:
        state = np.random.get_state()
        out = []
        for img in results["img"]:
            np.random.set_state(state)
            out.append(super().transform({"img": [img]})["img"][0])
        results["img"] = out
        return results
