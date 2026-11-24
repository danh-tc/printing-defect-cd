#!/usr/bin/env python
"""Export binary prediction masks from an Open-CD checkpoint, named like the GT files,
so they can be scored with scripts/evaluate.py.

    python scripts/opencd_predict.py --config changeformer_b0_taktpixel \
        --ckpt outputs/work_dirs/<run_id>/best_mIoU_iter_xxx.pth --split val \
        --out outputs/preds/<run_id>/val
Test split requires --final.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OCD = ROOT / "third_party" / "open-cd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="name in configs/opencd without .py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        sys.exit("Refusing to predict on test without --final.")

    ckpt = str(Path(a.ckpt).resolve())
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    os.chdir(OCD)  # data_root and _base_ paths are relative to Open-CD
    sys.path.insert(0, str(OCD))
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(f"configs/taktpixel/{a.config}.py")
    cfg.work_dir = str(out.parent / "_predict_tmp")
    cfg.load_from = None
    ds = cfg.test_dataloader.dataset
    ds.data_prefix = dict(img_path_from=f"{a.split}/A", img_path_to=f"{a.split}/B",
                          seg_map_path=f"{a.split}/label")
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(ckpt, map_location="cpu")
    model = runner.model.eval()

    n = 0
    with torch.no_grad():
        for data in runner.test_dataloader:
            for o in model.test_step(data):
                name = Path(o.seg_map_path).name
                pred = o.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(str(out / name), pred)
                n += 1
    print(f"Wrote {n} masks to {out}")


if __name__ == "__main__":
    main()
