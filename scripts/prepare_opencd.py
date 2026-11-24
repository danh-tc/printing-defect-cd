#!/usr/bin/env python
"""Convert the dataset to the LEVIR-CD-style layout used by Open-CD / OpenCD-Lite:
    <out>/{train,val,test}/{A,B,label}/<name>.png   (label values 0/255)

A and B are symlinked (no copy); masks are re-written as 0/255 uint8.
Verify the label convention expected by your Open-CD config before training.

Usage:
    python scripts/prepare_opencd.py --data data/Taktpixel2025PD-CD --out data/opencd_taktpixel
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdcd.data import SPLITS, index_split, read_mask  # noqa: E402


def link(src: Path, dst: Path, copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        dst.write_bytes(src.read_bytes())
    else:
        os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-dir", default="OUT")
    ap.add_argument("--label-value", type=int, default=255, choices=[1, 255])
    ap.add_argument("--copy", action="store_true", help="copy A/B instead of symlinking")
    args = ap.parse_args()

    for split in SPLITS:
        samples = index_split(args.data, split, args.mask_dir)
        dirs = {k: Path(args.out) / split / k for k in ("A", "B", "label")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        for s in tqdm(samples, desc=split):
            name = Path(s.name).with_suffix(".png").name
            link(s.master, dirs["A"] / s.name, args.copy)
            link(s.target, dirs["B"] / s.name, args.copy)
            m = read_mask(s.mask).astype(np.uint8) * args.label_value
            cv2.imwrite(str(dirs["label"] / name), m)
    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
