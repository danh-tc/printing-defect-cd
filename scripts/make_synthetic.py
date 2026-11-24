#!/usr/bin/env python
"""Create a tiny synthetic dataset with the same layout as Taktpixel2025PD-CD.
Used only to smoke-test the pipeline before the real data is available.

Usage:
    python scripts/make_synthetic.py --out data/synthetic --n 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CATS = ["pinhole", "hair", "line", "blackspot", "colorshift", "friction"]


def make_pair(rng, cat, h=256, w=256):
    base = np.full((h, w, 3), 230, np.uint8)
    for _ in range(6):  # printed pattern
        x, y = rng.integers(0, w - 60), rng.integers(0, h - 60)
        cv2.rectangle(base, (int(x), int(y)), (int(x + 50), int(y + 30)), tuple(int(c) for c in rng.integers(40, 200, 3)), -1)
    master = base.copy()
    target = base.copy()
    mask = np.zeros((h, w), np.uint8)
    x, y = int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30))
    if cat == "pinhole":
        cv2.circle(mask, (x, y), int(rng.integers(1, 3)), 1, -1)
    elif cat == "hair":
        pts = np.cumsum(rng.integers(-4, 5, (20, 2)), 0) + [x, y]
        cv2.polylines(mask, [pts.astype(np.int32)], False, 1, 1)
    elif cat == "line":
        cv2.line(mask, (x - 25, y), (x + 25, y + int(rng.integers(-5, 5))), 1, int(rng.integers(1, 4)))
    elif cat == "blackspot":
        cv2.circle(mask, (x, y), int(rng.integers(4, 10)), 1, -1)
    else:  # colorshift / friction: larger diffuse blob
        cv2.ellipse(mask, (x, y), (20, 12), float(rng.integers(0, 180)), 0, 360, 1, -1)
    m = mask.astype(bool)
    if cat in ("colorshift", "friction"):
        target[m] = np.clip(target[m].astype(int) + [25, -15, 10], 0, 255).astype(np.uint8)
    else:
        target[m] = 20
    noise = rng.normal(0, 2, target.shape)
    target = np.clip(target + noise, 0, 255).astype(np.uint8)
    return master, target, mask * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=12, help="images per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for split in ("train", "val", "test"):
        for d in ("A", "B", "OUT"):
            (Path(args.out) / split / d).mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            cat = CATS[i % len(CATS)]
            name = f"{cat}_synthsheet_{split}_{i // len(CATS):05d}_{i:04d}.png"
            m, t, g = make_pair(rng, cat)
            base = Path(args.out) / split
            cv2.imwrite(str(base / "A" / name), cv2.cvtColor(m, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(base / "B" / name), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(base / "OUT" / name), g)
    print(f"Synthetic dataset written to {args.out}")


if __name__ == "__main__":
    main()
