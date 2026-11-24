#!/usr/bin/env python
"""Build panels for manual annotation review: Master | Target | GT overlay | dE heatmap.

Samples N images stratified by category (default 50 from train) and writes
<out>/panels/*.png plus <out>/review_sheet.csv to fill in by hand.

Usage:
    python scripts/review_samples.py --data data/Taktpixel2025PD-CD --split train --n 50
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdcd.data import index_split, load_sample  # noqa: E402
from pdcd.morphology import delta_e_map  # noqa: E402


def stratified(samples, n, seed):
    by_cat = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)
    rng = random.Random(seed)
    for v in by_cat.values():
        rng.shuffle(v)
    picked, cats = [], sorted(by_cat)
    while len(picked) < n and any(by_cat.values()):
        for c in cats:
            if by_cat[c] and len(picked) < n:
                picked.append(by_cat[c].pop())
    return picked


def panel(master, target, mask, max_side=512):
    de = delta_e_map(master, target)
    heat = cv2.applyColorMap(np.clip(de * 255 / 30, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = target.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)
    row = np.concatenate([master, target, overlay, heat], axis=1)
    scale = max_side / max(master.shape[:2])
    if scale < 1:
        row = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-dir", default="OUT")
    ap.add_argument("--out", default="outputs/review")
    args = ap.parse_args()

    out = Path(args.out); (out / "panels").mkdir(parents=True, exist_ok=True)
    picked = stratified(index_split(args.data, args.split, args.mask_dir), args.n, args.seed)
    rows = []
    for s in picked:
        m, t, g = load_sample(s)
        cv2.imwrite(str(out / "panels" / f"{s.stem}.png"), cv2.cvtColor(panel(m, t, g), cv2.COLOR_RGB2BGR))
        rows.append({"name": s.name, "category": s.category,
                     "label_ok(y/n)": "", "multiple_defect_types(y/n)": "",
                     "misalignment(y/n)": "", "notes": ""})
    pd.DataFrame(rows).to_csv(out / "review_sheet.csv", index=False)
    print(f"Wrote {len(rows)} panels to {out/'panels'} and review_sheet.csv")


if __name__ == "__main__":
    main()
