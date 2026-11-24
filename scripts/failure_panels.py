#!/usr/bin/env python
"""Panels for GT components missed by all DL models (from compare_models.py joint_misses.csv).

Each panel (one per image, missed component boxed in yellow):
    Master | Target | Target + contours (GT red, model1 green, model2 blue) | dE heatmap
Writes <out>/panels/*.png and <out>/failure_sheet.csv (fill: label_ok, visible_defect, notes).

    python scripts/failure_panels.py --data $(cat .data_root) --split val \
        --joint outputs/compare/baselines_val/joint_misses.csv \
        --pred B1=outputs/preds/<run1>/val --pred B2=outputs/preds/<run2>/val \
        --out outputs/review_failures --n 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdcd.data import Sample, read_mask, read_rgb  # noqa: E402
from pdcd.morphology import delta_e_map, label_components  # noqa: E402

COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255)]


def contours(img, mask, color):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--joint", required=True)
    ap.add_argument("--pred", action="append", required=True, help="NAME=pred_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50, help="max components, stratified by category")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.split == "test":
        sys.exit("Failure analysis is val-only.")
    preds = dict(p.split("=", 1) for p in a.pred)
    out = Path(a.out); (out / "panels").mkdir(parents=True, exist_ok=True)

    j = pd.read_csv(a.joint)
    m = j[j.missed_all_dl]
    # round-robin over categories so rare classes (line, friction) are represented
    groups = [g.sample(frac=1, random_state=a.seed) for _, g in m.groupby("category")]
    picked, i = [], 0
    while len(picked) < min(a.n, len(m)):
        for g in groups:
            if i < len(g) and len(picked) < a.n:
                picked.append(g.iloc[i])
        i += 1
    picked = pd.DataFrame(picked)

    base = Path(a.data) / a.split
    rows = []
    for _, r in picked.iterrows():
        s = Sample(a.split, r["name"], base / "A" / r["name"], base / "B" / r["name"], base / "OUT" / r["name"])
        A, B, gt = read_rgb(s.master), read_rgb(s.target), read_mask(s.mask)
        lab, _ = label_components(gt)
        comp = lab == int(r["label"])
        ov = B.copy()
        contours(ov, gt, (255, 0, 0))
        for (name, d), col in zip(preds.items(), COLORS):
            contours(ov, read_mask(Path(d) / s.name), col)
        ys, xs = np.nonzero(comp)
        cv2.rectangle(ov, (int(xs.min()) - 3, int(ys.min()) - 3), (int(xs.max()) + 3, int(ys.max()) + 3), (255, 255, 0), 1)
        de = delta_e_map(A, B)
        heat = cv2.cvtColor(cv2.applyColorMap(np.clip(de * 255 / 30, 0, 255).astype(np.uint8),
                                              cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        row = np.concatenate([A, B, ov, heat], axis=1)
        row = cv2.resize(row, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST)
        fn = f"{r['category']}_{Path(s.name).stem}_c{int(r['label'])}.png"
        cv2.imwrite(str(out / "panels" / fn), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        rows.append({"panel": fn, "name": s.name, "label": int(r["label"]), "category": r["category"],
                     "area": int(r["area"]), "thickness": round(float(r["thickness"]), 1),
                     "delta_e": round(float(r["delta_e"]), 1),
                     **{f"iou_{k}": round(float(r[f"iou_{k}"]), 2) for k in preds},
                     "label_ok(y/n)": "", "visible_defect(y/n)": "", "failure_type": "", "notes": ""})
    pd.DataFrame(rows).to_csv(out / "failure_sheet.csv", index=False)
    print(f"Wrote {len(rows)} panels to {out/'panels'} (legend: GT red, "
          + ", ".join(f"{k} {c}" for k, c in zip(preds, ['green', 'blue', 'magenta'])) + ", missed comp yellow box)")


if __name__ == "__main__":
    main()
