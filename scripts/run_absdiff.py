#!/usr/bin/env python
"""B0: Absolute Difference / dE baseline.

1) --tune : grid-search (signal, blur, min_area, threshold) on VAL by pixel F1,
            write best config to <out>/best_config.json and the full grid to grid.csv.
2) --predict SPLIT : write predicted masks with the best config to <out>/<split>/.

Usage:
    python scripts/run_absdiff.py --data data/Taktpixel2025PD-CD --out outputs/preds/absdiff --tune
    python scripts/run_absdiff.py --data data/Taktpixel2025PD-CD --out outputs/preds/absdiff --predict val
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdcd.baselines.absdiff import AbsDiffConfig, difference_map, predict, predict_from_diff  # noqa: E402
from pdcd.data import index_split, load_sample  # noqa: E402

GRID = {
    "signal": ["de", "rgb"],
    "blur_sigma": [0.0, 1.0, 2.0],
    "min_area": [0, 4, 16],
}
THRESHOLDS = {"de": np.arange(2, 42, 2.0), "rgb": np.arange(5, 105, 5.0)}


def tune_one(sample):
    """Return {(signal, blur, min_area, thr): [tp, fp, fn]} for one image."""
    master, target, gt = load_sample(sample)
    res = {}
    for sig, blur in itertools.product(GRID["signal"], GRID["blur_sigma"]):
        d = difference_map(master, target, sig, blur)
        for ma, thr in itertools.product(GRID["min_area"], THRESHOLDS[sig]):
            p = predict_from_diff(d, thr, ma)
            res[(sig, blur, ma, float(thr))] = (int((p & gt).sum()), int((p & ~gt).sum()), int((~p & gt).sum()))
    return res


def predict_one(args):
    sample, cfg, out_dir = args
    master, target, _ = load_sample(sample)
    p = predict(master, target, cfg)
    cv2.imwrite(str(Path(out_dir) / sample.name), (p * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs/preds/absdiff")
    ap.add_argument("--mask-dir", default="OUT")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--predict", choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.tune:
        samples = index_split(args.data, "val", args.mask_dir)
        tot = {}
        with Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(tune_one, samples, chunksize=4), total=len(samples)):
                for k, v in r.items():
                    a = tot.setdefault(k, np.zeros(3, dtype=np.int64)); a += v
        rows = []
        for (sig, blur, ma, thr), (tp, fp, fn) in tot.items():
            pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            rows.append({"signal": sig, "blur_sigma": blur, "min_area": ma, "threshold": thr,
                         "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9),
                         "iou": tp / max(tp + fp + fn, 1)})
        grid = pd.DataFrame(rows).sort_values("f1", ascending=False)
        grid.to_csv(out / "grid.csv", index=False)
        best = grid.iloc[0]
        cfg = AbsDiffConfig(best.signal, float(best.blur_sigma), float(best.threshold), int(best.min_area))
        (out / "best_config.json").write_text(json.dumps(cfg.__dict__, indent=2))
        print("Best on val:", cfg, f"F1={best.f1:.4f} IoU={best.iou:.4f}")

    if args.predict:
        cfg = AbsDiffConfig(**json.loads((out / "best_config.json").read_text()))
        split_dir = out / args.predict; split_dir.mkdir(exist_ok=True)
        samples = index_split(args.data, args.predict, args.mask_dir)
        with Pool(args.workers) as pool:
            list(tqdm(pool.imap_unordered(predict_one, [(s, cfg, split_dir) for s in samples]),
                      total=len(samples)))
        print(f"Wrote {len(samples)} masks to {split_dir}")


if __name__ == "__main__":
    main()
