#!/usr/bin/env python
"""Evaluate a folder of predicted binary masks against GT.

Predictions: PNG masks named exactly like the GT files (non-zero = defect).

Outputs (in --out):
    summary.csv        pixel + component metrics, overall and per category
    per_component.csv  every GT component with descriptors and matched flag
    strat_thickness_contrast.csv, strat_area.csv, strat_category_thickness.csv
                       component recall @ --strat-thr by bins from bins.json

Usage:
    python scripts/evaluate.py --data data/Taktpixel2025PD-CD --split val \
        --pred outputs/preds/absdiff/val --bins outputs/analysis/bins.json \
        --out outputs/eval/absdiff_val
Test split requires --final (use it once, on frozen models).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdcd.analysis import assign_bins, load_bins, recall_table  # noqa: E402
from pdcd.data import index_split, load_sample, read_mask  # noqa: E402
from pdcd.metrics import Evaluator  # noqa: E402
from pdcd.morphology import delta_e_map, describe_components  # noqa: E402

THRS = (0.1, 0.3, 0.5)


def eval_one(args):
    sample, pred_dir = args
    master, target, gt = load_sample(sample)
    p = Path(pred_dir) / sample.name
    if not p.exists():
        raise FileNotFoundError(f"Missing prediction: {p}")
    pred = read_mask(p)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {sample.name}: pred {pred.shape} vs gt {gt.shape}")
    comps = describe_components(gt, delta_e_map(master, target))
    ev = Evaluator(iou_thrs=THRS)
    ev.update(pred, gt, sample.category, sample.name, comps)
    return ev


def merge(evs: list[Evaluator]) -> Evaluator:
    tot = Evaluator(iou_thrs=THRS)
    for ev in evs:
        for k, v in ev.pixel.items():
            tot.pixel.setdefault(k, type(v)()).__iadd__(v)
        for k, v in ev.comp.items():
            c = tot.comp.setdefault(k, type(v)())
            c.n_gt += v.n_gt; c.n_gt_matched += v.n_gt_matched
            c.n_pred += v.n_pred; c.n_pred_matched += v.n_pred_matched; c.n_images += v.n_images
        tot.per_component.extend(ev.per_component)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--bins", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-dir", default="OUT")
    ap.add_argument("--strat-thr", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--final", action="store_true", help="required to evaluate on test")
    ap.add_argument("--exclude", help="CSV with a 'name' column (e.g. results/analysis/dup_with_train.csv: "
                                      "val/test pairs byte-identical to a train pair) to drop from scoring")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("Refusing to evaluate on test without --final (test is for frozen models only).")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    samples = index_split(args.data, args.split, args.mask_dir)
    if args.exclude:
        drop = set(pd.read_csv(args.exclude)["name"])
        n0 = len(samples)
        samples = [s for s in samples if s.name not in drop]
        print(f"--exclude: dropped {n0 - len(samples)} / {n0} images")
    with Pool(args.workers) as pool:
        evs = list(tqdm(pool.imap_unordered(eval_one, [(s, args.pred) for s in samples], chunksize=8),
                        total=len(samples)))
    ev = merge(evs)

    summary = pd.DataFrame(ev.summary())
    summary.to_csv(out / "summary.csv", index=False)
    pc = assign_bins(pd.DataFrame(ev.per_component), load_bins(args.bins))
    pc.to_csv(out / "per_component.csv", index=False)
    pc_t = pc[pc.iou_thr == args.strat_thr]
    recall_table(pc_t, "thickness_bin", "contrast_bin").to_csv(out / "strat_thickness_contrast.csv", index=False)
    recall_table(pc_t, "area_bin").to_csv(out / "strat_area.csv", index=False)
    recall_table(pc_t, "category", "thickness_bin").to_csv(out / "strat_category_thickness.csv", index=False)

    cols = ["category", "pix_iou", "pix_f1", f"comp_recall@{args.strat_thr}",
            f"comp_f1@{args.strat_thr}", f"fp_per_image@{args.strat_thr}", f"n_gt_components@{args.strat_thr}"]
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary[cols].to_string(index=False))
    print(f"\nWrote evaluation to {out}")


if __name__ == "__main__":
    main()
