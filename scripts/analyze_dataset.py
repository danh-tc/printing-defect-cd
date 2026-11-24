#!/usr/bin/env python
"""Phase 1 dataset analysis.

Outputs (in --out):
    components.csv   one row per GT connected component (area, thickness, dE, ...)
    images.csv       one row per image (size, #components, defect fraction, hashes)
    summary.csv      per split x category statistics
    leakage.txt      exact-duplicate and sheet-id overlap checks across splits
    bins.json        stratification bins fitted on TRAIN only

Usage:
    python scripts/analyze_dataset.py --data data/Taktpixel2025PD-CD --out outputs/analysis
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdcd.analysis import make_bins, save_bins  # noqa: E402
from pdcd.data import SPLITS, index_dataset, load_sample  # noqa: E402
from pdcd.morphology import delta_e_map, describe_components  # noqa: E402


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sheet_id(stem: str) -> str:
    """Heuristic: drop the category prefix and the trailing tile index.
    blackspot_proofrog_a4pera_back_00003_0012 -> proofrog_a4pera_back_00003"""
    parts = stem.split("_")
    if len(parts) >= 3 and re.fullmatch(r"\d+", parts[-1]):
        return "_".join(parts[1:-1])
    return "_".join(parts[1:])


def process(sample):
    master, target, mask = load_sample(sample)
    de = delta_e_map(master, target)
    comps = describe_components(mask, de)
    img_row = {
        "split": sample.split, "name": sample.name, "category": sample.category,
        "height": mask.shape[0], "width": mask.shape[1],
        "size_match": master.shape == target.shape and master.shape[:2] == mask.shape,
        "n_components": len(comps), "defect_fraction": float(mask.mean()),
        "bg_delta_e_mean": float(de[~mask].mean()) if (~mask).any() else float("nan"),
        "md5_A": md5(sample.master), "md5_B": md5(sample.target),
        "sheet_id": sheet_id(sample.stem),
    }
    comp_rows = [{"split": sample.split, "name": sample.name, "category": sample.category,
                  **c.to_dict()} for c in comps]
    return img_row, comp_rows


def leakage_report(images: pd.DataFrame) -> str:
    lines = []
    for col in ("md5_A", "md5_B"):
        g = images.groupby(col)["split"].nunique()
        dup = g[g > 1]
        lines.append(f"{col}: {len(dup)} identical files appear in more than one split")
    lines.append("")
    by_sheet = defaultdict(set)
    for sid, sp in zip(images["sheet_id"], images["split"]):
        by_sheet[sid].add(sp)
    shared = {k: v for k, v in by_sheet.items() if len(v) > 1}
    lines.append(f"sheet_id (heuristic, filename-based): {len(by_sheet)} unique, "
                 f"{len(shared)} shared across splits")
    for k, v in list(shared.items())[:20]:
        lines.append(f"  {k}: {sorted(v)}")
    lines.append("\nNote: sheet_id overlap is not necessarily leakage (tiles of one sheet "
                 "may be split by design); inspect before concluding.")
    return "\n".join(lines)


def summarize(images: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    img = images.groupby(["split", "category"]).agg(
        n_images=("name", "size"),
        imgs_multi_component=("n_components", lambda s: int((s > 1).sum())),
        imgs_no_component=("n_components", lambda s: int((s == 0).sum())),
        mean_defect_fraction=("defect_fraction", "mean"),
    )
    q = lambda p: (lambda s: s.quantile(p))  # noqa: E731
    cmp = comps.groupby(["split", "category"]).agg(
        n_components=("area", "size"),
        area_p10=("area", q(0.1)), area_p50=("area", "median"), area_p90=("area", q(0.9)),
        thick_p10=("thickness", q(0.1)), thick_p50=("thickness", "median"),
        thick_p90=("thickness", q(0.9)),
        de_p10=("delta_e", q(0.1)), de_p50=("delta_e", "median"), de_p90=("delta_e", q(0.9)),
        frac_thick_le4=("thickness", lambda s: float((s <= 4).mean())),
    )
    return img.join(cmp, how="left").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs/analysis")
    ap.add_argument("--mask-dir", default="OUT")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = index_dataset(args.data, SPLITS, args.mask_dir)
    samples = [s for sp in SPLITS for s in ds[sp]]
    print({sp: len(v) for sp, v in ds.items()})

    img_rows, comp_rows = [], []
    with Pool(args.workers) as pool:
        for ir, cr in tqdm(pool.imap_unordered(process, samples, chunksize=8), total=len(samples)):
            img_rows.append(ir); comp_rows.extend(cr)

    images = pd.DataFrame(img_rows).sort_values(["split", "name"])
    comps = pd.DataFrame(comp_rows).sort_values(["split", "name", "label"])
    images.to_csv(out / "images.csv", index=False)
    comps.to_csv(out / "components.csv", index=False)
    summarize(images, comps).to_csv(out / "summary.csv", index=False)
    (out / "leakage.txt").write_text(leakage_report(images))
    save_bins(make_bins(comps[comps.split == "train"], args.stride), out / "bins.json")

    bad = images[~images.size_match]
    if len(bad):
        print(f"WARNING: {len(bad)} images with size mismatch between A/B/mask")
    print(f"Wrote analysis to {out}")


if __name__ == "__main__":
    main()
