#!/usr/bin/env python
"""Compare several evaluated models (outputs of scripts/evaluate.py) on the same split.

Writes to --out:
    overall.csv          pixel + component metrics at every IoU thr, per model
    per_category.csv     comp recall / FP per image / pixel IoU per category x model
    strat_<axis>.csv     component recall (+ Wilson 95% CI) per bin x model, for
                         thickness (train-quantile bins), contrast, area, thickness x contrast
    joint_misses.csv     GT components and which models matched them (@--thr)
    joint_summary.csv    #components missed by all DL models vs by exactly one, per category

Thickness is re-binned with TRAIN quantiles (results/analysis/components.csv), because the
stride-relative bins in bins.json leave G1/G2 almost empty on this dataset.

Usage:
    python scripts/compare_models.py --thr 0.3 --out outputs/compare/baselines_val \
        --model B0=outputs/eval/absdiff_val \
        --model B1=outputs/eval/<run>_val --model B2=outputs/eval/<run>_val --dl B1 B2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THICK_Q = (0.25, 0.5, 0.75)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def quantile_bins(train_comps: pd.DataFrame) -> dict:
    t = train_comps["thickness"]
    edges = [float(t.quantile(q)) for q in THICK_Q]
    labels = [f"T1(<={edges[0]:.1f})", f"T2(<={edges[1]:.1f})", f"T3(<={edges[2]:.1f})", f"T4(>{edges[2]:.1f})"]
    return {"edges": edges, "labels": labels}


def strat(pc: pd.DataFrame, keys: list[str], model: str) -> pd.DataFrame:
    g = pc.groupby(keys, observed=False)["matched"]
    t = pd.DataFrame({"n": g.size(), "hit": g.sum()}).reset_index()
    t["recall"] = t.hit / t.n.replace(0, np.nan)
    ci = [wilson(int(h), int(n)) for h, n in zip(t.hit, t.n)]
    t["ci_lo"], t["ci_hi"] = zip(*ci) if ci else ([], [])
    t["model"] = model
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=eval_dir")
    ap.add_argument("--dl", nargs="*", default=[], help="model names counted for joint misses")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--train-comps", default=str(ROOT / "results/analysis/components.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = dict(m.split("=", 1) for m in a.model)

    tr = pd.read_csv(a.train_comps)
    tr = tr[tr.split == "train"]
    qb = quantile_bins(tr)

    overall, percat, strats, pcs = [], [], [], {}
    for name, d in models.items():
        d = Path(d)
        s = pd.read_csv(d / "summary.csv")
        s.insert(0, "model", name)
        overall.append(s[s.category == "__all__"])
        percat.append(s[s.category != "__all__"])
        pc = pd.read_csv(d / "per_component.csv")
        pc = pc[np.isclose(pc.iou_thr, a.thr)].copy()
        pc["thick_q"] = pd.cut(pc.thickness, [-np.inf, *qb["edges"], np.inf], labels=qb["labels"])
        pcs[name] = pc
        for keys in (["thick_q"], ["contrast_bin"], ["area_bin"], ["thick_q", "contrast_bin"],
                     ["category"], ["category", "contrast_bin"]):
            strats.append(("_x_".join(keys), strat(pc, keys, name)))

    pd.concat(overall).to_csv(out / "overall.csv", index=False)
    pd.concat(percat).to_csv(out / "per_category.csv", index=False)
    for axis in dict.fromkeys(k for k, _ in strats):
        pd.concat([t for k, t in strats if k == axis]).to_csv(out / f"strat_{axis}.csv", index=False)

    if a.dl:
        key = ["name", "label"]
        base = pcs[a.dl[0]][key + ["category", "area", "thickness", "delta_e", "thick_q", "contrast_bin",
                                    "area_bin"]].copy()
        for m in models:
            base = base.merge(pcs[m][key + ["matched", "best_iou"]].rename(
                columns={"matched": f"hit_{m}", "best_iou": f"iou_{m}"}), on=key, how="left")
        hits = base[[f"hit_{m}" for m in a.dl]]
        base["n_dl_hit"] = hits.sum(1)
        base["missed_all_dl"] = base.n_dl_hit == 0
        base.to_csv(out / "joint_misses.csv", index=False)
        js = base.groupby("category").agg(
            n=("label", "size"),
            missed_all_dl=("missed_all_dl", "sum"),
            missed_exactly_one=("n_dl_hit", lambda s: int((s == len(a.dl) - 1).sum())),
        )
        js["frac_missed_all_dl"] = js.missed_all_dl / js.n
        js.loc["__all__"] = [js.n.sum(), js.missed_all_dl.sum(), js.missed_exactly_one.sum(),
                             js.missed_all_dl.sum() / js.n.sum()]
        js.to_csv(out / "joint_summary.csv")
    (out / "thickness_quantile_bins.txt").write_text(
        f"train thickness quantiles {THICK_Q}: {qb['edges']}\nlabels: {qb['labels']}\n")
    print(f"Wrote comparison to {out}")


if __name__ == "__main__":
    main()
