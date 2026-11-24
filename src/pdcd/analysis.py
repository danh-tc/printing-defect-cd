"""Stratified analysis helpers: bin definitions and recall tables.

Bins are fixed from the TRAIN split only (scripts/analyze_dataset.py writes
bins.json) and reused unchanged for val/test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Thickness bins relative to a feature stride s (default s=4, ChangeFormer).
# G1 very thin: T/s <= 0.5 | G2 thin: (0.5, 1] | G3 medium: (1, 2] | G4 thick: > 2
THICKNESS_REL_EDGES = (0.5, 1.0, 2.0)
THICKNESS_LABELS = ("G1_very_thin", "G2_thin", "G3_medium", "G4_thick")


def make_bins(train_components: pd.DataFrame, stride: int = 4) -> dict:
    a = train_components["area"].to_numpy()
    de = train_components["delta_e"].to_numpy()
    return {
        "stride": stride,
        "thickness_edges_px": [e * stride for e in THICKNESS_REL_EDGES],
        "thickness_labels": list(THICKNESS_LABELS),
        "area_edges": [float(np.quantile(a, q)) for q in (1 / 3, 2 / 3)],
        "area_labels": ["small", "medium", "large"],
        "delta_e_edges": [float(np.nanmedian(de))],
        "delta_e_labels": ["low_contrast", "high_contrast"],
        "source": "train split quantiles",
    }


def save_bins(bins: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(bins, indent=2))


def load_bins(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _cut(values: pd.Series, edges, labels) -> pd.Series:
    # right-closed bins: (-inf, e1], (e1, e2], ..., (en, inf)
    return pd.cut(values, [-np.inf, *edges, np.inf], labels=labels, right=True)


def assign_bins(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    df = df.copy()
    df["thickness_bin"] = _cut(df["thickness"], bins["thickness_edges_px"], bins["thickness_labels"])
    df["area_bin"] = _cut(df["area"], bins["area_edges"], bins["area_labels"])
    df["contrast_bin"] = _cut(df["delta_e"], bins["delta_e_edges"], bins["delta_e_labels"])
    return df


def recall_table(per_comp: pd.DataFrame, rows: str, cols: str | None = None) -> pd.DataFrame:
    """Component recall with counts, e.g. rows='thickness_bin', cols='contrast_bin'."""
    keys = [rows] + ([cols] if cols else [])
    g = per_comp.groupby(keys, observed=False)["matched"]
    t = pd.DataFrame({"recall": g.mean(), "n": g.size()}).reset_index()
    if cols:
        rec = t.pivot(index=rows, columns=cols, values="recall")
        n = t.pivot(index=rows, columns=cols, values="n")
        rec.columns = [f"recall|{c}" for c in rec.columns]
        n.columns = [f"n|{c}" for c in n.columns]
        return pd.concat([rec, n], axis=1).reset_index()
    return t
