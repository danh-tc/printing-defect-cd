"""Pixel-level and component-level evaluation.

Component matching: one-to-one Hungarian matching between GT and predicted
8-connected components on IoU; a pair counts as a match if IoU >= thr.
    Component Recall    = matched GT / #GT components
    Component Precision = matched pred / #pred components
    FP/image            = unmatched predicted components per image
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .morphology import label_components

EPS = 1e-9


@dataclass
class PixelCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        p, g = pred.astype(bool), gt.astype(bool)
        self.tp += int(np.sum(p & g))
        self.fp += int(np.sum(p & ~g))
        self.fn += int(np.sum(~p & g))
        self.tn += int(np.sum(~p & ~g))

    def __iadd__(self, o: "PixelCounts"):
        self.tp += o.tp; self.fp += o.fp; self.fn += o.fn; self.tn += o.tn
        return self

    def summary(self) -> dict:
        prec = self.tp / (self.tp + self.fp + EPS)
        rec = self.tp / (self.tp + self.fn + EPS)
        return {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec + EPS),
            "iou": self.tp / (self.tp + self.fp + self.fn + EPS),
        }


def iou_matrix(gt_lab: np.ndarray, n_gt: int, pr_lab: np.ndarray, n_pr: int) -> np.ndarray:
    """IoU between every GT and predicted component, via joint histogram."""
    if n_gt == 0 or n_pr == 0:
        return np.zeros((n_gt, n_pr), dtype=np.float64)
    joint = np.bincount(
        gt_lab.ravel().astype(np.int64) * (n_pr + 1) + pr_lab.ravel(),
        minlength=(n_gt + 1) * (n_pr + 1),
    ).reshape(n_gt + 1, n_pr + 1)
    inter = joint[1:, 1:].astype(np.float64)
    area_gt = joint[1:, :].sum(1, keepdims=True)
    area_pr = joint[:, 1:].sum(0, keepdims=True)
    return inter / (area_gt + area_pr - inter + EPS)


@dataclass
class ImageComponentResult:
    n_gt: int
    n_pred: int
    gt_matched: np.ndarray          # bool per GT component (label order 1..n_gt)
    gt_best_iou: np.ndarray         # best IoU of each GT comp with any pred comp
    n_pred_matched: int

    @property
    def n_fp(self) -> int:
        return self.n_pred - self.n_pred_matched


def match_components(pred: np.ndarray, gt: np.ndarray, thr: float = 0.3,
                     gt_lab=None, n_gt=None) -> ImageComponentResult:
    if gt_lab is None:
        gt_lab, n_gt = label_components(gt)
    pr_lab, n_pr = label_components(pred)
    ious = iou_matrix(gt_lab, n_gt, pr_lab, n_pr)
    matched = np.zeros(n_gt, dtype=bool)
    n_pm = 0
    if n_gt and n_pr:
        r, c = linear_sum_assignment(-ious)
        ok = ious[r, c] >= thr
        matched[r[ok]] = True
        n_pm = int(ok.sum())
    best = ious.max(1) if n_pr else np.zeros(n_gt)
    return ImageComponentResult(n_gt, n_pr, matched, best, n_pm)


@dataclass
class ComponentCounts:
    n_gt: int = 0
    n_gt_matched: int = 0
    n_pred: int = 0
    n_pred_matched: int = 0
    n_images: int = 0

    def update(self, r: ImageComponentResult) -> None:
        self.n_gt += r.n_gt
        self.n_gt_matched += int(r.gt_matched.sum())
        self.n_pred += r.n_pred
        self.n_pred_matched += r.n_pred_matched
        self.n_images += 1

    def summary(self) -> dict:
        rec = self.n_gt_matched / (self.n_gt + EPS)
        prec = self.n_pred_matched / (self.n_pred + EPS)
        return {
            "comp_recall": rec,
            "comp_precision": prec,
            "comp_f1": 2 * prec * rec / (prec + rec + EPS),
            "fp_per_image": (self.n_pred - self.n_pred_matched) / max(self.n_images, 1),
            "n_gt_components": self.n_gt,
            "n_images": self.n_images,
        }


@dataclass
class Evaluator:
    """Accumulates metrics overall and per category for a list of thresholds."""
    iou_thrs: tuple = (0.1, 0.3, 0.5)
    pixel: dict = field(default_factory=dict)       # category -> PixelCounts
    comp: dict = field(default_factory=dict)        # (category, thr) -> ComponentCounts
    per_component: list = field(default_factory=list)  # rows for stratified analysis

    def update(self, pred: np.ndarray, gt: np.ndarray, category: str, name: str,
               comp_desc: list | None = None) -> None:
        for cat in ("__all__", category):
            self.pixel.setdefault(cat, PixelCounts()).update(pred, gt)
        gt_lab, n_gt = label_components(gt)
        for thr in self.iou_thrs:
            r = match_components(pred, gt, thr, gt_lab, n_gt)
            for cat in ("__all__", category):
                self.comp.setdefault((cat, thr), ComponentCounts()).update(r)
            if comp_desc is not None:
                assert len(comp_desc) == n_gt, "component descriptors out of sync with GT"
                for d, hit, biou in zip(comp_desc, r.gt_matched, r.gt_best_iou):
                    self.per_component.append({
                        "name": name, "category": category, "iou_thr": thr,
                        **d.to_dict(), "matched": bool(hit), "best_iou": float(biou),
                    })

    def summary(self) -> list[dict]:
        rows = []
        for cat, pc in sorted(self.pixel.items()):
            row = {"category": cat, **{f"pix_{k}": v for k, v in pc.summary().items()}}
            for thr in self.iou_thrs:
                cs = self.comp[(cat, thr)].summary()
                row.update({f"{k}@{thr}": v for k, v in cs.items()})
            rows.append(row)
        return rows
