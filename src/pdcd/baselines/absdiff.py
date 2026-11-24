"""Training-free baseline: per-pixel colour difference + threshold.

Two difference signals:
    "rgb" : max_c |A_c - B_c|              (0..255)
    "de"  : CIE76 dE in CIELAB             (0..~100+)
Optional Gaussian pre-blur suppresses scanner noise / sub-pixel misregistration,
and an optional minimum component area removes isolated speckles.
The threshold (and blur/min-area) must be selected on the validation split only.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..morphology import delta_e_map, label_components


@dataclass(frozen=True)
class AbsDiffConfig:
    signal: str = "de"        # "de" | "rgb"
    blur_sigma: float = 0.0
    threshold: float = 10.0
    min_area: int = 0


def difference_map(master: np.ndarray, target: np.ndarray, signal: str, blur_sigma: float) -> np.ndarray:
    if blur_sigma > 0:
        master = cv2.GaussianBlur(master, (0, 0), blur_sigma)
        target = cv2.GaussianBlur(target, (0, 0), blur_sigma)
    if signal == "de":
        return delta_e_map(master, target)
    if signal == "rgb":
        return np.abs(master.astype(np.int16) - target.astype(np.int16)).max(-1).astype(np.float32)
    raise ValueError(f"unknown signal: {signal}")


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    lab, n = label_components(mask)
    if n == 0:
        return mask
    areas = np.bincount(lab.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[lab]


def predict_from_diff(diff: np.ndarray, threshold: float, min_area: int = 0) -> np.ndarray:
    return remove_small(diff > threshold, min_area)


def predict(master: np.ndarray, target: np.ndarray, cfg: AbsDiffConfig) -> np.ndarray:
    d = difference_map(master, target, cfg.signal, cfg.blur_sigma)
    return predict_from_diff(d, cfg.threshold, cfg.min_area)
