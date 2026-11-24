"""Per-component morphology and contrast descriptors.

Thickness estimate (validated on synthetic masks, see tests/test_morphology.py):
    T_i = max(1, P90_{p in skeleton(U_i)} D_U(p))
where U_i is C_i upsampled 2x (nearest) and D_U its Euclidean distance transform
to the nearest background pixel. Exact for axis-aligned widths 1-8 px.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.morphology import skeletonize

CONNECTIVITY = 8


@dataclass
class Component:
    label: int
    area: int
    thickness: float
    length: float          # skeleton pixel count (proxy for length)
    bbox_h: int
    bbox_w: int
    delta_e: float         # mean CIE76 dE between master and target inside component
    cy: float
    cx: float

    def to_dict(self) -> dict:
        return asdict(self)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labelling. Returns (labels int32, num_components)."""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=CONNECTIVITY)
    return labels.astype(np.int32), n - 1


def delta_e_map(master_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    """Per-pixel CIE76 colour difference between two uint8 RGB images."""
    lab_a = rgb2lab(master_rgb)
    lab_b = rgb2lab(target_rgb)
    return np.sqrt(((lab_a - lab_b) ** 2).sum(-1)).astype(np.float32)


def component_thickness(comp: np.ndarray) -> tuple[float, float]:
    """Thickness and skeleton length of a single boolean component (cropped).

    Computed on a 2x nearest-upsampled crop: a width-w structure becomes 2w
    (always even), whose central skeleton pixels have EDT exactly w. On the
    original grid EDT alone cannot separate odd/even widths (2D-1 is exact for
    odd w but under-estimates even w by 1 px).
    """
    up = np.pad(np.repeat(np.repeat(comp, 2, 0), 2, 1), 1)
    dist = ndimage.distance_transform_edt(up)
    skel = skeletonize(up)
    vals = dist[skel]
    if vals.size == 0:  # degenerate; fall back to max distance
        vals = np.array([dist.max()])
    t = max(1.0, float(np.percentile(vals, 90)))
    return t, float(skel.sum()) / 2.0


def describe_components(mask: np.ndarray, de_map: np.ndarray | None = None) -> list[Component]:
    labels, n = label_components(mask)
    out: list[Component] = []
    if n == 0:
        return out
    slices = ndimage.find_objects(labels)
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        comp = labels[sl] == idx
        area = int(comp.sum())
        t, length = component_thickness(comp)
        ys, xs = np.nonzero(comp)
        de = float(de_map[sl][comp].mean()) if de_map is not None else float("nan")
        out.append(Component(
            label=idx, area=area, thickness=t, length=float(length),
            bbox_h=sl[0].stop - sl[0].start, bbox_w=sl[1].stop - sl[1].start,
            delta_e=de, cy=float(ys.mean() + sl[0].start), cx=float(xs.mean() + sl[1].start),
        ))
    return out
