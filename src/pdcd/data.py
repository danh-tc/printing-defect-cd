"""Dataset indexing for Taktpixel2025PD-CD.

Expected layout (Zenodo release):
    <root>/{train,val,test}/A/*.png    master (reference)
    <root>/{train,val,test}/B/*.png    target (inspected print)
    <root>/{train,val,test}/OUT/*.png  binary defect mask

The defect category is encoded as the first token of the file name,
e.g. ``blackspot_proofrog_a4pera_back_00003_0012.png`` -> ``blackspot``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Canonical names for the 6 categories; raw filename prefixes are mapped here.
CATEGORY_ALIASES = {
    "blackspot": "black_spot",
    "black_spot": "black_spot",
    "colorshift": "color_shift",
    "color_shift": "color_shift",
    "friction": "friction",
    "hair": "hair",
    "line": "line",
    "pinhole": "pinhole",
}


@dataclass(frozen=True)
class Sample:
    split: str
    name: str          # file name, identical in A/B/OUT
    master: Path
    target: Path
    mask: Path

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @property
    def category(self) -> str:
        return parse_category(self.name)


def parse_category(name: str) -> str:
    stem = Path(name).stem.lower()
    for key in sorted(CATEGORY_ALIASES, key=len, reverse=True):
        if stem.startswith(key):
            return CATEGORY_ALIASES[key]
    return stem.split("_")[0] or "unknown"


def _list_images(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    return {p.name: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS}


def index_split(root: str | Path, split: str, mask_dir: str = "OUT") -> list[Sample]:
    """Return samples present in all three folders; raise if folders disagree."""
    base = Path(root) / split
    a, b, m = (_list_images(base / d) for d in ("A", "B", mask_dir))
    common = sorted(set(a) & set(b) & set(m))
    missing = (set(a) | set(b) | set(m)) - set(common)
    if missing:
        raise ValueError(
            f"[{split}] {len(missing)} files not present in all of A/B/{mask_dir}, "
            f"e.g. {sorted(missing)[:5]}"
        )
    return [Sample(split, n, a[n], b[n], m[n]) for n in common]


def index_dataset(root: str | Path, splits=SPLITS, mask_dir: str = "OUT") -> dict[str, list[Sample]]:
    return {s: index_split(root, s, mask_dir) for s in splits}


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    """Read a mask as bool. Any non-zero pixel is foreground."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Cannot read mask: {path}")
    return m > 0


def load_sample(s: Sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return read_rgb(s.master), read_rgb(s.target), read_mask(s.mask)
