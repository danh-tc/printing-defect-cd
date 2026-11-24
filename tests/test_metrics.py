import numpy as np
import pytest

from pdcd.metrics import Evaluator, PixelCounts, match_components


def blank():
    return np.zeros((64, 64), bool)


def test_pixel_perfect():
    g = blank(); g[5:10, 5:10] = True
    pc = PixelCounts(); pc.update(g, g)
    s = pc.summary()
    assert s["f1"] == pytest.approx(1.0) and s["iou"] == pytest.approx(1.0)


def test_component_match_and_fp():
    g = blank(); g[5:10, 5:10] = True; g[40:42, 40:60] = True   # 2 GT comps
    p = blank(); p[5:10, 5:10] = True; p[30:33, 5:8] = True     # 1 hit, 1 FP, 1 miss
    r = match_components(p, g, thr=0.3)
    assert r.n_gt == 2 and r.n_pred == 2
    assert r.gt_matched.tolist() == [True, False]
    assert r.n_fp == 1


def test_iou_threshold_applies():
    g = blank(); g[0:10, 0:10] = True            # area 100
    p = blank(); p[0:10, 0:2] = True             # IoU 0.2
    assert match_components(p, g, 0.1).gt_matched[0]
    assert not match_components(p, g, 0.3).gt_matched[0]


def test_one_to_one():
    g = blank(); g[0:10, 0:10] = True
    p = blank(); p[0:10, 0:4] = True; p[0:10, 6:10] = True    # fragmented prediction
    r = match_components(p, g, thr=0.3)
    assert r.gt_matched.sum() == 1 and r.n_pred_matched == 1 and r.n_fp == 1


def test_empty_cases():
    r = match_components(blank(), blank())
    assert r.n_gt == 0 and r.n_pred == 0 and r.n_fp == 0
    g = blank(); g[1, 1] = True
    r = match_components(blank(), g)
    assert r.gt_matched.tolist() == [False]


def test_evaluator_per_category():
    ev = Evaluator(iou_thrs=(0.3,))
    g = blank(); g[5:10, 5:10] = True
    ev.update(g, g, "hair", "a.png")
    ev.update(blank(), g, "line", "b.png")
    rows = {r["category"]: r for r in ev.summary()}
    assert rows["hair"]["comp_recall@0.3"] == pytest.approx(1.0)
    assert rows["line"]["comp_recall@0.3"] == pytest.approx(0.0)
    assert rows["__all__"]["comp_recall@0.3"] == pytest.approx(0.5)
