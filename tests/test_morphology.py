"""Thickness estimator sanity check on synthetic shapes of known width (1-8 px)."""
import cv2
import numpy as np
import pytest

from pdcd.morphology import describe_components

H = W = 128


def _line(width, kind):
    m = np.zeros((H, W), np.uint8)
    if kind == "horizontal":
        m[60:60 + width, 20:100] = 1
    elif kind == "vertical":
        m[20:100, 60:60 + width] = 1
    elif kind == "diagonal":
        # draw a 45-degree band with perpendicular width ~ `width`
        yy, xx = np.mgrid[:H, :W]
        d = np.abs(yy - xx) / np.sqrt(2)
        m[(d < width / 2) & (xx > 20) & (xx < 100)] = 1
    return m.astype(bool)


@pytest.mark.parametrize("width", range(1, 9))
@pytest.mark.parametrize("kind", ["horizontal", "vertical"])
def test_axis_aligned_lines_exact(width, kind):
    (c,) = describe_components(_line(width, kind))
    assert abs(c.thickness - width) <= 0.5, (kind, width, c.thickness)


@pytest.mark.parametrize("width", range(2, 9))
def test_diagonal_lines_close(width):
    (c,) = describe_components(_line(width, "diagonal"))
    assert abs(c.thickness - width) <= 1.5, (width, c.thickness)


@pytest.mark.parametrize("radius", [2, 3, 5, 8])
def test_disk_thickness_is_inscribed_diameter(radius):
    # A digitised disk is jagged, so its inscribed circle is ~20% smaller than the
    # nominal diameter 2r+1. Thickness = inscribed diameter, so accept [1.25r, 2r+1].
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (64, 64), radius, 1, -1)
    (c,) = describe_components(m.astype(bool))
    assert 1.25 * radius <= c.thickness <= 2 * radius + 1, (radius, c.thickness)


def test_single_pixel():
    m = np.zeros((H, W), bool); m[10, 10] = True
    (c,) = describe_components(m)
    assert c.thickness == 1.0 and c.area == 1


def test_delta_e_component_mean():
    m = np.zeros((H, W), bool); m[10:20, 10:20] = True
    de = np.zeros((H, W), np.float32); de[m] = 7.0
    (c,) = describe_components(m, de)
    assert c.delta_e == pytest.approx(7.0)
