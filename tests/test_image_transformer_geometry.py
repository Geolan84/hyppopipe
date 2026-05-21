"""Tests for geometry and morphology steps on :class:`ImageTransformer`."""

import cv2
import numpy as np
import torch
from torch import from_numpy

from hyppopipe.pipeline.image.transform import ImageTransformer


def _disk_chw_uint8(
    size: int = 64,
    radius: int = 22,
    center: tuple[int, int] = (32, 32),
    color_bgr: tuple[int, int, int] = (40, 180, 220),
) -> torch.Tensor:
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(bgr, center, radius, color_bgr, thickness=-1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return from_numpy(rgb.copy()).permute(2, 0, 1)


def test_contour_mask_zeros_background_outside_disk() -> None:
    tensor = _disk_chw_uint8()
    out = ImageTransformer().contour_mask(
        threshold=20,
        min_contour_area_ratio=0.01,
    )(tensor)
    assert out.shape == tensor.shape
    corners = out[:, 0, 0].tolist() + out[:, -1, -1].tolist()
    assert all(v == 0 for v in corners)


def test_ellipse_crop_shrinks_to_roi() -> None:
    tensor = _disk_chw_uint8(size=80, radius=28)
    out = ImageTransformer().ellipse_crop(
        threshold=20,
        min_contour_area_ratio=0.01,
    )(tensor)
    assert out.shape[-1] < tensor.shape[-1]
    assert out.shape[-2] < tensor.shape[-2]


def test_min_area_rect_crop_on_rotated_blob() -> None:
    bgr = np.zeros((80, 80, 3), dtype=np.uint8)
    rect = ((40, 40), (50, 20), 30)
    box = cv2.boxPoints(rect).astype(int)
    cv2.fillConvexPoly(bgr, box, (200, 100, 50))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = from_numpy(rgb.copy()).permute(2, 0, 1)
    out = ImageTransformer().min_area_rect_crop(
        threshold=10,
        min_contour_area_ratio=0.01,
        reject_frame_contour=False,
    )(tensor)
    assert out.shape[-2] <= 55
    assert out.shape[-1] <= 55


def test_morphology_close_runs_on_color_tensor() -> None:
    tensor = _disk_chw_uint8()
    out = ImageTransformer().morphology(op="close", kernel_size=5)(tensor)
    assert out.shape == tensor.shape


def test_remove_small_components_keeps_large_blob() -> None:
    tensor = _disk_chw_uint8()
    out = ImageTransformer().remove_small_components(
        threshold=20,
        min_area=100,
    )(tensor)
    assert int(out.max()) > 0


def test_flood_fill_from_center() -> None:
    tensor = _disk_chw_uint8()
    out = ImageTransformer().flood_fill(
        lo_diff=(80, 80, 80),
        hi_diff=(80, 80, 80),
        new_color=(0, 0, 0),
    )(tensor)
    assert out.shape == tensor.shape
