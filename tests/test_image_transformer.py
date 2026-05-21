"""Tests for :class:`~hyppopipe.pipeline.image.transform.ImageTransformer`."""

import logging

import numpy as np
import pytest
import torch

from hyppopipe.pipeline.image.transform import ImageTransformer


def test_opencv_accepts_rgba_chw() -> None:
    tensor = torch.randint(0, 256, (4, 16, 16), dtype=torch.uint8)
    out = ImageTransformer().opencv(lambda img: img)(tensor)
    assert out.shape == (3, 16, 16)


def test_opencv_skips_invalid_tensor_shape(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    tensor = torch.randint(0, 256, (3, 8, 8, 8), dtype=torch.uint8)
    out = ImageTransformer().opencv(lambda img: img)(tensor)
    assert out is tensor
    assert "invalid input tensor" in caplog.text


def test_opencv_skips_failed_callable(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    tensor = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)

    def boom(_img: np.ndarray) -> np.ndarray:
        raise RuntimeError("opencv failed")

    out = ImageTransformer().opencv(boom)(tensor)
    assert torch.equal(out, tensor)
    assert "OpenCV callable failed" in caplog.text


def test_circle_crop_no_contour_warns_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    tensor = torch.zeros(3, 32, 32, dtype=torch.uint8)
    out = ImageTransformer().circle_crop().resize(8)(tensor)
    assert "keeping previous image" in caplog.text
    assert out.shape[-2:] == (8, 8)


def test_circle_crop_rejects_full_frame_contour(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    tensor = torch.full((3, 64, 64), 255, dtype=torch.uint8)
    out = ImageTransformer().circle_crop()(tensor)
    assert torch.equal(out, tensor)
    assert "no contour matched filters" in caplog.text


def test_opencv_bad_return_skips_step(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    tensor = torch.randint(0, 256, (3, 12, 12), dtype=torch.uint8)
    out = ImageTransformer().opencv(lambda _img: "not an array")(tensor)
    assert torch.equal(out, tensor)
    assert "expected numpy.ndarray" in caplog.text
