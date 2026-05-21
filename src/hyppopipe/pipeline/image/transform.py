"""Composable torchvision and OpenCV transforms for :class:`~hyppopipe.data.image.Image`."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from numbers import Number
from typing import Any, Literal, Self

import cv2
import numpy as np
from torch import Tensor, from_numpy, uint8
from torch import dtype as torch_dtype
from torchvision.transforms import v2 as transforms

from hyppopipe.data.image import Image

logger = logging.getLogger(__name__)

TransformStep = Callable[[Tensor], Tensor]
OpenCvFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class ContourSelectConfig:
    """Contour detection and filtering for OpenCV contour-based steps.

    Used by :meth:`ImageTransformer.ellipse_crop`, :meth:`ImageTransformer.min_area_rect_crop`,
    :meth:`ImageTransformer.convex_hull_mask`, and :meth:`ImageTransformer.contour_mask`.

    Attributes:
        threshold: Grayscale binarization level (higher → fewer faint regions).
        min_contour_area_ratio: Ignore contours smaller than this fraction of the image.
        max_contour_area_ratio: Ignore contours larger than this (often the image frame).
        reject_frame_contour: Skip contours whose bounding box touches all four edges.
        frame_margin_px: Edge tolerance when detecting a frame-filling contour.
    """

    threshold: int = 10
    min_contour_area_ratio: float = 0.02
    max_contour_area_ratio: float = 0.90
    reject_frame_contour: bool = True
    frame_margin_px: int = 3


@dataclass(frozen=True, slots=True)
class CircleCropConfig(ContourSelectConfig):
    """:class:`ContourSelectConfig` plus circle-specific filters for :meth:`ImageTransformer.circle_crop`.

    Attributes:
        min_radius_ratio: Minimum enclosing-circle radius vs ``min(H, W)``.
        max_radius_ratio: Maximum radius vs ``min(H, W)`` (limits full-frame squares).
        min_circularity: ``4π·area/perimeter²``; ``0`` disables the check.
    """

    min_radius_ratio: float = 0.05
    max_radius_ratio: float = 0.45
    min_circularity: float = 0.0


@dataclass(frozen=True, slots=True)
class MaskBackgroundConfig:
    """Background applied outside a contour or component mask.

    Attributes:
        value: ``(B, G, R)`` for color images, or a single intensity for grayscale.
    """

    value: tuple[int, int, int] | int = 0


@dataclass(frozen=True, slots=True)
class MorphologyConfig:
    """Tuning for :meth:`ImageTransformer.morphology`."""

    op: Literal[
        "erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"
    ] = "close"
    kernel_size: int = 5
    iterations: int = 1
    shape: Literal["ellipse", "rect", "cross"] = "ellipse"
    threshold: int = 10
    binary: bool = False


@dataclass(frozen=True, slots=True)
class RemoveSmallComponentsConfig:
    """Tuning for :meth:`ImageTransformer.remove_small_components`."""

    threshold: int = 10
    min_area: int = 64
    connectivity: Literal[4, 8] = 8


@dataclass(frozen=True, slots=True)
class FloodFillConfig:
    """Tuning for :meth:`ImageTransformer.flood_fill`."""

    seed_xy: tuple[int, int] | None = None
    lo_diff: tuple[int, int, int] = (20, 20, 20)
    hi_diff: tuple[int, int, int] = (20, 20, 20)
    new_color: tuple[int, int, int] | int = 0


MORPH_SHAPE = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}
MORPH_OP = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


class ImageTransformer:
    """Fluent builder that applies transforms in the order they were added.

    Torchvision steps run on CHW tensors; OpenCV steps receive BGR HWC ``uint8``
    arrays (standard OpenCV layout) and are converted automatically.

    OpenCV steps skip themselves on invalid input or failed callables (with a
    warning) and pass the tensor through to the next step.
    """

    __slots__ = ("_steps",)

    def __init__(self) -> None:
        """Start with an empty transform list."""
        self._steps: list[TransformStep] = []

    @classmethod
    def from_compose(cls, compose: transforms.Compose) -> Self:
        """Create a transformer from a ``torchvision.transforms.v2.Compose``.

        Each child transform becomes its own sequential step (order preserved).
        """
        return cls.from_transforms(list(compose.transforms))  # type: ignore[arg-type]

    @classmethod
    def from_transforms(cls, transformations: list[transforms.Transform]) -> Self:
        """Create a transformer from torchvision v2 transforms (one step each)."""
        transformer = cls()
        for transform in transformations:
            transformer.torchvision(transform)
        return transformer

    def _append_torchvision(self, transform: transforms.Transform) -> None:
        def step(tensor: Tensor) -> Tensor:
            return transform(tensor)

        self._steps.append(step)

    def _append_opencv(self, fn: OpenCvFn, *, step_name: str = "opencv") -> None:
        def step(tensor: Tensor) -> Tensor:
            try:
                bgr = _tensor_to_hwc_bgr_uint8(tensor)
            except ValueError as exc:
                logger.warning(
                    "%s: invalid input tensor shape %s — %s",
                    step_name,
                    tuple(tensor.shape),
                    exc,
                )
                return tensor
            try:
                raw_out = fn(bgr)
            except Exception:
                logger.warning(
                    "%s: OpenCV callable failed, keeping previous tensor",
                    step_name,
                    exc_info=True,
                )
                return tensor
            out = _normalize_opencv_output(raw_out, step_name=step_name)
            if out is None:
                return tensor
            try:
                return _hwc_bgr_to_tensor(out, dtype=tensor.dtype)
            except Exception:
                logger.warning(
                    "%s: failed to convert OpenCV output to tensor, keeping previous tensor",
                    step_name,
                    exc_info=True,
                )
                return tensor

        self._steps.append(step)

    def torchvision(self, transform: transforms.Transform) -> Self:
        """Append a ``torchvision.transforms.v2`` transform (runs on CHW tensor)."""
        self._append_torchvision(transform)
        return self

    def opencv(self, fn: OpenCvFn, *, step_name: str = "opencv") -> Self:
        """Append an OpenCV callable (BGR HWC ``uint8`` in/out)."""
        self._append_opencv(fn, step_name=step_name)
        return self

    def add(self, step: TransformStep) -> Self:
        """Append a custom CHW tensor → CHW tensor step."""
        self._steps.append(step)
        return self

    def __call__(self, image: Image | Tensor) -> Image | Tensor:
        """Apply all configured transforms in registration order."""
        original_image = image
        if isinstance(image, Image):
            tensor = image.body
        else:
            tensor = image
        for step in self._steps:
            tensor = step(tensor)
        if isinstance(original_image, Image):
            return replace(original_image, body=tensor)
        return tensor

    def resize(
        self,
        size: int | tuple[int, ...] | list[int],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Append ``torchvision.transforms.v2.Resize``."""
        hw = (size, size) if isinstance(size, int) else tuple(size)
        antialias = kwargs.pop("antialias", True)
        self._append_torchvision(
            transforms.Resize(hw, antialias=antialias, *args, **kwargs)
        )
        return self

    def rotate(self, degrees: float | tuple[float, float]) -> Self:
        """Append ``torchvision.transforms.v2.RandomRotation``."""
        angle_range: tuple[float, float]
        if isinstance(degrees, Number):
            angle_range = (float(degrees), float(degrees))
        else:
            angle_range = (float(degrees[0]), float(degrees[1]))
        self._append_torchvision(transforms.RandomRotation(angle_range))
        return self

    def normalize(self, mean: tuple[float, ...], std: tuple[float, ...]) -> Self:
        """Append ``torchvision.transforms.v2.Normalize``."""
        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length")
        self._append_torchvision(transforms.Normalize(mean, std))
        return self

    def gaussian_blur(
        self,
        kernel_size: int,
        sigma: float | tuple[float, float],
    ) -> Self:
        """Append ``torchvision.transforms.v2.GaussianBlur``."""
        self._append_torchvision(transforms.GaussianBlur(kernel_size, sigma))
        return self

    def sharpen(self, factor: float = 2.0) -> Self:
        """Append ``torchvision.transforms.v2.RandomAdjustSharpness`` with ``p=1``."""
        if factor < 0:
            raise ValueError("factor must be non-negative")
        self._append_torchvision(
            transforms.RandomAdjustSharpness(sharpness_factor=factor, p=1.0)
        )
        return self

    def center_crop(self, size: int | tuple[int, ...] | list[int]) -> Self:
        """Append ``torchvision.transforms.v2.CenterCrop``."""
        hw = (size, size) if isinstance(size, int) else tuple(size)
        self._append_torchvision(transforms.CenterCrop(hw))
        return self

    def circle_crop(
        self, config: CircleCropConfig | None = None, **kwargs: Any
    ) -> Self:
        """Append OpenCV crop around a filtered contour's enclosing circle.

        Args:
            config: Full tuning object; overrides keyword arguments when set.
            **kwargs: Fields for :class:`CircleCropConfig` (e.g. ``threshold=40``).

        Example::

            ImageTransformer().circle_crop(
                threshold=40,
                max_contour_area_ratio=0.75,
                min_circularity=0.5,
            )
        """
        config = _resolve_circle_crop_config(config, kwargs)
        self._append_opencv(_make_circle_crop_fn(config), step_name="circle_crop")
        return self

    def ellipse_crop(
        self,
        config: ContourSelectConfig | None = None,
        **kwargs: Any,
    ) -> Self:
        """Crop axis-aligned square around :func:`cv2.fitEllipse` of the selected contour."""
        config = _resolve_contour_config(config, kwargs)
        self._append_opencv(
            _make_opencv_step(
                lambda img: _try_ellipse_crop_bgr(img, config),
                step_name="ellipse_crop",
            ),
            step_name="ellipse_crop",
        )
        return self

    def min_area_rect_crop(
        self,
        config: ContourSelectConfig | None = None,
        **kwargs: Any,
    ) -> Self:
        """Perspective-crop the oriented minimum-area rectangle of the selected contour."""
        config = _resolve_contour_config(config, kwargs)
        self._append_opencv(
            _make_opencv_step(
                lambda img: _try_min_area_rect_crop_bgr(img, config),
                step_name="min_area_rect_crop",
            ),
            step_name="min_area_rect_crop",
        )
        return self

    def convex_hull_mask(
        self,
        config: ContourSelectConfig | None = None,
        *,
        background: MaskBackgroundConfig | tuple[int, int, int] | int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Zero (or fill) pixels outside the convex hull of the selected contour."""
        config = _resolve_contour_config(config, kwargs)
        bg = _resolve_mask_background(background)
        self._append_opencv(
            _make_opencv_step(
                lambda img: _try_contour_mask_bgr(
                    img, config, convex=True, background=bg
                ),
                step_name="convex_hull_mask",
            ),
            step_name="convex_hull_mask",
        )
        return self

    def contour_mask(
        self,
        config: ContourSelectConfig | None = None,
        *,
        background: MaskBackgroundConfig | tuple[int, int, int] | int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Zero (or fill) pixels outside the selected contour polygon."""
        config = _resolve_contour_config(config, kwargs)
        bg = _resolve_mask_background(background)
        self._append_opencv(
            _make_opencv_step(
                lambda img: _try_contour_mask_bgr(
                    img, config, convex=False, background=bg
                ),
                step_name="contour_mask",
            ),
            step_name="contour_mask",
        )
        return self

    def morphology(
        self,
        config: MorphologyConfig | None = None,
        **kwargs: Any,
    ) -> Self:
        """Apply :func:`cv2.morphologyEx` on grayscale (optionally binarized first)."""
        if config is None:
            config = MorphologyConfig(**kwargs) if kwargs else MorphologyConfig()
        elif kwargs:
            raise ValueError("pass either config or keyword arguments, not both")

        def run(image: np.ndarray) -> np.ndarray:
            return _morphology_bgr(image, config)

        self._append_opencv(run, step_name="morphology")
        return self

    def remove_small_components(
        self,
        config: RemoveSmallComponentsConfig | None = None,
        *,
        background: MaskBackgroundConfig | tuple[int, int, int] | int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Drop connected components smaller than ``min_area`` after thresholding."""
        if config is None:
            config = (
                RemoveSmallComponentsConfig(**kwargs)
                if kwargs
                else RemoveSmallComponentsConfig()
            )
        elif kwargs:
            raise ValueError("pass either config or keyword arguments, not both")
        bg = _resolve_mask_background(background)

        self._append_opencv(
            _make_opencv_step(
                lambda img: _remove_small_components_bgr(img, config, bg),
                step_name="remove_small_components",
            ),
            step_name="remove_small_components",
        )
        return self

    def flood_fill(
        self,
        config: FloodFillConfig | None = None,
        **kwargs: Any,
    ) -> Self:
        """Region growing from ``seed_xy`` (image center when omitted)."""
        if config is None:
            config = FloodFillConfig(**kwargs) if kwargs else FloodFillConfig()
        elif kwargs:
            raise ValueError("pass either config or keyword arguments, not both")

        def run(image: np.ndarray) -> np.ndarray:
            return _flood_fill_bgr(image, config)

        self._append_opencv(run, step_name="flood_fill")
        return self

    def clear(self) -> None:
        """Remove all transforms from the recipe."""
        self._steps.clear()


def _validate_tensor_for_opencv(tensor: Tensor) -> None:
    """Raise ``ValueError`` when a tensor cannot be converted for OpenCV."""
    if tensor.ndim not in (2, 3):
        msg = (
            f"expected 2D or 3D tensor, got {tensor.ndim}D shape {tuple(tensor.shape)}"
        )
        raise ValueError(msg)
    if tensor.numel() == 0:
        raise ValueError("tensor is empty")
    if (
        tensor.ndim == 3
        and tensor.shape[0] not in (1, 3, 4)
        and tensor.shape[-1] not in (1, 3, 4)
    ):
        msg = (
            "expected CHW or HWC with 1, 3, or 4 channels "
            f"(RGBA alpha is ignored for OpenCV), got shape {tuple(tensor.shape)}"
        )
        raise ValueError(msg)


def _tensor_to_hwc_bgr_uint8(tensor: Tensor) -> np.ndarray:
    """CHW (or HWC) tensor → contiguous BGR ``uint8`` HWC for OpenCV."""
    _validate_tensor_for_opencv(tensor)
    x = tensor.detach().cpu()
    if x.dtype != uint8:
        logger.warning(
            "opencv step: input dtype %s, converting to uint8 via clamp",
            x.dtype,
        )
        if x.is_floating_point() and float(x.max()) <= 1.0:
            x = (x.clamp(0.0, 1.0) * 255.0).to(uint8)
        else:
            x = x.clamp(0, 255).to(uint8)
    if x.ndim == 2:
        return np.ascontiguousarray(x.numpy())
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        x = x.permute(2, 0, 1)
    if x.shape[0] == 1:
        return np.ascontiguousarray(x[0].numpy())
    # Drop alpha (4th channel) when present; OpenCV steps operate on BGR only.
    rgb = np.ascontiguousarray(x[:3].permute(1, 2, 0).numpy())
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _normalize_opencv_output(array: Any, *, step_name: str) -> np.ndarray | None:
    """Return a valid BGR/HW array or ``None`` if the callable output is unusable."""
    if not isinstance(array, np.ndarray):
        logger.warning(
            "%s: callable returned %r, expected numpy.ndarray",
            step_name,
            type(array),
        )
        return None
    if array.ndim not in (2, 3):
        logger.warning(
            "%s: callable returned array with ndim=%d, expected 2 or 3",
            step_name,
            array.ndim,
        )
        return None
    if array.size == 0:
        logger.warning("%s: callable returned empty array", step_name)
        return None
    if array.dtype != np.uint8:
        logger.warning(
            "%s: output dtype %s, converting to uint8",
            step_name,
            array.dtype,
        )
        if np.issubdtype(array.dtype, np.floating):
            arr_f = np.asarray(array, dtype=np.float64)
            if arr_f.max() <= 1.0:
                array = (arr_f * 255.0).astype(np.uint8)
            else:
                array = np.clip(arr_f, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _hwc_bgr_to_tensor(array: np.ndarray, *, dtype: torch_dtype) -> Tensor:
    """BGR or 2D ``uint8`` array → CHW tensor matching ``dtype``."""
    arr = np.ascontiguousarray(array)
    if arr.ndim == 2:
        return from_numpy(arr.copy()).to(dtype=dtype)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return from_numpy(arr[:, :, 0].copy()).unsqueeze(0).to(dtype=dtype)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return from_numpy(rgb.copy()).permute(2, 0, 1).to(dtype=dtype)


def _bgr_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _contour_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def _is_frame_contour(
    contour: np.ndarray, height: int, width: int, margin: int
) -> bool:
    x, y, box_w, box_h = cv2.boundingRect(contour)
    return (
        x <= margin
        and y <= margin
        and x + box_w >= width - margin
        and y + box_h >= height - margin
    )


def _select_contour(
    contours: list[np.ndarray],
    *,
    height: int,
    width: int,
    config: ContourSelectConfig,
) -> np.ndarray | None:
    """Pick the largest contour that passes ``config`` filters."""
    image_area = float(height * width)
    min_side = float(min(height, width))
    circle = config if isinstance(config, CircleCropConfig) else None
    best: np.ndarray | None = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        area_ratio = area / image_area
        if area_ratio < config.min_contour_area_ratio:
            continue
        if area_ratio > config.max_contour_area_ratio:
            continue
        if config.reject_frame_contour and _is_frame_contour(
            contour, height, width, config.frame_margin_px
        ):
            continue
        if circle is not None:
            if circle.min_circularity > 0.0:
                if _contour_circularity(contour) < circle.min_circularity:
                    continue
            (_cx, _cy), radius = cv2.minEnclosingCircle(contour)
            radius_ratio = float(radius) / min_side
            if radius_ratio < circle.min_radius_ratio:
                continue
            if radius_ratio > circle.max_radius_ratio:
                continue
        if area > best_area:
            best = contour
            best_area = area
    return best


def _resolve_contour_from_bgr(
    image: np.ndarray,
    config: ContourSelectConfig,
) -> tuple[np.ndarray | None, int, int, str]:
    """Return ``(contour, height, width, skip_reason)``; contour is ``None`` on failure."""
    if image.ndim not in (2, 3):
        return None, 0, 0, "unsupported array dimensions"
    gray = _bgr_to_gray(image)
    _, binary = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, 0, "no contours after threshold"
    height, width = gray.shape[:2]
    contour = _select_contour(contours, height=height, width=width, config=config)
    if contour is None:
        return (
            None,
            height,
            width,
            "no contour matched filters (try relaxing ContourSelectConfig)",
        )
    return contour, height, width, ""


def _pad_bgr_for_crop(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    half_w: int,
    half_h: int,
) -> tuple[np.ndarray, int, int]:
    """Pad so ``[y-half_h:y+half_h, x-half_w:x+half_w]`` fits; return padded image and shifted center."""
    height, width = image.shape[:2]
    pad_left = max(0, half_w - x)
    pad_top = max(0, half_h - y)
    pad_right = max(0, (x + half_w) - width)
    pad_bottom = max(0, (y + half_h) - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
        )
        x, y = x + pad_left, y + pad_top
    return image, x, y


def _order_box_points(box: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(box, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _apply_mask_bgr(
    image: np.ndarray,
    mask: np.ndarray,
    background: MaskBackgroundConfig,
) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        return image
    if background.value == 0:
        return cv2.bitwise_and(image, image, mask=mask)
    out = image.copy()
    if image.ndim == 2:
        bg = (
            int(background.value)
            if isinstance(background.value, int)
            else int(background.value[0])
        )
        out[mask == 0] = bg
        return out
    if isinstance(background.value, int):
        fill = (background.value, background.value, background.value)
    else:
        fill = background.value
    out[mask == 0] = fill
    return out


def _contour_mask_from_bgr(
    image: np.ndarray,
    contour: np.ndarray,
    *,
    convex: bool,
) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if convex:
        cv2.fillConvexPoly(mask, cv2.convexHull(contour), 255)
    else:
        cv2.fillPoly(mask, [contour], 255)
    return mask


def _try_circle_crop_bgr(
    image: np.ndarray,
    config: CircleCropConfig,
) -> tuple[np.ndarray, bool, str]:
    """Return ``(result, applied, reason)``; on skip ``result`` is the input image."""
    output = image.copy()
    contour, height, width, reason = _resolve_contour_from_bgr(image, config)
    if contour is None:
        return image, False, reason
    (x, y), r = cv2.minEnclosingCircle(contour)
    x, y, r = int(x), int(y), int(r)
    if r <= 0:
        return image, False, "enclosing circle radius is zero"
    pad_left = max(0, r - x)
    pad_top = max(0, r - y)
    pad_right = max(0, (x + r) - width)
    pad_bottom = max(0, (y + r) - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        output = cv2.copyMakeBorder(
            output, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT
        )
        x, y = x + pad_left, y + pad_top
    cropped = output[y - r : y + r, x - r : x + r]
    if cropped.size == 0:
        return image, False, "crop region is empty"
    return cropped, True, ""


def _try_ellipse_crop_bgr(
    image: np.ndarray,
    config: ContourSelectConfig,
) -> tuple[np.ndarray, bool, str]:
    contour, _, _, reason = _resolve_contour_from_bgr(image, config)
    if contour is None:
        return image, False, reason
    if len(contour) < 5:
        return image, False, "fitEllipse requires at least 5 contour points"
    center, axes, _angle = cv2.fitEllipse(contour)
    cx, cy = int(center[0]), int(center[1])
    half_w, half_h = int(axes[0] / 2), int(axes[1] / 2)
    if half_w <= 0 or half_h <= 0:
        return image, False, "ellipse axes are zero"
    output, cx, cy = _pad_bgr_for_crop(
        image.copy(), x=cx, y=cy, half_w=half_w, half_h=half_h
    )
    cropped = output[cy - half_h : cy + half_h, cx - half_w : cx + half_w]
    if cropped.size == 0:
        return image, False, "ellipse crop region is empty"
    return cropped, True, ""


def _try_min_area_rect_crop_bgr(
    image: np.ndarray,
    config: ContourSelectConfig,
) -> tuple[np.ndarray, bool, str]:
    contour, _, _, reason = _resolve_contour_from_bgr(image, config)
    if contour is None:
        return image, False, reason
    rect = cv2.minAreaRect(contour)
    box_w, box_h = int(rect[1][0]), int(rect[1][1])
    if box_w <= 0 or box_h <= 0:
        return image, False, "minAreaRect has zero size"
    src = _order_box_points(cv2.boxPoints(rect))
    dst = np.array(
        [[0, box_h - 1], [0, 0], [box_w - 1, 0], [box_w - 1, box_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    cropped = cv2.warpPerspective(image, matrix, (box_w, box_h))
    if cropped.size == 0:
        return image, False, "min area rect crop is empty"
    return cropped, True, ""


def _try_contour_mask_bgr(
    image: np.ndarray,
    config: ContourSelectConfig,
    *,
    convex: bool,
    background: MaskBackgroundConfig,
) -> tuple[np.ndarray, bool, str]:
    contour, _, _, reason = _resolve_contour_from_bgr(image, config)
    if contour is None:
        return image, False, reason
    mask = _contour_mask_from_bgr(image, contour, convex=convex)
    return _apply_mask_bgr(image, mask, background), True, ""


def _morphology_bgr(image: np.ndarray, config: MorphologyConfig) -> np.ndarray:
    gray = _bgr_to_gray(image)
    if config.binary:
        _, gray = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY)
    k = max(1, int(config.kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(MORPH_SHAPE[config.shape], (k, k))
    op = MORPH_OP[config.op]
    if op in (cv2.MORPH_ERODE, cv2.MORPH_DILATE):
        result = cv2.morphologyEx(
            gray, op, kernel, iterations=max(1, config.iterations)
        )
    else:
        result = cv2.morphologyEx(gray, op, kernel, iterations=config.iterations)
    if image.ndim == 2:
        return result
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def _remove_small_components_bgr(
    image: np.ndarray,
    config: RemoveSmallComponentsConfig,
    background: MaskBackgroundConfig,
) -> tuple[np.ndarray, bool, str]:
    gray = _bgr_to_gray(image)
    _, binary = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY)
    connectivity = 4 if config.connectivity == 4 else 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )
    if num_labels <= 1:
        return image, False, "no foreground components"
    mask = np.zeros_like(binary)
    kept = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= config.min_area:
            mask[labels == label] = 255
            kept += 1
    if kept == 0:
        return image, False, "all components below min_area"
    return _apply_mask_bgr(image, mask, background), True, ""


def _flood_fill_bgr(image: np.ndarray, config: FloodFillConfig) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    if config.seed_xy is None:
        seed = (width // 2, height // 2)
    else:
        seed = config.seed_xy
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    if output.ndim == 2:
        new_val = (
            int(config.new_color)
            if isinstance(config.new_color, int)
            else int(config.new_color[0])
        )
        lo = int(config.lo_diff[0])
        hi = int(config.hi_diff[0])
    else:
        if isinstance(config.new_color, int):
            new_val = (config.new_color, config.new_color, config.new_color)
        else:
            new_val = config.new_color
        lo = config.lo_diff
        hi = config.hi_diff
    cv2.floodFill(
        output,
        flood_mask,
        seed,
        new_val,
        loDiff=lo,
        upDiff=hi,
        flags=4,
    )
    return output


def _make_opencv_step(
    fn: Callable[[np.ndarray], tuple[np.ndarray, bool, str]],
    *,
    step_name: str,
) -> OpenCvFn:
    def run(image: np.ndarray) -> np.ndarray:
        result, applied, reason = fn(image)
        if not applied:
            logger.warning("%s: %s, keeping previous image", step_name, reason)
            return image
        return result

    return run


def _make_circle_crop_fn(config: CircleCropConfig) -> OpenCvFn:
    return _make_opencv_step(
        lambda img: _try_circle_crop_bgr(img, config), step_name="circle_crop"
    )


def _resolve_contour_config(
    config: ContourSelectConfig | None,
    kwargs: dict[str, Any],
) -> ContourSelectConfig:
    if config is None:
        return ContourSelectConfig(**kwargs) if kwargs else ContourSelectConfig()
    if kwargs:
        raise ValueError("pass either config or keyword arguments, not both")
    return config


def _resolve_circle_crop_config(
    config: CircleCropConfig | None,
    kwargs: dict[str, Any],
) -> CircleCropConfig:
    if config is None:
        return CircleCropConfig(**kwargs) if kwargs else CircleCropConfig()
    if kwargs:
        raise ValueError("pass either config or keyword arguments, not both")
    return config


def _resolve_mask_background(
    background: MaskBackgroundConfig | tuple[int, int, int] | int | None,
) -> MaskBackgroundConfig:
    if background is None:
        return MaskBackgroundConfig()
    if isinstance(background, MaskBackgroundConfig):
        return background
    return MaskBackgroundConfig(value=background)
