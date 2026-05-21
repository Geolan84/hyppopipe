"""Train/validation data transforms configured on :class:`~hyppopipe.train.trainer.Trainer`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F
from torch import Tensor
from torchvision.models import WeightsEnum
from torchvision.transforms import v2 as tv_v2

from hyppopipe.train.tasks.classification_transforms import (
    TransformFn,
    _to_float01,
    classification_transform_from_spec,
    classification_transforms_for_weights,
    default_transform_spec,
    ensure_channel_count,
    normalize_tensor_imagenet_style,
    transform_spec_from_weights,
)

ImageTransformFn = Callable[[Tensor], Tensor]


def _spatial_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    return tuple(image_size)


def wrap_v2_classification_transform(
    transform: tv_v2.Transform,
    *,
    canonical_channels: int,
    image_size: int | tuple[int, int] = 224,
) -> TransformFn:
    """Wrap a v2 transform so inputs are float CHW, resized for batching, then normalized.

    Medical/fundus images often differ in resolution; without a resize/crop step,
    ``DataLoader`` cannot stack a batch.
    """
    hw = _spatial_size(image_size)

    def apply(tensor: Tensor) -> Tensor:
        x = _to_float01(tensor)
        x = ensure_channel_count(x, canonical_channels)
        if tuple(x.shape[-2:]) != hw:
            x = F.interpolate(
                x.unsqueeze(0),
                size=hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        x = transform(x)
        return normalize_tensor_imagenet_style(x)

    return apply


def coerce_classification_transform_fn(
    transform: TransformFn | tv_v2.Transform | Any,
    *,
    canonical_channels: int,
    image_size: int | tuple[int, int] = 224,
) -> TransformFn:
    """Return a ``TransformFn``, wrapping torchvision v2 transforms when needed."""
    if isinstance(transform, tv_v2.Transform):
        return wrap_v2_classification_transform(
            transform,
            canonical_channels=canonical_channels,
            image_size=image_size,
        )
    if not callable(transform):
        msg = f"Expected a transform callable or torchvision.transforms.v2.Transform, got {type(transform)!r}"
        raise TypeError(msg)
    return transform


@dataclass(frozen=True, slots=True)
class ClassificationTransforms:
    """Image-only train/val transforms for classification; spec is stored for inference."""

    train: TransformFn
    val: TransformFn
    transform_spec: dict[str, Any] | None = None

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        *,
        canonical_channels: int,
    ) -> ClassificationTransforms:
        """Build transforms from a serializable spec dict."""
        return cls(
            train=classification_transform_from_spec(
                spec, canonical_channels=canonical_channels, train=True
            ),
            val=classification_transform_from_spec(
                spec, canonical_channels=canonical_channels, train=False
            ),
            transform_spec=dict(spec),
        )

    @classmethod
    def default(cls, *, canonical_channels: int = 3) -> ClassificationTransforms:
        """Default augmentation and validation pipeline for classification."""
        spec = default_transform_spec(canonical_channels=canonical_channels)
        return cls.from_spec(spec, canonical_channels=canonical_channels)

    @classmethod
    def from_weights(
        cls,
        weights: WeightsEnum,
        *,
        canonical_channels: int,
    ) -> ClassificationTransforms:
        """Align train/val transforms with a torchvision ``WeightsEnum`` preset."""
        train_tf, val_tf, spec = classification_transforms_for_weights(
            weights, canonical_channels=canonical_channels
        )
        return cls(train=train_tf, val=val_tf, transform_spec=spec)

    @classmethod
    def from_v2(
        cls,
        train: tv_v2.Transform | None = None,
        val: tv_v2.Transform | None = None,
        *,
        image_size: int | tuple[int, int] = 224,
        canonical_channels: int = 3,
    ) -> ClassificationTransforms:
        """Build transforms from v2 stages; resizes to ``image_size`` before augmentation."""
        if train is None:
            train = tv_v2.Identity()
        if val is None:
            val = tv_v2.Identity()
        spec = default_transform_spec(canonical_channels=canonical_channels)
        spec = {**spec, "crop_size": _spatial_size(image_size)[0]}
        return cls(
            train=wrap_v2_classification_transform(
                train,
                canonical_channels=canonical_channels,
                image_size=image_size,
            ),
            val=wrap_v2_classification_transform(
                val,
                canonical_channels=canonical_channels,
                image_size=image_size,
            ),
            transform_spec=spec,
        )


@dataclass(frozen=True, slots=True)
class DetectionTransforms:
    """Optional image-only transforms for detection training (boxes are not updated)."""

    train: ImageTransformFn | None = None
    val: ImageTransformFn | None = None


@dataclass(frozen=True, slots=True)
class SegmentationTransforms:
    """Optional image-only transforms for segmentation training."""

    train: ImageTransformFn | None = None
    val: ImageTransformFn | None = None


def classification_transform_spec_for_inference(
    transforms: ClassificationTransforms | None,
    *,
    weights_enum: WeightsEnum | None,
    canonical_channels: int,
) -> dict[str, Any] | None:
    """Resolve transform spec to persist in classification inference metadata."""
    if transforms is not None and transforms.transform_spec is not None:
        return dict(transforms.transform_spec)
    if weights_enum is not None:
        return transform_spec_from_weights(weights_enum)
    if transforms is not None:
        return None
    return default_transform_spec(canonical_channels=canonical_channels)
