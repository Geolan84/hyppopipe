from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.models import WeightsEnum

from hyppopipe.train.model_spec import _resolve_torchvision_weights_enum

TransformFn = Callable[[Tensor], Tensor]

_IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)
_IMAGENET_RESIZE = 256
_IMAGENET_CROP = 224


def normalize_stats_for_channels(
    num_channels: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if num_channels <= 0:
        msg = f"num_channels must be positive, got {num_channels}"
        raise ValueError(msg)
    n_ref = len(_IMAGENET_MEAN)
    if num_channels <= n_ref:
        return (
            tuple(_IMAGENET_MEAN[:num_channels]),
            tuple(_IMAGENET_STD[:num_channels]),
        )
    pad_m = list(_IMAGENET_MEAN)
    pad_s = list(_IMAGENET_STD)
    avg_m = sum(pad_m) / n_ref
    avg_s = sum(pad_s) / n_ref
    while len(pad_m) < num_channels:
        pad_m.append(avg_m)
        pad_s.append(avg_s)
    return tuple(pad_m), tuple(pad_s)


def ensure_channel_count(tensor: Tensor, target_channels: int) -> Tensor:
    if target_channels <= 0:
        msg = f"target_channels must be positive, got {target_channels}"
        raise ValueError(msg)
    c = int(tensor.shape[0])
    if c == target_channels:
        return tensor
    device, dtype = tensor.device, tensor.dtype
    h, w = int(tensor.shape[1]), int(tensor.shape[2])
    if c < target_channels:
        if c == 1:
            return tensor.repeat(target_channels, 1, 1)
        reps = (target_channels + c - 1) // c
        stacked = tensor.repeat(reps, 1, 1)
        return stacked[:target_channels].contiguous()
    bucket = (
        (torch.arange(c, device=device, dtype=torch.float64) * target_channels / c)
        .long()
        .clamp(0, target_channels - 1)
    )
    summed = torch.zeros(target_channels, h, w, device=device, dtype=dtype)
    counts = torch.zeros(target_channels, device=device, dtype=dtype)
    for k in range(c):
        b = int(bucket[k].item())
        summed[b] += tensor[k]
        counts[b] += 1.0
    return summed / counts.view(-1, 1, 1).clamp(min=1.0)


def normalize_tensor(
    tensor: Tensor, mean: tuple[float, ...], std: tuple[float, ...]
) -> Tensor:
    c = int(tensor.shape[0])
    if len(mean) != c or len(std) != c:
        mean, std = normalize_stats_for_channels(c)
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(c, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(c, 1, 1)
    return (tensor - mean_t) / std_t


def normalize_tensor_imagenet_style(tensor: Tensor) -> Tensor:
    c = int(tensor.shape[0])
    mean, std = normalize_stats_for_channels(c)
    return normalize_tensor(tensor, mean, std)


def _to_float01(tensor: Tensor) -> Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0)
    x = tensor.float()
    if x.max() > 1.5:
        return x.div_(255.0)
    return x


def _int_size(value: int | list[int] | tuple[int, ...]) -> int:
    if isinstance(value, int):
        return value
    return int(value[0])


def transform_spec_from_weights(weights: WeightsEnum) -> dict[str, Any]:
    tf = weights.transforms()
    weights_fqn = (
        f"{weights.__class__.__module__}."
        f"{weights.__class__.__qualname__}.{weights.name}"
    )
    return {
        "kind": "torchvision_weights",
        "weights_enum": weights_fqn,
        "crop_size": _int_size(tf.crop_size),
        "resize_size": _int_size(tf.resize_size),
        "mean": tuple(float(x) for x in tf.mean),
        "std": tuple(float(x) for x in tf.std),
    }


def default_transform_spec(*, canonical_channels: int = 3) -> dict[str, Any]:
    mean, std = normalize_stats_for_channels(canonical_channels)
    return {
        "kind": "builtin",
        "crop_size": _IMAGENET_CROP,
        "resize_size": _IMAGENET_RESIZE,
        "mean": mean,
        "std": std,
    }


def _compose_train_from_spec(
    spec: dict[str, Any], canonical_channels: int
) -> TransformFn:
    crop = int(spec["crop_size"])
    mean = tuple(float(x) for x in spec["mean"])
    std = tuple(float(x) for x in spec["std"])

    def _apply(tensor: Tensor) -> Tensor:
        x = _to_float01(tensor)
        x = ensure_channel_count(x, canonical_channels)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            x, scale=(0.08, 1.0), ratio=(0.75, 1.333)
        )
        x = transforms.functional.crop(x, i, j, h, w)
        x = transforms.functional.resize(x, [crop, crop], antialias=True)
        return normalize_tensor(x, mean, std)

    return _apply


def _compose_val_from_spec(
    spec: dict[str, Any], canonical_channels: int
) -> TransformFn:
    kind = spec.get("kind")
    if kind == "torchvision_weights":
        weights = _resolve_torchvision_weights_enum(str(spec["weights_enum"]))
        eval_tf = weights.transforms()

        def _apply_eval(tensor: Tensor) -> Tensor:
            x = _to_float01(tensor)
            x = ensure_channel_count(x, canonical_channels)
            return eval_tf(x)

        return _apply_eval

    crop = int(spec["crop_size"])
    resize = int(spec["resize_size"])
    mean = tuple(float(x) for x in spec["mean"])
    std = tuple(float(x) for x in spec["std"])

    def _apply_builtin(tensor: Tensor) -> Tensor:
        x = _to_float01(tensor)
        x = ensure_channel_count(x, canonical_channels)
        x = transforms.functional.resize(x, resize, antialias=True)
        x = transforms.functional.center_crop(x, crop)
        return normalize_tensor(x, mean, std)

    return _apply_builtin


def classification_transform_from_spec(
    spec: dict[str, Any] | None,
    *,
    canonical_channels: int,
    train: bool,
) -> TransformFn:
    resolved = (
        spec
        if spec is not None
        else default_transform_spec(canonical_channels=canonical_channels)
    )
    if train:
        return _compose_train_from_spec(resolved, canonical_channels)
    return _compose_val_from_spec(resolved, canonical_channels)


def classification_transforms_for_weights(
    weights: WeightsEnum,
    *,
    canonical_channels: int,
) -> tuple[TransformFn, TransformFn, dict[str, Any]]:
    spec = transform_spec_from_weights(weights)
    train_tf = _compose_train_from_spec(spec, canonical_channels)
    val_tf = _compose_val_from_spec(spec, canonical_channels)
    return train_tf, val_tf, spec


def default_classification_transform(
    *,
    resize: tuple[int, int] = (_IMAGENET_CROP, _IMAGENET_CROP),
    canonical_channels: int,
) -> transforms.Compose:
    spec = default_transform_spec(canonical_channels=canonical_channels)
    if resize != (_IMAGENET_CROP, _IMAGENET_CROP):
        spec = {**spec, "crop_size": resize[0], "resize_size": resize[0]}
    val_fn = _compose_val_from_spec(spec, canonical_channels)
    return transforms.Compose([transforms.Lambda(val_fn)])
