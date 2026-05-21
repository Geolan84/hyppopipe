"""Image transforms and specs for classification training and inference.

Builds train/validation pipelines from torchvision weight presets or portable
JSON-like ``transform_spec`` dicts, with channel adaptation for non-RGB inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.models import WeightsEnum
from torchvision.transforms import functional as F

from hyppopipe.train.model_spec import _resolve_torchvision_weights_enum

TransformFn = Callable[[Tensor], Tensor]

_IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)
_IMAGENET_RESIZE = 256
_IMAGENET_CROP = 224


def normalize_stats_for_channels(
    num_channels: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ImageNet-style mean and std tuples for ``num_channels``.

    For fewer than three channels, the first channels use ImageNet stats; for
    more, means and stds are padded by repeating the average channel values.

    Args:
        num_channels: Number of input channels (must be positive).

    Returns:
        Pair ``(mean, std)``, each a tuple of length ``num_channels``.

    Raises:
        ValueError: If ``num_channels`` is not positive.
    """
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
    """Resize channel dimension of a CHW tensor to ``target_channels``.

    Replicates or pools channels when the input count differs from the target,
    preserving spatial size ``H x W``.

    Args:
        tensor: Float or byte tensor of shape ``(C, H, W)``.
        target_channels: Desired channel count (must be positive).

    Returns:
        Tensor of shape ``(target_channels, H, W)``.

    Raises:
        ValueError: If ``target_channels`` is not positive.
    """
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
    """Apply per-channel normalization ``(x - mean) / std``.

    If ``mean``/``std`` lengths do not match channel count, stats are recomputed
    via ``normalize_stats_for_channels``.

    Args:
        tensor: CHW tensor to normalize.
        mean: Per-channel means.
        std: Per-channel standard deviations.

    Returns:
        Normalized tensor with the same shape as ``tensor``.
    """
    c = int(tensor.shape[0])
    if len(mean) != c or len(std) != c:
        mean, std = normalize_stats_for_channels(c)
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(c, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(c, 1, 1)
    return (tensor - mean_t) / std_t


def normalize_tensor_imagenet_style(tensor: Tensor) -> Tensor:
    """Normalize a CHW tensor using ImageNet mean/std adapted to its channel count.

    Args:
        tensor: Input image tensor.

    Returns:
        Normalized tensor.
    """
    c = int(tensor.shape[0])
    mean, std = normalize_stats_for_channels(c)
    return normalize_tensor(tensor, mean, std)


def _to_float01(tensor: Tensor) -> Tensor:
    """Convert uint8 or float tensors to float32 in ``[0, 1]``."""
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0)
    x = tensor.float()
    if x.max() > 1.5:
        return x.div_(255.0)
    return x


def _int_size(value: int | list[int] | tuple[int, ...]) -> int:
    """Extract a scalar size from torchvision-style size specs."""
    if isinstance(value, int):
        return value
    return int(value[0])


def transform_spec_from_weights(weights: WeightsEnum) -> dict[str, Any]:
    """Build a portable transform spec from a torchvision ``WeightsEnum``.

    Args:
        weights: Pretrained weights whose ``transforms()`` define crop and norm.

    Returns:
        Dict with ``kind='torchvision_weights'``, FQN, crop/resize sizes, mean, std.
    """
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
    """Default ImageNet-style transform spec when no weights enum is available.

    Args:
        canonical_channels: Channel count for mean/std adaptation.

    Returns:
        Dict with ``kind='builtin'``, resize/crop sizes, and normalized stats.
    """
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
    """Build a training transform: random crop, resize, normalize.

    Args:
        spec: Transform spec from weights or ``default_transform_spec``.
        canonical_channels: Target channel count after ``ensure_channel_count``.

    Returns:
        Callable that maps a CHW tensor to a normalized training tensor.
    """
    crop = int(spec["crop_size"])
    mean = tuple(float(x) for x in spec["mean"])
    std = tuple(float(x) for x in spec["std"])

    def _apply(tensor: Tensor) -> Tensor:
        """Apply random crop, resize, and normalization for training."""
        x = _to_float01(tensor)
        x = ensure_channel_count(x, canonical_channels)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            x, scale=[0.08, 1.0], ratio=[0.75, 1.333]
        )
        x = F.crop(x, i, j, h, w)
        x = F.resize(x, [crop, crop], antialias=True)
        return normalize_tensor(x, mean, std)

    return _apply


def _compose_val_from_spec(
    spec: dict[str, Any], canonical_channels: int
) -> TransformFn:
    """Build a validation transform from a spec (weights eval or center crop).

    Args:
        spec: Transform spec; may reference ``weights_enum`` for eval transforms.
        canonical_channels: Target channel count after ``ensure_channel_count``.

    Returns:
        Callable that maps a CHW tensor to a normalized validation tensor.
    """
    kind = spec.get("kind")
    if kind == "torchvision_weights":
        weights = _resolve_torchvision_weights_enum(str(spec["weights_enum"]))
        eval_tf = weights.transforms()

        def _apply_eval(tensor: Tensor) -> Tensor:
            """Apply torchvision eval transforms from pretrained weights."""
            x = _to_float01(tensor)
            x = ensure_channel_count(x, canonical_channels)
            return eval_tf(x)

        return _apply_eval

    crop = int(spec["crop_size"])
    resize = int(spec["resize_size"])
    mean = tuple(float(x) for x in spec["mean"])
    std = tuple(float(x) for x in spec["std"])

    def _apply_builtin(tensor: Tensor) -> Tensor:
        """Apply resize, center crop, and normalization from a builtin spec."""
        x = _to_float01(tensor)
        x = ensure_channel_count(x, canonical_channels)
        x = F.resize(x, [resize, resize], antialias=True)
        x = F.center_crop(x, [crop, crop])
        return normalize_tensor(x, mean, std)

    return _apply_builtin


def classification_transform_from_spec(
    spec: dict[str, Any] | None,
    *,
    canonical_channels: int,
    train: bool,
) -> TransformFn:
    """Return train or validation transform callable from a spec (or defaults).

    Args:
        spec: Optional transform spec; uses ``default_transform_spec`` if ``None``.
        canonical_channels: Channel count for ``ensure_channel_count``.
        train: If ``True``, use training augmentation; otherwise validation pipeline.

    Returns:
        Per-sample transform callable for ``Dataset`` wrappers.
    """
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
    """Build train/val transforms and spec from a ``WeightsEnum``.

    Args:
        weights: Pretrained weights defining eval transforms and normalization.
        canonical_channels: Channel count for channel adaptation.

    Returns:
        Tuple of train transform, val transform, and serializable spec dict.
    """
    spec = transform_spec_from_weights(weights)
    train_tf = _compose_train_from_spec(spec, canonical_channels)
    val_tf = _compose_val_from_spec(spec, canonical_channels)
    return train_tf, val_tf, spec


def default_classification_transform(
    *,
    resize: tuple[int, int] = (_IMAGENET_CROP, _IMAGENET_CROP),
    canonical_channels: int,
) -> transforms.Compose:
    """Legacy ``Compose`` wrapper around the builtin validation transform.

    Args:
        resize: Optional ``(H, W)`` override for crop and resize sizes in the spec.
        canonical_channels: Channel count for normalization stats.

    Returns:
        ``torchvision.transforms.Compose`` applying the validation pipeline.
    """
    spec = default_transform_spec(canonical_channels=canonical_channels)
    if resize != (_IMAGENET_CROP, _IMAGENET_CROP):
        spec = {**spec, "crop_size": resize[0], "resize_size": resize[0]}
    val_fn = _compose_val_from_spec(spec, canonical_channels)
    return transforms.Compose([transforms.Lambda(val_fn)])
