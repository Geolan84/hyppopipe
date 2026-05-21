"""Adapt torchvision classification backbones for custom classes and channel counts.

Locates stem convolutions and classification heads across common architectures
(ResNet, ViT, EfficientNet-style) and rebuilds them for training and inference.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Linear, Module, Sequential
from torch.nn import Conv2d


def _resize_conv1_in_channels(weight: Tensor, new_in: int) -> Tensor:
    """Resample convolution weights when changing the input channel count.

    Fewer channels: average old channels into buckets. More channels: pick
    representative source channels by uniform indexing.

    Args:
        weight: 4D conv weight ``(out_ch, in_ch, kh, kw)``.
        new_in: Target input channel count.

    Returns:
        Resized weight tensor of shape ``(out_ch, new_in, kh, kw)``.
    """
    old_in = weight.shape[1]
    if old_in == new_in:
        return weight.clone()
    out_ch, _, kh, kw = weight.shape
    device, dtype = weight.device, weight.dtype
    if new_in < old_in:
        bucket = (
            (torch.arange(old_in, device=device, dtype=torch.float64) * new_in / old_in)
            .long()
            .clamp(0, new_in - 1)
        )
        summed = torch.zeros(out_ch, new_in, kh, kw, device=device, dtype=dtype)
        counts = torch.zeros(new_in, device=device, dtype=dtype)
        for k in range(old_in):
            b = int(bucket[k].item())
            summed[:, b] += weight[:, k]
            counts[b] += 1.0
        return summed / counts.clamp(min=1.0).view(1, new_in, 1, 1)
    pick = (
        (torch.arange(new_in, device=device, dtype=torch.float64) * old_in / new_in)
        .long()
        .clamp(0, old_in - 1)
    )
    return torch.stack([weight[:, int(pick[j].item())] for j in range(new_in)], dim=1)


def find_stem_conv(model: Module) -> Conv2d | None:
    """Find the first stem ``Conv2d`` in common torchvision classifiers.

    Args:
        model: Classification backbone.

    Returns:
        Stem convolution if found (``conv1``, ``conv_proj``, or first in ``features``).
    """
    conv1 = getattr(model, "conv1", None)
    if isinstance(conv1, Conv2d):
        return conv1
    conv_proj = getattr(model, "conv_proj", None)
    if isinstance(conv_proj, Conv2d):
        return conv_proj
    features = getattr(model, "features", None)
    if features is not None and len(features) > 0:
        first = features[0]
        if isinstance(first, Conv2d):
            return first
        if isinstance(first, Sequential) and len(first) > 0:
            inner = first[0]
            if isinstance(inner, Conv2d):
                return inner
    return None


def classifier_output_features(model: Module) -> int | None:
    """Return the number of logits from the classification head, if detectable.

    Args:
        model: Classification backbone.

    Returns:
        Output feature size, or ``None`` if no known head layout is found.
    """
    fc = getattr(model, "fc", None)
    if isinstance(fc, Linear):
        return int(fc.out_features)
    heads = getattr(model, "heads", None)
    if heads is not None:
        head = getattr(heads, "head", None)
        if isinstance(head, Linear):
            return int(head.out_features)
    clf = getattr(model, "classifier", None)
    if isinstance(clf, Linear):
        return int(clf.out_features)
    if isinstance(clf, Sequential):
        for layer in reversed(list(clf.children())):
            if isinstance(layer, Linear):
                return int(layer.out_features)
    return None


def _replace_linear(linear: Linear, num_classes: int) -> Linear:
    """Create a new ``Linear`` with the same input size and new output size."""
    return Linear(int(linear.in_features), num_classes)


def adapt_classifier_backbone(model: Module, num_classes: int) -> Module:
    """Replace the classification head for common torchvision architectures.

    Supports ``fc``, ``classifier`` (Linear or Sequential), and ViT ``heads.head``.

    Args:
        model: Classification backbone.
        num_classes: Number of output classes (logits).

    Returns:
        The same ``model`` instance with an updated head.

    Raises:
        NotImplementedError: If no supported head attribute is found.
    """
    current = classifier_output_features(model)
    if current == num_classes:
        return model

    fc = getattr(model, "fc", None)
    if isinstance(fc, Linear):
        model.fc = _replace_linear(fc, num_classes)
        return model

    heads = getattr(model, "heads", None)
    if heads is not None:
        head = getattr(heads, "head", None)
        if isinstance(head, Linear):
            heads.head = _replace_linear(head, num_classes)
            return model

    clf = getattr(model, "classifier", None)
    if isinstance(clf, Linear):
        model.classifier = _replace_linear(clf, num_classes)
        return model
    if isinstance(clf, Sequential):
        children = list(clf.children())
        for idx in range(len(children) - 1, -1, -1):
            layer = children[idx]
            if isinstance(layer, Linear):
                children[idx] = _replace_linear(layer, num_classes)
                model.classifier = Sequential(*children)
                return model

    msg = (
        "Classifier head adaptation is not implemented for this architecture; "
        "expected ``fc``, ``classifier`` (Linear or Sequential), or ViT ``heads.head``"
    )
    raise NotImplementedError(msg)


def adapt_classifier_input_channels(model: Module, in_channels: int) -> Module:
    """Replace the stem convolution when input channel count differs from pretrained.

    Args:
        model: Classification backbone.
        in_channels: Expected input channels (e.g. 1 for grayscale, 3 for RGB).

    Returns:
        The same ``model`` instance, possibly with a new stem ``Conv2d``.
    """
    stem = find_stem_conv(model)
    if stem is None:
        return model
    old_in = stem.in_channels
    if old_in == in_channels:
        return model
    replacement = Conv2d(
        in_channels,
        stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        dilation=stem.dilation,
        groups=stem.groups,
        bias=stem.bias is not None,
        padding_mode=stem.padding_mode,
    )
    with torch.no_grad():
        replacement.weight.copy_(_resize_conv1_in_channels(stem.weight, in_channels))
        if stem.bias is not None:
            replacement.bias.copy_(stem.bias)
    if getattr(model, "conv1", None) is stem:
        model.conv1 = replacement
    elif getattr(model, "conv_proj", None) is stem:
        model.conv_proj = replacement
    else:
        features = model.features
        if isinstance(features[0], Conv2d) and features[0] is stem:
            features[0] = replacement
        elif isinstance(features[0], Sequential) and features[0][0] is stem:
            features[0][0] = replacement
    return model


def stem_input_channels(model: Module) -> int | None:
    """Return stem input channel count if a stem conv is found.

    Args:
        model: Classification backbone.

    Returns:
        ``in_channels`` of the stem conv, or ``None``.
    """
    stem = find_stem_conv(model)
    if stem is None:
        return None
    return int(stem.in_channels)


def prepare_classification_model_from_meta(
    model: Module,
    *,
    num_classes: int,
    canonical_in_channels: int,
) -> Module:
    """Rebuild stem and head from saved inference metadata.

    Args:
        model: Base model shell before ``load_state_dict``.
        num_classes: Target number of classes.
        canonical_in_channels: Target input channel count.

    Returns:
        Model with adapted stem and classification head.
    """
    model = adapt_classifier_input_channels(model, canonical_in_channels)
    return adapt_classifier_backbone(model, num_classes)
