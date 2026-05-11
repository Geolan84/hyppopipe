from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from hyppopipe.data.dataset.adapters import (
    adapt_dataset_for_classification,
    adapt_split_for_roi_classification,
)
from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.data.image import Image
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.train.config import TrainingConfig
from hyppopipe.train.tasks.base import TrainingTask

_IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)


def _normalize_stats_for_channels(
    num_channels: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean/std aligned with channel count; ImageNet tuple is truncated or tiled."""
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


def _normalize_tensor_imagenet_style(tensor: Tensor) -> Tensor:
    c = int(tensor.shape[0])
    mean_t = torch.tensor(
        _normalize_stats_for_channels(c)[0],
        dtype=tensor.dtype,
        device=tensor.device,
    ).view(c, 1, 1)
    std_t = torch.tensor(
        _normalize_stats_for_channels(c)[1],
        dtype=tensor.dtype,
        device=tensor.device,
    ).view(c, 1, 1)
    return (tensor - mean_t) / std_t


def _ensure_channel_count(tensor: Tensor, target_channels: int) -> Tensor:
    """Maps CHW tensor to ``target_channels`` (expand by tiling / shrink by grouped mean)."""
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


def _resize_conv1_in_channels(weight: Tensor, new_in: int) -> Tensor:
    """Remap first-conv weights when in_channels changes (e.g. RGB backbone + grayscale data)."""
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


def adapt_classifier_input_channels(model: Module, in_channels: int) -> Module:
    """If backbone has ``conv1`` (e.g. ResNet), reshape its weights to ``in_channels``."""
    conv1 = getattr(model, "conv1", None)
    if not isinstance(conv1, torch.nn.Conv2d):
        return model
    old_in = conv1.in_channels
    if old_in == in_channels:
        return model
    replacement = torch.nn.Conv2d(
        in_channels,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        dilation=conv1.dilation,
        groups=conv1.groups,
        bias=conv1.bias is not None,
        padding_mode=conv1.padding_mode,
    )
    with torch.no_grad():
        replacement.weight.copy_(_resize_conv1_in_channels(conv1.weight, in_channels))
        if conv1.bias is not None:
            replacement.bias.copy_(conv1.bias)
    model.conv1 = replacement
    return model


class _ImageTensorDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        base: Dataset[tuple[Any, int]],
        transform: transforms.Compose,
    ):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        raw, target = self.base[index]
        x = raw.body if isinstance(raw, Image) else raw
        if not isinstance(x, torch.Tensor):
            msg = "Expected torch.Tensor or hyppopipe.data.image.Image"
            raise TypeError(msg)
        x = self.transform(x)
        return x, int(target)


def default_classification_transform(
    *,
    resize: tuple[int, int] = (224, 224),
    canonical_channels: int,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(
                lambda t: t.float() / 255.0 if t.dtype != torch.float32 else t
            ),
            transforms.Resize(resize),
            transforms.Lambda(lambda t: _ensure_channel_count(t, canonical_channels)),
            transforms.Lambda(_normalize_tensor_imagenet_style),
        ]
    )


def infer_canonical_input_channels(
    dataset: Dataset[Any],
    *,
    max_samples: int = 16384,
) -> int:
    """Largest channel count seen in the first ``max_samples`` items (mixed grayscale/RGB-safe)."""
    n = len(dataset)
    if n == 0:
        msg = "Cannot infer input channels from an empty dataset"
        raise ValueError(msg)
    take = n if n <= max_samples else max_samples
    max_c = 1
    for i in range(take):
        raw, _ = dataset[i]
        x = raw.body if isinstance(raw, Image) else raw
        if not isinstance(x, torch.Tensor):
            msg = "Expected torch.Tensor or hyppopipe.data.image.Image"
            raise TypeError(msg)
        if x.ndim != 3:
            msg = f"Expected CHW tensor for classification, got shape {tuple(x.shape)}"
            raise ValueError(msg)
        max_c = max(max_c, int(x.shape[0]))
    return max_c


def _classification_core_splits(
    data: SplitData,
    classifier: ImageClassifier,
) -> tuple[Dataset[Any], Dataset[Any]]:
    if classifier.source_mode == "roi":
        return (
            adapt_split_for_roi_classification(data.train),
            adapt_split_for_roi_classification(data.val),
        )
    return (
        adapt_dataset_for_classification(data.train),
        adapt_dataset_for_classification(data.val),
    )


def infer_num_classes(dataset: Dataset[Any]) -> int:
    if hasattr(dataset, "classes"):
        classes = getattr(dataset, "classes")
        if isinstance(classes, list):
            return len(classes)
    targets: set[int] = set()
    n = len(dataset)
    for i in range(min(n, 2048)):
        _, y = dataset[i]
        targets.add(int(y))
    if len(targets) >= n or n <= 2048:
        return max(targets) + 1 if targets else 1
    msg = (
        "Could not infer num_classes reliably; set classes on the dataset "
        "or use a dataset with a ``classes`` attribute"
    )
    raise ValueError(msg)


def prepare_classification_model_from_meta(
    model: Module,
    *,
    num_classes: int,
    canonical_in_channels: int,
) -> Module:
    """Rebuild classifier architecture using values captured during training."""
    model = adapt_classifier_input_channels(model, canonical_in_channels)
    return adapt_classifier_backbone(model, num_classes)


def adapt_classifier_backbone(model: Module, num_classes: int) -> Module:
    if hasattr(model, "fc") and isinstance(model.fc, Module):
        in_features = model.fc.in_features  # type: ignore[attr-defined]
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model
    if hasattr(model, "classifier") and isinstance(model.classifier, Module):
        clf = model.classifier
        if hasattr(clf, "in_features"):
            in_features = clf.in_features  # type: ignore[attr-defined]
            model.classifier = torch.nn.Linear(in_features, num_classes)
            return model
    msg = (
        "Classifier head adaptation is not implemented for this architecture; "
        "expected a module with an ``fc`` head (e.g. ResNet)"
    )
    raise NotImplementedError(msg)


def prepare_classification_model(
    model: Module,
    train_dataset: Dataset[Any],
    classifier: ImageClassifier,
    *,
    in_channels: int | None = None,
) -> Module:
    num_classes = (
        classifier.num_classes
        if classifier.num_classes is not None
        else infer_num_classes(train_dataset)
    )
    in_ch = (
        in_channels
        if in_channels is not None
        else infer_canonical_input_channels(train_dataset)
    )
    model = adapt_classifier_input_channels(model, in_ch)
    return adapt_classifier_backbone(model, num_classes)


def classification_train_val_loaders(
    data: SplitData,
    config: TrainingConfig,
    classifier: ImageClassifier,
    *,
    canonical_channels: int | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_core, val_core = _classification_core_splits(data, classifier)
    cc = canonical_channels
    if cc is None:
        cc = max(
            infer_canonical_input_channels(train_core),
            infer_canonical_input_channels(val_core),
        )

    train_tf = (
        classifier.train_transform
        if classifier.train_transform is not None
        else default_classification_transform(canonical_channels=cc)
    )
    val_tf = (
        classifier.val_transform
        if classifier.val_transform is not None
        else default_classification_transform(canonical_channels=cc)
    )

    train_ds = _ImageTensorDataset(train_core, train_tf)
    val_ds = _ImageTensorDataset(val_core, val_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


class ClassificationTrainingTask(TrainingTask):
    def __init__(self, classifier: ImageClassifier) -> None:
        self._classifier = classifier

    def inference_meta_from_prepared(self, prepared: Module) -> dict[str, Any]:
        meta: dict[str, Any] = {"task": "classification"}
        fc = getattr(prepared, "fc", None)
        if fc is not None and hasattr(fc, "out_features"):
            meta["num_classes"] = int(fc.out_features)  # type: ignore[attr-defined]
        conv1 = getattr(prepared, "conv1", None)
        if isinstance(conv1, torch.nn.Conv2d):
            meta["canonical_in_channels"] = int(conv1.in_channels)
        return meta

    def split_lengths(self, data: SplitData) -> tuple[int, int]:
        train_core, val_core = _classification_core_splits(data, self._classifier)
        return len(train_core), len(val_core)

    def prepare(
        self,
        model: Module,
        data: SplitData,
        config: TrainingConfig,
    ) -> tuple[Module, DataLoader[Any], DataLoader[Any]]:
        train_cls, val_cls = _classification_core_splits(data, self._classifier)
        canonical_c = max(
            infer_canonical_input_channels(train_cls),
            infer_canonical_input_channels(val_cls),
        )
        prepared = prepare_classification_model(
            model,
            train_cls,
            self._classifier,
            in_channels=canonical_c,
        )
        train_ld, val_ld = classification_train_val_loaders(
            data,
            config,
            self._classifier,
            canonical_channels=canonical_c,
        )
        return prepared, train_ld, val_ld

    def create_criterion(self, device: torch.device, config: TrainingConfig) -> Module:
        return config.default_classification_loss().to(device)

    def train_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> tuple[float, int]:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        n = x.size(0)
        return loss.item() * n, n

    def eval_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        device: torch.device,
    ) -> tuple[float, int]:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        n = x.size(0)
        return loss.item() * n, n
