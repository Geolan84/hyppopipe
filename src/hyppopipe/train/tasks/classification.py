from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import WeightsEnum

from hyppopipe.data.dataset.adapters import (
    adapt_dataset_for_classification,
    adapt_split_for_roi_classification,
)
from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.data.image import Image
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.train.config import TrainingConfig
from hyppopipe.train.tasks.base import TrainingTask
from hyppopipe.train.tasks.classification_model import (
    adapt_classifier_backbone,
    adapt_classifier_input_channels,
    classifier_output_features,
    stem_input_channels,
)
from hyppopipe.train.tasks.classification_transforms import (
    classification_transform_from_spec,
    classification_transforms_for_weights,
    default_transform_spec,
    ensure_channel_count,
    normalize_tensor_imagenet_style,
    transform_spec_from_weights,
)

_ensure_channel_count = ensure_channel_count
_normalize_tensor_imagenet_style = normalize_tensor_imagenet_style


class _ImageTensorDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        base: Dataset[tuple[Any, int]],
        transform: transforms.Compose | Any,
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


def infer_canonical_input_channels(
    dataset: Dataset[Any],
    *,
    max_samples: int = 16384,
) -> int:
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
    transform_spec: dict[str, Any] | None = None,
    weights: WeightsEnum | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_core, val_core = _classification_core_splits(data, classifier)
    cc = canonical_channels
    if cc is None:
        cc = max(
            infer_canonical_input_channels(train_core),
            infer_canonical_input_channels(val_core),
        )

    if classifier.train_transform is not None:
        train_tf = classifier.train_transform
    elif weights is not None:
        train_tf, _, _ = classification_transforms_for_weights(
            weights, canonical_channels=cc
        )
    else:
        train_tf = classification_transform_from_spec(
            transform_spec,
            canonical_channels=cc,
            train=True,
        )

    if classifier.val_transform is not None:
        val_tf = classifier.val_transform
    elif weights is not None:
        _, val_tf, _ = classification_transforms_for_weights(
            weights, canonical_channels=cc
        )
    else:
        val_tf = classification_transform_from_spec(
            transform_spec,
            canonical_channels=cc,
            train=False,
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
        batch_size=config.resolve_val_batch_size(),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


class ClassificationTrainingTask(TrainingTask):
    def __init__(self, classifier: ImageClassifier) -> None:
        self._classifier = classifier
        self._transform_spec: dict[str, Any] | None = None

    def inference_meta_from_prepared(self, prepared: Module) -> dict[str, Any]:
        meta: dict[str, Any] = {"task": "classification"}
        out_features = classifier_output_features(prepared)
        if out_features is not None:
            meta["num_classes"] = out_features
        stem_ch = stem_input_channels(prepared)
        if stem_ch is not None:
            meta["canonical_in_channels"] = stem_ch
        if self._transform_spec is not None:
            meta["transform_spec"] = dict(self._transform_spec)
        return meta

    def split_lengths(self, data: SplitData) -> tuple[int, int]:
        train_core, val_core = _classification_core_splits(data, self._classifier)
        return len(train_core), len(val_core)

    def prepare(
        self,
        model: Module,
        data: SplitData,
        config: TrainingConfig,
        *,
        weights_enum: WeightsEnum | None = None,
    ) -> tuple[Module, DataLoader[Any], DataLoader[Any]]:
        train_cls, val_cls = _classification_core_splits(data, self._classifier)
        canonical_c = max(
            infer_canonical_input_channels(train_cls),
            infer_canonical_input_channels(val_cls),
        )
        if weights_enum is not None:
            self._transform_spec = transform_spec_from_weights(weights_enum)
        elif (
            self._classifier.train_transform is None
            and self._classifier.val_transform is None
        ):
            self._transform_spec = default_transform_spec(
                canonical_channels=canonical_c
            )
        else:
            self._transform_spec = None

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
            transform_spec=self._transform_spec,
            weights=weights_enum,
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
