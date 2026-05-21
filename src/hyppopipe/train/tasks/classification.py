"""Classification training task: datasets, model prep, and batch loops.

Wires ``ImageClassifier`` pipeline steps to ``Trainer`` via ``ClassificationTrainingTask``,
including ROI vs full-image modes and transform spec export for inference.
"""

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
from hyppopipe.train.objectives import EpochMetric
from hyppopipe.train.tasks.base import TrainingTask
from hyppopipe.train.tasks.classification_model import (
    adapt_classifier_backbone,
    adapt_classifier_input_channels,
    classifier_output_features,
    stem_input_channels,
)
from hyppopipe.train.tasks.classification_transforms import (
    ensure_channel_count,
    normalize_tensor_imagenet_style,
)
from hyppopipe.train.transforms import (
    ClassificationTransforms,
    classification_transform_spec_for_inference,
    coerce_classification_transform_fn,
)

_ensure_channel_count = ensure_channel_count
_normalize_tensor_imagenet_style = normalize_tensor_imagenet_style


class _ImageTensorDataset(Dataset[tuple[torch.Tensor, int]]):
    """Wrap a label dataset and apply a per-sample tensor transform."""

    def __init__(
        self,
        base: Dataset[tuple[Any, int]],
        transform: transforms.Compose | Any,
    ):
        """Initialize the wrapper.

        Args:
            base: Dataset yielding ``(image, label)`` with ``Image`` or ``Tensor``.
            transform: Callable applied to the image tensor before return.
        """
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
    """Infer maximum channel count over a sample of the dataset.

    Scans up to ``max_samples`` items and returns the max ``C`` in CHW tensors.

    Args:
        dataset: Dataset whose items are ``(image, label)``.
        max_samples: Upper bound on items to scan.

    Returns:
        Maximum channel dimension observed.

    Raises:
        ValueError: If the dataset is empty or tensors are not CHW.
        TypeError: If images are neither ``Image`` nor ``Tensor``.
    """
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
    """Adapt train/val splits for full-image or ROI classification."""
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
    """Infer number of classes from ``dataset.classes`` or label values.

    Args:
        dataset: Classification dataset.

    Returns:
        Number of classes (``len(classes)`` or ``max(label) + 1``).

    Raises:
        ValueError: If classes cannot be inferred reliably from a large dataset.
    """
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
    """Adapt stem channels and classification head for the training dataset.

    Args:
        model: Base classifier backbone.
        train_dataset: Training split used to infer classes and channels.
        classifier: Step functor (may fix ``num_classes``).
        in_channels: Optional override for input channel count.

    Returns:
        Model with matching stem and ``num_classes`` output logits.
    """
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


def _resolve_classification_transforms(
    transforms: ClassificationTransforms | None,
    *,
    weights: WeightsEnum | None,
    canonical_channels: int,
) -> ClassificationTransforms:
    if transforms is not None:
        return transforms
    if weights is not None:
        return ClassificationTransforms.from_weights(
            weights, canonical_channels=canonical_channels
        )
    return ClassificationTransforms.default(canonical_channels=canonical_channels)


def classification_train_val_loaders(
    data: SplitData,
    config: TrainingConfig,
    classifier: ImageClassifier,
    *,
    canonical_channels: int | None = None,
    transforms: ClassificationTransforms | None = None,
    weights: WeightsEnum | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build train and validation dataloaders for classification.

    Args:
        data: Train/validation splits.
        config: Batch size, workers, and related settings.
        classifier: Step functor (ROI mode and class count).
        canonical_channels: Channel count for transforms; inferred if ``None``.
        transforms: Train/val transforms from :class:`~hyppopipe.train.trainer.Trainer`.
        weights: Optional ``WeightsEnum`` when transforms are not set explicitly.

    Returns:
        Train and validation ``DataLoader`` instances.
    """
    train_core, val_core = _classification_core_splits(data, classifier)
    cc = canonical_channels
    if cc is None:
        cc = max(
            infer_canonical_input_channels(train_core),
            infer_canonical_input_channels(val_core),
        )

    resolved = _resolve_classification_transforms(
        transforms, weights=weights, canonical_channels=cc
    )
    crop_size = int((resolved.transform_spec or {}).get("crop_size", 224))
    train_fn = coerce_classification_transform_fn(
        resolved.train,
        canonical_channels=cc,
        image_size=crop_size,
    )
    val_fn = coerce_classification_transform_fn(
        resolved.val,
        canonical_channels=cc,
        image_size=crop_size,
    )

    train_ds = _ImageTensorDataset(train_core, train_fn)
    val_ds = _ImageTensorDataset(val_core, val_fn)

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
    """``TrainingTask`` implementation for ``ImageClassifier`` pipeline steps."""

    def __init__(self, classifier: ImageClassifier) -> None:
        """Store the classifier step configuration.

        Args:
            classifier: Pipeline ``ImageClassifier`` functor for this step.
        """
        self._classifier = classifier
        self._transform_spec: dict[str, Any] | None = None
        self._resolved_transforms: ClassificationTransforms | None = None

    def inference_meta_from_prepared(self, prepared: Module) -> dict[str, Any]:
        """Export class count, channels, and transform spec for inference."""
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
        """Return lengths after ROI or classification dataset adaptation."""
        train_core, val_core = _classification_core_splits(data, self._classifier)
        return len(train_core), len(val_core)

    def prepare(
        self,
        model: Module,
        data: SplitData,
        config: TrainingConfig,
        *,
        weights_enum: WeightsEnum | None = None,
        transforms: ClassificationTransforms | None = None,
    ) -> tuple[Module, DataLoader[Any], DataLoader[Any]]:
        """Prepare model, dataloaders, and persisted transform spec."""
        if transforms is not None and not isinstance(
            transforms, ClassificationTransforms
        ):
            msg = (
                f"Classification training expects ClassificationTransforms or None, "
                f"got {type(transforms).__name__}"
            )
            raise TypeError(msg)

        train_cls, val_cls = _classification_core_splits(data, self._classifier)
        canonical_c = max(
            infer_canonical_input_channels(train_cls),
            infer_canonical_input_channels(val_cls),
        )
        self._resolved_transforms = _resolve_classification_transforms(
            transforms,
            weights=weights_enum,
            canonical_channels=canonical_c,
        )
        self._transform_spec = classification_transform_spec_for_inference(
            self._resolved_transforms,
            weights_enum=weights_enum,
            canonical_channels=canonical_c,
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
            transforms=self._resolved_transforms,
            weights=weights_enum,
        )
        return prepared, train_ld, val_ld

    def create_criterion(self, device: torch.device, config: TrainingConfig) -> Module:
        """Return the default cross-entropy loss on ``device``."""
        return config.default_classification_loss().to(device)

    def update_monitor(
        self,
        metric: EpochMetric,
        model: Module,
        batch: Any,
        device: torch.device,
    ) -> None:
        """Accumulate classification metric from one validation batch."""
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model(x)
        metric.update(logits, y)

    def train_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> tuple[float, int]:
        """Forward, backward, and optimizer step for one classification batch."""
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
        """Validation forward pass without parameter updates."""
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        n = x.size(0)
        return loss.item() * n, n
