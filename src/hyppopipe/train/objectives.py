"""Training loss and validation-monitor configuration.

Loss modules drive backpropagation; :class:`MonitorSpec` selects the scalar used
for early stopping and best-checkpoint selection (often a metric such as F1 or
Dice, which may differ from the loss).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor
from torch.nn import Module

from hyppopipe.data.metrics.segmentation import BCEDiceLoss
from hyppopipe.train.config import TrainingConfig

LossFactory: TypeAlias = Module | Callable[[torch.device, TrainingConfig], Module]
MonitorMode: TypeAlias = Literal["min", "max"]


def resolve_loss(
    loss: LossFactory | None,
    device: torch.device,
    config: TrainingConfig,
    *,
    default: Callable[[torch.device, TrainingConfig], Module],
) -> Module:
    """Build a loss module from an explicit spec or a task default factory."""
    if loss is None:
        return default(device, config)
    if isinstance(loss, Module):
        return loss.to(device)
    built = loss(device, config)
    return built.to(device)


class EpochMetric(ABC):
    """Running validation metric aggregated over one epoch."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state before a new validation epoch."""

    @abstractmethod
    def update(self, *args: Tensor, **kwargs: Tensor) -> None:
        """Incorporate one batch (signature is task-specific)."""

    @abstractmethod
    def compute(self) -> float:
        """Return the scalar metric for the completed epoch."""


@dataclass(frozen=True, slots=True)
class MonitorSpec:
    """Validation metric used for logging, early stopping, and best checkpoints."""

    name: str
    mode: MonitorMode
    factory: Callable[[], EpochMetric]

    @classmethod
    def custom(
        cls,
        name: str,
        metric_factory: Callable[[], EpochMetric],
        *,
        mode: MonitorMode,
    ) -> MonitorSpec:
        """Wrap a user-defined :class:`EpochMetric` factory."""
        return cls(name=name, mode=mode, factory=metric_factory)


class _ClassificationAccuracyMetric(EpochMetric):
    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, logits: Tensor, targets: Tensor) -> None:
        preds = logits.argmax(dim=1)
        self._correct += int((preds == targets).sum().item())
        self._total += int(targets.numel())

    def compute(self) -> float:
        return self._correct / max(self._total, 1)


class _ClassificationConfusionMetric(EpochMetric):
    """Base for precision / recall / F1 from a confusion matrix."""

    def __init__(self, *, num_classes: int | None = None) -> None:
        self._num_classes = num_classes
        self._confusion: Tensor | None = None

    def reset(self) -> None:
        self._confusion = None

    def update(self, logits: Tensor, targets: Tensor) -> None:
        preds = logits.argmax(dim=1)
        num_classes = self._num_classes or int(
            max(logits.size(1), targets.max().item() + 1, preds.max().item() + 1)
        )
        indices = targets.reshape(-1) * num_classes + preds.reshape(-1)
        batch_cm = torch.bincount(indices, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )
        if self._confusion is None:
            self._confusion = batch_cm
        else:
            if self._confusion.shape != batch_cm.shape:
                size = max(self._confusion.size(0), batch_cm.size(0))
                self._confusion = _pad_confusion(self._confusion, size)
                batch_cm = _pad_confusion(batch_cm, size)
            self._confusion += batch_cm

    def _per_class_prf(self) -> tuple[Tensor, Tensor, Tensor]:
        if self._confusion is None:
            z = torch.zeros(1)
            return z, z, z
        cm = self._confusion.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def _macro(self, tensor: Tensor) -> float:
        if tensor.numel() == 0:
            return 0.0
        return float(tensor.mean().item())


def _pad_confusion(cm: Tensor, size: int) -> Tensor:
    if cm.size(0) >= size:
        return cm
    out = torch.zeros(size, size, dtype=cm.dtype, device=cm.device)
    out[: cm.size(0), : cm.size(1)] = cm
    return out


class _ClassificationPrecisionMetric(_ClassificationConfusionMetric):
    def compute(self) -> float:
        precision, _, _ = self._per_class_prf()
        return self._macro(precision)


class _ClassificationRecallMetric(_ClassificationConfusionMetric):
    def compute(self) -> float:
        _, recall, _ = self._per_class_prf()
        return self._macro(recall)


class _ClassificationF1Metric(_ClassificationConfusionMetric):
    def compute(self) -> float:
        _, _, f1 = self._per_class_prf()
        return self._macro(f1)


# --- Segmentation metrics (semantic, class-index masks) -----------------------


def _seg_logits(output: Tensor | dict[str, Tensor]) -> Tensor:
    if isinstance(output, dict):
        return output["out"]
    return output


class _SegmentationPixelAccuracyMetric(EpochMetric):
    def __init__(self, *, ignore_index: int | None = None) -> None:
        self._ignore_index = ignore_index
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, logits: Tensor, targets: Tensor) -> None:
        preds = logits.argmax(dim=1)
        if self._ignore_index is not None:
            mask = targets != self._ignore_index
            self._correct += int((preds[mask] == targets[mask]).sum().item())
            self._total += int(mask.sum().item())
        else:
            self._correct += int((preds == targets).sum().item())
            self._total += int(targets.numel())

    def compute(self) -> float:
        return self._correct / max(self._total, 1)


class _SegmentationMeanIoUMetric(EpochMetric):
    def __init__(
        self, *, num_classes: int | None = None, ignore_index: int = -1
    ) -> None:
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self._intersection: Tensor | None = None
        self._union: Tensor | None = None

    def reset(self) -> None:
        self._intersection = None
        self._union = None

    def update(self, logits: Tensor, targets: Tensor) -> None:
        preds = logits.argmax(dim=1)
        num_classes = self._num_classes or int(
            max(logits.size(1), targets.max().item() + 1, preds.max().item() + 1)
        )
        if self._intersection is None:
            self._intersection = torch.zeros(num_classes, dtype=torch.float64)
            self._union = torch.zeros(num_classes, dtype=torch.float64)
        for cls in range(num_classes):
            if cls == self._ignore_index:
                continue
            pred_c = preds == cls
            target_c = targets == cls
            inter = (pred_c & target_c).sum().double()
            union = (pred_c | target_c).sum().double()
            if cls >= self._intersection.size(0):
                self._grow(cls + 1)
            self._intersection[cls] += inter
            self._union[cls] += union

    def _grow(self, size: int) -> None:
        assert self._intersection is not None and self._union is not None
        new_i = torch.zeros(size, dtype=torch.float64)
        new_u = torch.zeros(size, dtype=torch.float64)
        new_i[: self._intersection.size(0)] = self._intersection
        new_u[: self._union.size(0)] = self._union
        self._intersection = new_i
        self._union = new_u

    def compute(self) -> float:
        if self._intersection is None or self._union is None:
            return 0.0
        valid = self._union > 0
        if not valid.any():
            return 0.0
        iou = self._intersection[valid] / (self._union[valid] + 1e-8)
        return float(iou.mean().item())


class _SegmentationMeanDiceMetric(EpochMetric):
    """Mean Dice over classes with at least one foreground pixel in target or pred."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        skip_background: bool = True,
    ) -> None:
        self._num_classes = num_classes
        self._skip_background = skip_background
        self._intersection: Tensor | None = None
        self._pred_sum: Tensor | None = None
        self._target_sum: Tensor | None = None

    def reset(self) -> None:
        self._intersection = None
        self._pred_sum = None
        self._target_sum = None

    def update(self, logits: Tensor, targets: Tensor) -> None:
        preds = logits.argmax(dim=1)
        num_classes = self._num_classes or int(
            max(logits.size(1), targets.max().item() + 1, preds.max().item() + 1)
        )
        start = 1 if self._skip_background else 0
        if self._intersection is None:
            self._intersection = torch.zeros(num_classes, dtype=torch.float64)
            self._pred_sum = torch.zeros(num_classes, dtype=torch.float64)
            self._target_sum = torch.zeros(num_classes, dtype=torch.float64)
        for cls in range(start, num_classes):
            pred_c = preds == cls
            target_c = targets == cls
            if cls >= self._intersection.size(0):
                self._grow(cls + 1)
            self._intersection[cls] += (pred_c & target_c).sum().double()
            self._pred_sum[cls] += pred_c.sum().double()
            self._target_sum[cls] += target_c.sum().double()

    def _grow(self, size: int) -> None:
        assert self._intersection is not None
        assert self._pred_sum is not None
        assert self._target_sum is not None
        for buf in (self._intersection, self._pred_sum, self._target_sum):
            new = torch.zeros(size, dtype=torch.float64)
            new[: buf.size(0)] = buf
            if buf is self._intersection:
                self._intersection = new
            elif buf is self._pred_sum:
                self._pred_sum = new
            else:
                self._target_sum = new

    def compute(self) -> float:
        if (
            self._intersection is None
            or self._pred_sum is None
            or self._target_sum is None
        ):
            return 0.0
        dice_scores: list[float] = []
        start = 1 if self._skip_background else 0
        for cls in range(start, self._intersection.size(0)):
            denom = self._pred_sum[cls] + self._target_sum[cls]
            if denom <= 0:
                continue
            dice = (2.0 * self._intersection[cls]) / (denom + 1e-8)
            dice_scores.append(float(dice.item()))
        if not dice_scores:
            return 0.0
        return sum(dice_scores) / len(dice_scores)


class ClassificationObjectives:
    """Built-in classification losses and validation monitors."""

    @staticmethod
    def loss(
        kind: Literal["cross_entropy"] = "cross_entropy",
        **kwargs: object,
    ) -> LossFactory:
        """Return a loss factory; ``kind`` selects the built-in implementation."""
        if kind != "cross_entropy":
            msg = f"Unknown classification loss {kind!r}"
            raise ValueError(msg)

        def factory(device: torch.device, config: TrainingConfig) -> Module:
            del config
            return torch.nn.CrossEntropyLoss(**kwargs).to(device)

        return factory

    @staticmethod
    def monitor(
        name: Literal["accuracy", "precision", "recall", "f1_macro"],
        *,
        num_classes: int | None = None,
    ) -> MonitorSpec:
        """Return a validation monitor spec (mode ``max`` for all built-ins)."""
        match name:
            case "accuracy":
                return MonitorSpec(
                    name="accuracy",
                    mode="max",
                    factory=_ClassificationAccuracyMetric,
                )
            case "precision":
                return MonitorSpec(
                    name="precision_macro",
                    mode="max",
                    factory=lambda: _ClassificationPrecisionMetric(
                        num_classes=num_classes
                    ),
                )
            case "recall":
                return MonitorSpec(
                    name="recall_macro",
                    mode="max",
                    factory=lambda: _ClassificationRecallMetric(
                        num_classes=num_classes
                    ),
                )
            case "f1_macro":
                return MonitorSpec(
                    name="f1_macro",
                    mode="max",
                    factory=lambda: _ClassificationF1Metric(num_classes=num_classes),
                )
        msg = f"Unknown classification monitor {name!r}"
        raise ValueError(msg)


class SegmentationObjectives:
    """Built-in semantic-segmentation losses and validation monitors."""

    @staticmethod
    def loss(
        kind: Literal["cross_entropy", "bce_dice"] = "cross_entropy",
        **kwargs: object,
    ) -> LossFactory:
        """Return a loss factory for semantic segmentation."""
        match kind:
            case "cross_entropy":

                def ce_factory(device: torch.device, config: TrainingConfig) -> Module:
                    del config
                    return torch.nn.CrossEntropyLoss(**kwargs).to(device)

                return ce_factory
            case "bce_dice":

                def bce_factory(device: torch.device, config: TrainingConfig) -> Module:
                    del config
                    return BCEDiceLoss(**kwargs).to(device)

                return bce_factory
        msg = f"Unknown segmentation loss {kind!r}"
        raise ValueError(msg)

    @staticmethod
    def monitor(
        name: Literal["pixel_accuracy", "mean_iou", "dice"],
        *,
        num_classes: int | None = None,
        ignore_index: int | None = None,
        skip_background: bool = True,
    ) -> MonitorSpec:
        """Return a validation monitor spec for semantic masks."""
        match name:
            case "pixel_accuracy":
                return MonitorSpec(
                    name="pixel_accuracy",
                    mode="max",
                    factory=lambda: _SegmentationPixelAccuracyMetric(
                        ignore_index=ignore_index
                    ),
                )
            case "mean_iou":
                return MonitorSpec(
                    name="mean_iou",
                    mode="max",
                    factory=lambda: _SegmentationMeanIoUMetric(
                        num_classes=num_classes,
                        ignore_index=ignore_index if ignore_index is not None else -1,
                    ),
                )
            case "dice":
                return MonitorSpec(
                    name="mean_dice",
                    mode="max",
                    factory=lambda: _SegmentationMeanDiceMetric(
                        num_classes=num_classes,
                        skip_background=skip_background,
                    ),
                )
        msg = f"Unknown segmentation monitor {name!r}"
        raise ValueError(msg)


def extract_segmentation_logits(output: Tensor | dict[str, Tensor]) -> Tensor:
    """Public helper for tasks updating segmentation monitors."""
    return _seg_logits(output)
