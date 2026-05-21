"""Abstract training task interface for pipeline step types.

Defines the contract between ``Trainer`` and task-specific implementations
(classification, detection, segmentation) for data preparation and batch loops.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.train.config import TrainingConfig
from hyppopipe.train.objectives import EpochMetric


def model_label_for_module(model: Module) -> str:
    """Return a short human-readable label for a model (typically the class name).

    Args:
        model: PyTorch module whose type name is used for logging and artifacts.

    Returns:
        The unqualified class name of ``model``.
    """
    return model.__class__.__name__


class TrainingTask(ABC):
    """Training strategy for one pipeline step type.

    Subclasses adapt datasets, rebuild model heads, build dataloaders, and
    implement train/validation batch steps for a specific task (classification,
    detection, or segmentation).
    """

    def split_lengths(self, data: SplitData) -> tuple[int, int]:
        """Return train and validation sample counts after task-specific adaptation.

        Default implementation uses ``len`` on ``data.train`` and ``data.val``.
        Override when splits are wrapped or filtered before training.

        Args:
            data: Train/validation (and optionally test) split container.

        Returns:
            Pair ``(train_size, val_size)``.

        Raises:
            TypeError: If lengths cannot be computed from the split objects.
        """
        try:
            return len(data.train), len(data.val)
        except TypeError as e:
            raise TypeError(
                "Cannot compute train/val sizes from this split; "
                "override split_lengths on the task or pass torch Dataset splits."
            ) from e

    @abstractmethod
    def prepare(
        self,
        model: Module,
        data: SplitData,
        config: TrainingConfig,
        *,
        weights_enum: Any | None = None,
        transforms: Any | None = None,
    ) -> tuple[Module, DataLoader[Any], DataLoader[Any]]:
        """Prepare the model and train/validation dataloaders.

        Args:
            model: Base model (often from a torchvision factory).
            data: Dataset splits for training and validation.
            config: Hyperparameters (batch size, workers, etc.).
            weights_enum: Optional pretrained ``WeightsEnum`` for architecture hints.
            transforms: Optional task-specific transforms from :class:`~hyppopipe.train.trainer.Trainer`.

        Returns:
            Tuple of prepared model, train loader, and validation loader.
        """

    @abstractmethod
    def create_criterion(self, device: torch.device, config: TrainingConfig) -> Module:
        """Build the default loss module for this task on ``device``.

        Args:
            device: Target device for loss tensors.
            config: Training configuration (may select loss type).

        Returns:
            Loss module moved to ``device``.
        """

    def update_monitor(
        self,
        metric: EpochMetric,
        model: Module,
        batch: Any,
        device: torch.device,
    ) -> None:
        """Update a validation metric from one batch (override per task)."""
        del metric, model, batch, device

    @abstractmethod
    def train_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> tuple[float, int]:
        """Run one training step (forward, backward, optimizer step).

        Args:
            model: Model in training mode.
            batch: Task-specific batch from the dataloader.
            criterion: Loss module from ``create_criterion``.
            optimizer: Optimizer for ``model`` parameters.
            device: Device for inputs and computation.

        Returns:
            Weighted loss sum (``loss.item() * batch_size``) and sample count.
        """

    @abstractmethod
    def eval_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        device: torch.device,
    ) -> tuple[float, int]:
        """Run one validation step without parameter updates.

        Args:
            model: Model (eval or train mode per task requirements).
            batch: Task-specific batch from the validation loader.
            criterion: Loss module from ``create_criterion``.
            device: Device for inputs and computation.

        Returns:
            Weighted loss sum and sample count, same convention as ``train_batch``.
        """

    def inference_meta_from_prepared(self, prepared: Module) -> dict[str, Any]:
        """Collect metadata needed to rebuild ``prepared`` at inference time.

        Called after ``prepare`` so exported bundles can restore head sizes,
        transforms, and task-specific options.

        Args:
            prepared: Model returned from ``prepare``.

        Returns:
            JSON-serializable metadata dict; empty by default.
        """
        return {}
