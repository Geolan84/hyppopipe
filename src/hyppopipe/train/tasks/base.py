from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.train.config import TrainingConfig


def model_label_for_module(model: Module) -> str:
    return model.__class__.__name__


class TrainingTask(ABC):
    """Training task: model/data setup and forward/loss steps for a pipeline step type."""

    def split_lengths(self, data: SplitData) -> tuple[int, int]:
        """Размеры train/val после приведения к данным задачи (по умолчанию ``len`` сплита)."""
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
    ) -> tuple[Module, DataLoader[Any], DataLoader[Any]]:
        """Return the prepared model and train/val dataloaders."""

    @abstractmethod
    def create_criterion(self, device: torch.device, config: TrainingConfig) -> Module:
        """Loss module on ``device`` (may depend on task and config)."""

    @abstractmethod
    def train_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> tuple[float, int]:
        """One training batch: backward + optimizer step. Returns (loss sum, sample count)."""

    @abstractmethod
    def eval_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        device: torch.device,
    ) -> tuple[float, int]:
        """One validation batch without gradients."""

    def inference_meta_from_prepared(self, prepared: Module) -> dict[str, Any]:
        """Metadata needed to rebuild ``prepared`` for inference (after ``prepare``)."""
        return {}
