"""Training hyperparameters and device utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from hyppopipe.train.objectives import LossFactory, MonitorSpec

import torch
from torch.nn import Module
from torch.optim import Optimizer


@dataclass(slots=True)
class EarlyStoppingConfig:
    """Early stopping and best-checkpoint settings for :class:`~hyppopipe.train.trainer.Trainer`."""

    enabled: bool = True
    patience: int = 5
    delta: float = 0.0
    verbose: bool = False
    save_path: str | None = None
    save_to_disk: bool = True


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Pick the torch device used for training and inference.

    Args:
        device: Explicit device; when None, prefers MPS, then CUDA, then CPU.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device is not None:
        return torch.device(device)
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


OptimizerFactory = Callable[[Iterable[torch.nn.Parameter]], Optimizer]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters shared across training tasks."""

    epochs: int = 10
    batch_size: int = 32
    val_batch_size: int | None = None
    lr: float = 1e-3
    num_workers: int = 0
    seed: int | None = None
    device: str | torch.device | None = None

    optimizer_name: Literal["adamw", "adam", "sgd"] = "adamw"
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer_factory: OptimizerFactory | None = None

    early_stopping: EarlyStoppingConfig | None = field(
        default_factory=EarlyStoppingConfig
    )

    loss: LossFactory | None = None
    """Optional loss override (module or ``(device, config) -> Module`` factory)."""
    monitor: MonitorSpec | None = None
    """Optional validation monitor for early stopping and best checkpoints."""

    def copy_with(self, **changes: Any) -> TrainingConfig:
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)

    def resolve_val_batch_size(self) -> int:
        """Validation batch size, defaulting to ``batch_size``."""
        if self.val_batch_size is not None:
            return self.val_batch_size
        return self.batch_size

    def build_optimizer(self, params: Iterable[torch.nn.Parameter]) -> Optimizer:
        """Construct the optimizer from ``optimizer_name`` or ``optimizer_factory``."""
        if self.optimizer_factory is not None:
            return self.optimizer_factory(params)
        kwargs: dict[str, Any] = {"lr": self.lr, **self.optimizer_kwargs}
        match self.optimizer_name:
            case "adamw":
                return torch.optim.AdamW(params, **kwargs)
            case "adam":
                return torch.optim.Adam(params, **kwargs)
            case "sgd":
                return torch.optim.SGD(params, **kwargs)

    def default_classification_loss(self) -> Module:
        """Default cross-entropy loss for classification tasks."""
        return torch.nn.CrossEntropyLoss()


def apply_seed(seed: int | None) -> None:
    """Set random seeds for torch CPU, CUDA, and MPS when ``seed`` is not None.

    Args:
        seed: Seed value, or None to skip.
    """
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)
