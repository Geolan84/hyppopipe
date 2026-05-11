from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch
from torch.nn import Module
from torch.optim import Optimizer


@dataclass(slots=True)
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 5
    delta: float = 0.0
    verbose: bool = False
    save_path: str | None = None
    save_to_disk: bool = True


def resolve_device(device: str | torch.device | None) -> torch.device:
    """
    Resolves target device for computing

    Args:
        device (str | torch.device | None): Manual setting of device

    Returns:
        torch.device: Target backend
    """
    if device is not None:
        return torch.device(device)
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


OptimizerFactory = Callable[[Iterable[torch.nn.Parameter]], Optimizer]


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
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

    def copy_with(self, **changes: Any) -> TrainingConfig:
        return replace(self, **changes)

    def build_optimizer(self, params: Iterable[torch.nn.Parameter]) -> Optimizer:
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
        return torch.nn.CrossEntropyLoss()


def apply_seed(seed: int | None) -> None:
    """
    Applies random seed for ML backend, if it is provided.

    Args:
        seed (int | None): Random seed
    """
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)
