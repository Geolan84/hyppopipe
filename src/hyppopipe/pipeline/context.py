from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class PipelineContext:
    """Mutable bag of artifacts passed between pipeline steps.

    Similar in spirit to a Kedro catalog entry or sklearn Pipelines carrying state.
    """

    dataset: Any | None = None
    train_dataset: Any | None = None
    val_dataset: Any | None = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    model: nn.Module | None = None
    device: torch.device | None = None
    num_classes: int | None = None
    class_names: list[str] | None = None
    checkpoint_path: str | None = None
    history: dict[str, list[float]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
