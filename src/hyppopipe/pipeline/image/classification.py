from __future__ import annotations

from typing import Any, Literal

SourceMode = Literal["full", "roi"]


class ImageClassifier:
    """Classification step functor; fields are used during training."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        train_transform: Any = None,
        val_transform: Any = None,
        source_mode: SourceMode = "full",
    ) -> None:
        self.num_classes = num_classes
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.source_mode = source_mode

    def __call__(self) -> None: ...
