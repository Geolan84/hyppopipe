from __future__ import annotations

from typing import Callable

import torch


class ImageLocalizer:
    """Шаг пайплайна для локализации / детекции (обучение через ``DetectionTrainingTask``)."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        train_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        val_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Args:
            num_classes: всего классов **включая фон** для torchvision Faster R-CNN.
                Если ``None``, берётся ``len(dataset.classes) + 1``.
            train_transform / val_transform: только над изображением (CHW float);
                геометрические аугментации нужно согласовывать с bbox самостоятельно.
        """
        self.num_classes = num_classes
        self.train_transform = train_transform
        self.val_transform = val_transform

    def __call__(self) -> None: ...
