from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

SegmentationKind = Literal["instance", "semantic"]


class ImageSegmentator:
    """Шаг пайплайна для semantic или instance segmentation."""

    def __init__(
        self,
        *,
        kind: SegmentationKind = "instance",
        num_classes: int | None = None,
        input_channels: int = 3,
        image_size: tuple[int, int] | None = (224, 224),
        train_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        val_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Args:
            kind: желаемый **формат разметки** датасета: ``instance`` (таргеты как у
                Mask R-CNN) или ``semantic`` (class map). Фактическое обучение
                подстраивается под архитектуру модели: для FCN/DeepLab при
                ``kind='instance'`` автоматически используются semantic-маски
                (см. ``SegmentationTrainingTask``).
            num_classes: число классов включая фон. Если ``None``, берётся из
                датасета или масок.
            input_channels: число каналов после дефолтной подготовки semantic inputs.
            image_size: размер ``(H, W)`` для semantic batching. Для instance
                segmentation модели torchvision сами ресайзят изображения.
            train_transform / val_transform: только над изображением; геометрию
                маски нужно согласовывать в кастомном датасете.
        """
        if kind not in ("instance", "semantic"):
            raise ValueError("kind must be 'instance' or 'semantic'")
        self.kind = kind
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size
        self.train_transform = train_transform
        self.val_transform = val_transform

    def __call__(self) -> None: ...
