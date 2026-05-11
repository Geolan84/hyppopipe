"""Детекционный сплит → классификация по вырезанному ROI (как в chained predict)."""

from __future__ import annotations

import logging
from collections.abc import Sized
from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from hyppopipe.data.dataset.adapters.detection import adapt_dataset_for_detection

logger = logging.getLogger(__name__)


def _crop_xyxy_chw(chw: Tensor, box: Tensor) -> Tensor:
    """CHW tensor, box xyxy в пикселях."""
    x1, y1, x2, y2 = box.detach().cpu().tolist()
    x1i = max(int(x1), 0)
    y1i = max(int(y1), 0)
    x2i = min(int(x2), int(chw.shape[2]))
    y2i = min(int(y2), int(chw.shape[1]))
    if x2i <= x1i or y2i <= y1i:
        return chw
    return chw[:, y1i:y2i, x1i:x2i].contiguous()


def _label_from_largest_fg_box(target: dict[str, Tensor]) -> int:
    boxes = target["boxes"]
    labels = target["labels"]
    if boxes.numel() == 0 or labels.numel() == 0:
        return 0
    fg = labels > 0
    if not bool(fg.any()):
        return max(0, int(labels[0].item()) - 1)
    vb = boxes[fg]
    vl = labels[fg]
    areas = (vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1])
    j = int(torch.argmax(areas).item())
    return max(0, int(vl[j].item()) - 1)


def resolve_roi_classification_label(dataset: Dataset[Any], index: int) -> int:
    """Метка класса для ROI-обучения: ``roi_classification_label`` или эвристика по GT-боксам."""
    if isinstance(dataset, Subset):
        inner_i = int(dataset.indices[index])
        return resolve_roi_classification_label(
            cast(Dataset[Any], dataset.dataset), inner_i
        )
    fn = getattr(dataset, "roi_classification_label", None)
    if callable(fn):
        return int(fn(index))
    _img, target = dataset[index]
    if not isinstance(target, dict):
        return 0
    return _label_from_largest_fg_box(cast(dict[str, Tensor], target))


BoxPolicy = Literal["largest_area"]


class RoiCropClassificationDataset(Sized, Dataset[tuple[Tensor, int]]):
    """Обёртка над детекционным датасетом: на выходе кроп CHW float и int-метка класса."""

    def __init__(
        self,
        detection_dataset: Dataset[tuple[Tensor, dict[str, Tensor]]],
        *,
        box_policy: BoxPolicy = "largest_area",
    ) -> None:
        self._base = detection_dataset
        self._box_policy: BoxPolicy = box_policy
        classes = getattr(detection_dataset, "classes", None)
        self.classes: list[str] | None = (
            list(classes) if isinstance(classes, list) else None
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        _ = self._box_policy
        img, target = self._base[index]
        if not isinstance(img, Tensor):
            msg = "RoiCropClassificationDataset expects CHW image tensor from detection dataset"
            raise TypeError(msg)
        boxes = target["boxes"]
        labels = target["labels"]
        crop_tensor = img
        if boxes.numel() > 0 and labels.numel() > 0:
            fg = labels > 0
            if bool(fg.any()):
                vb = boxes[fg]
                areas = (vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1])
                j = int(torch.argmax(areas).item())
                crop_tensor = _crop_xyxy_chw(img, vb[j])
            else:
                logger.debug(
                    "RoiCropClassificationDataset: no foreground boxes at index %s; full image",
                    index,
                )
        else:
            logger.debug(
                "RoiCropClassificationDataset: empty boxes at index %s; full image",
                index,
            )
        y = resolve_roi_classification_label(self._base, index)
        return crop_tensor, y


def adapt_split_for_roi_classification(split_part: Any) -> RoiCropClassificationDataset:
    """``SplitData.train``/``val`` → датасет *(crop, class_index)* в духе predict после детектора."""
    det = adapt_dataset_for_detection(split_part)
    return RoiCropClassificationDataset(det)
