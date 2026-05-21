"""Adapt detection splits to ROI crop classification training.

Wraps a detection dataset to yield cropped CHW tensors and integer class labels,
matching the chained predict flow (localize then classify a region).
"""

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
    """Crop a CHW tensor using an xyxy box in pixel coordinates."""
    x1, y1, x2, y2 = box.detach().cpu().tolist()
    x1i = max(int(x1), 0)
    y1i = max(int(y1), 0)
    x2i = min(int(x2), int(chw.shape[2]))
    y2i = min(int(y2), int(chw.shape[1]))
    if x2i <= x1i or y2i <= y1i:
        return chw
    return chw[:, y1i:y2i, x1i:x2i].contiguous()


def _label_from_largest_fg_box(target: dict[str, Tensor]) -> int:
    """Return 0-based class index from the largest foreground box in a detection target."""
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
    """Resolve the 0-based class label for ROI classification at ``index``.

    Uses ``roi_classification_label`` when implemented on the dataset (or inner
    dataset of a ``Subset``); otherwise falls back to the largest foreground
    ground-truth box in the detection target.

    Args:
        dataset: Detection or compatible dataset.
        index: Sample index.

    Returns:
        0-based class index for the ROI at ``index``.
    """
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
    """Wrap a detection dataset to emit ROI crops and class indices.

    Crops the largest foreground bounding box from each sample (or the full image
    when boxes are empty). Class labels are resolved via
    ``resolve_roi_classification_label``.
    """

    def __init__(
        self,
        detection_dataset: Dataset[tuple[Tensor, dict[str, Tensor]]],
        *,
        box_policy: BoxPolicy = "largest_area",
    ) -> None:
        """Create a ROI classification view over a detection dataset.

        Args:
            detection_dataset: Source dataset yielding ``(image, target)`` pairs.
            box_policy: Box selection strategy; only ``"largest_area"`` is used.
        """
        self._base = detection_dataset
        self._box_policy: BoxPolicy = box_policy
        classes = getattr(detection_dataset, "classes", None)
        self.classes: list[str] | None = (
            list(classes) if isinstance(classes, list) else None
        )

    def __len__(self) -> int:
        """Number of samples in the underlying detection dataset."""
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return a cropped CHW tensor and 0-based class index.

        Args:
            index: Sample index.

        Returns:
            Cropped (or full) image tensor and integer class label.

        Raises:
            TypeError: If the base dataset does not return a CHW ``Tensor`` image.
        """
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
    """Adapt a ``SplitData`` train/val part to ROI crop classification samples.

    Resolves detection datasets from split resources (e.g. YAML splits) and
    wraps them so each item is ``(crop_tensor, class_index)``, analogous to
    training after a detector in a chained pipeline.

    Args:
        split_part: A split resource or dataset accepted by
            ``adapt_dataset_for_detection``.

    Returns:
        Dataset yielding cropped tensors and 0-based class indices.
    """
    det = adapt_dataset_for_detection(split_part)
    return RoiCropClassificationDataset(det)
