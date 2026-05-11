from __future__ import annotations

import torch
from torch.utils.data import Dataset

from hyppopipe.data.dataset.adapters.roi_classification import (
    RoiCropClassificationDataset,
)
from hyppopipe.pipeline.step import normalize_step_inputs


def test_normalize_step_inputs_prefers_input_first() -> None:
    assert normalize_step_inputs({"__input__", "b", "a"}) == ("__input__", "a", "b")


def test_normalize_step_inputs_sorted_without_input() -> None:
    assert normalize_step_inputs({"z", "a"}) == ("a", "z")


def test_normalize_step_inputs_tuple_preserved() -> None:
    assert normalize_step_inputs(("z", "a")) == ("z", "a")


class _TinyDet(Dataset):
    classes = ["bg_unused", "fg"]

    def roi_classification_label(self, index: int) -> int:
        del index
        return 1

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del index
        img = torch.zeros(3, 100, 100)
        boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0], [0.0, 0.0, 50.0, 50.0]])
        labels = torch.tensor([1, 1], dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels}


def test_roi_crop_largest_fg_box() -> None:
    ds = RoiCropClassificationDataset(_TinyDet())
    crop, y = ds[0]
    assert crop.shape == (3, 50, 50)
    assert y == 1


class _EmptyBoxes(Dataset):
    classes = ["a"]

    def roi_classification_label(self, index: int) -> int:
        del index
        return 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del index
        img = torch.ones(3, 10, 10)
        z = torch.zeros((0, 4), dtype=torch.float32)
        return img, {"boxes": z, "labels": torch.zeros(0, dtype=torch.int64)}


def test_roi_crop_empty_boxes_full_image() -> None:
    ds = RoiCropClassificationDataset(_EmptyBoxes())
    crop, y = ds[0]
    assert crop.shape == (3, 10, 10)
    assert y == 0
