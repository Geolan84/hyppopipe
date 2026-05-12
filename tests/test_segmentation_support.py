from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from hyppopipe.data.dataset.readers.image_folder import PairedImageMaskFolderDataset
from hyppopipe.data.dataset.readers.yaml_segmentation_dataset import (
    parse_yolo_segmentation_instance_target,
    parse_yolo_segmentation_semantic_mask,
)
from hyppopipe.pipeline.image.segmentation import ImageSegmentator
from hyppopipe.train.tasks.dispatch import dispatch_training_task
from hyppopipe.train.tasks.segmentation import (
    SegmentationTrainingTask,
    adapt_semantic_segmentation_model,
    infer_segmentation_backend,
    resolve_segmentation_data_kind,
)


def test_yolo_polygon_parses_instance_masks() -> None:
    target = parse_yolo_segmentation_instance_target(
        "0 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0",
        img_w=4,
        img_h=4,
        num_foreground_classes=1,
    )

    assert target["labels"].tolist() == [1]
    assert target["masks"].shape == (1, 4, 4)
    torch.testing.assert_close(
        target["boxes"],
        torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
    )


def test_yolo_bbox_line_can_build_semantic_mask() -> None:
    mask = parse_yolo_segmentation_semantic_mask(
        "1 0.5 0.5 1.0 1.0",
        img_w=4,
        img_h=4,
        num_foreground_classes=2,
    )

    assert mask.shape == (4, 4)
    assert int(mask.max().item()) == 2


def test_paired_image_mask_folder_returns_semantic_class_map(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    write_png(torch.zeros(3, 4, 4, dtype=torch.uint8), str(images / "case.png"))
    mask = torch.zeros(1, 4, 4, dtype=torch.uint8)
    mask[:, 1:3, 1:3] = 255
    write_png(mask, str(masks / "case.png"))

    dataset = PairedImageMaskFolderDataset(tmp_path)
    image, class_map = dataset[0]

    assert image.shape == (3, 4, 4)
    assert class_map.shape == (4, 4)
    assert sorted(class_map.unique().tolist()) == [0, 1]
    assert dataset.as_segmentation_dataset(kind="semantic") is dataset


def test_paired_image_mask_as_split_data_train_val(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for i in range(10):
        write_png(torch.zeros(3, 4, 4, dtype=torch.uint8), str(images / f"im{i}.png"))
        write_png(torch.zeros(1, 4, 4, dtype=torch.uint8), str(masks / f"im{i}.png"))

    dataset = PairedImageMaskFolderDataset(tmp_path)
    split = dataset.as_split_data((0.7, 0.3), seed=0)

    assert len(split.train) + len(split.val) == len(dataset)


def test_dispatch_resolves_segmentation_task() -> None:
    task = dispatch_training_task(ImageSegmentator(kind="semantic"))

    assert isinstance(task, SegmentationTrainingTask)


def test_adapt_semantic_segmentation_model_replaces_last_conv() -> None:
    model = torch.nn.Module()
    model.classifier = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 2, kernel_size=1),
    )

    adapt_semantic_segmentation_model(model, num_classes=5)

    assert model.classifier[-1].out_channels == 5


def test_infer_segmentation_backend_detects_conv_classifier() -> None:
    model = torch.nn.Module()
    model.classifier = torch.nn.Sequential(torch.nn.Conv2d(3, 2, kernel_size=1))

    assert infer_segmentation_backend(model) == "semantic"


def test_infer_segmentation_backend_detects_mask_rcnn_heads() -> None:
    class _Roi(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mask_predictor = torch.nn.Identity()

    class _Det(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.roi_heads = _Roi()

    assert infer_segmentation_backend(_Det()) == "mask_rcnn"


def test_resolve_segmentation_data_kind_mask_rcnn_vs_semantic() -> None:
    assert (
        resolve_segmentation_data_kind(user_kind="instance", backend="mask_rcnn")
        == "instance"
    )
    assert (
        resolve_segmentation_data_kind(user_kind="semantic", backend="semantic")
        == "semantic"
    )
    assert (
        resolve_segmentation_data_kind(user_kind="instance", backend="semantic")
        == "semantic"
    )
    with pytest.raises(ValueError, match="Mask R-CNN"):
        resolve_segmentation_data_kind(user_kind="semantic", backend="mask_rcnn")
