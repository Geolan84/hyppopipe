from __future__ import annotations

from pathlib import Path

import torch

from hyppopipe.data.dataset.readers.yaml_detection_dataset import (
    _flat_yolo_roi_cls_from_label_file,
    _parse_yolo_bbox_lines,
)


def test_parse_yolo_bbox_lines_keeps_bbox_format() -> None:
    boxes, labels = _parse_yolo_bbox_lines(
        "1 0.5 0.5 0.2 0.4",
        img_w=100,
        img_h=50,
    )

    assert labels.tolist() == [1]
    torch.testing.assert_close(
        boxes,
        torch.tensor([[40.0, 15.0, 60.0, 35.0]], dtype=torch.float32),
    )


def test_parse_yolo_segmentation_line_as_detection_bbox() -> None:
    boxes, labels = _parse_yolo_bbox_lines(
        "2 0.25 0.10 0.75 0.20 0.50 0.90",
        img_w=200,
        img_h=100,
    )

    assert labels.tolist() == [2]
    torch.testing.assert_close(
        boxes,
        torch.tensor([[50.0, 10.0, 150.0, 90.0]], dtype=torch.float32),
    )


def test_flat_yolo_roi_cls_uses_polygon_bbox_area(tmp_path: Path) -> None:
    label_path = tmp_path / "sample.txt"
    label_path.write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.1 0.1",
                "1 0.10 0.10 0.90 0.20 0.80 0.80",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        _flat_yolo_roi_cls_from_label_file(
            label_path,
            num_foreground_classes=2,
        )
        == 1
    )
