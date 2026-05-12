"""Датасеты сегментации из YAML Ultralytics/YOLOv5."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torchvision.io import ImageReadMode, read_image

from hyppopipe.data.dataset.errors import InvalidDatasetConfigError
from hyppopipe.data.dataset.readers.yaml_dataset import resolve_ultralytics_split_entry
from hyppopipe.data.dataset.readers.yaml_detection_dataset import (
    DetectionLayout,
    _iter_images_rglob,
    _swap_images_dir_to_labels,
)

SegmentationKind = Literal["instance", "semantic"]


def _clip_normalized(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _normalized_polygon_from_yolo_parts(
    parts: list[str],
) -> tuple[int, list[float]] | None:
    if len(parts) == 5:
        cls_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        coords = [x1, y1, x2, y1, x2, y2, x1, y2]
    elif len(parts) >= 7 and len(parts) % 2 == 1:
        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
    else:
        return None
    return cls_id, [_clip_normalized(x) for x in coords]


def _polygon_points(
    coords: list[float], *, img_w: int, img_h: int
) -> np.ndarray[Any, Any]:
    points: list[list[int]] = []
    for x_norm, y_norm in zip(coords[0::2], coords[1::2], strict=True):
        x = int(round(x_norm * (img_w - 1)))
        y = int(round(y_norm * (img_h - 1)))
        points.append([max(0, min(x, img_w - 1)), max(0, min(y, img_h - 1))])
    return np.asarray(points, dtype=np.int32)


def _empty_instance_target(img_h: int, img_w: int) -> dict[str, Tensor]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "masks": torch.zeros((0, img_h, img_w), dtype=torch.uint8),
        "area": torch.zeros((0,), dtype=torch.float32),
        "iscrowd": torch.zeros((0,), dtype=torch.int64),
    }


def parse_yolo_segmentation_instance_target(
    text: str,
    *,
    img_w: int,
    img_h: int,
    num_foreground_classes: int,
) -> dict[str, Tensor]:
    """Парсит YOLO bbox/polygon строки в target для torchvision Mask R-CNN."""
    boxes: list[list[float]] = []
    labels: list[int] = []
    masks: list[Tensor] = []
    areas: list[float] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = _normalized_polygon_from_yolo_parts(line.split())
        if parsed is None:
            continue
        cls_id, coords = parsed
        if not (0 <= cls_id < num_foreground_classes):
            raise ValueError(
                f"class id {cls_id} out of range [0, {num_foreground_classes - 1}]"
            )

        mask_np = np.zeros((img_h, img_w), dtype=np.uint8)
        points = _polygon_points(coords, img_w=img_w, img_h=img_h)
        cv2.fillPoly(mask_np, [points], 1)
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max() + 1)
        y2 = float(ys.max() + 1)
        boxes.append([x1, y1, x2, y2])
        labels.append(cls_id + 1)
        masks.append(torch.from_numpy(mask_np))
        areas.append(float(mask_np.sum()))

    if not masks:
        return _empty_instance_target(img_h, img_w)

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "masks": torch.stack(masks).to(dtype=torch.uint8),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
    }


def parse_yolo_segmentation_semantic_mask(
    text: str,
    *,
    img_w: int,
    img_h: int,
    num_foreground_classes: int,
) -> Tensor:
    """Парсит YOLO bbox/polygon строки в semantic class-map с фоном 0."""
    mask_np = np.zeros((img_h, img_w), dtype=np.int32)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = _normalized_polygon_from_yolo_parts(line.split())
        if parsed is None:
            continue
        cls_id, coords = parsed
        if not (0 <= cls_id < num_foreground_classes):
            raise ValueError(
                f"class id {cls_id} out of range [0, {num_foreground_classes - 1}]"
            )
        points = _polygon_points(coords, img_w=img_w, img_h=img_h)
        cv2.fillPoly(mask_np, [points], cls_id + 1)
    return torch.from_numpy(mask_np).to(dtype=torch.int64)


class ConcatSegmentationDataset(ConcatDataset):
    def __init__(
        self,
        datasets: list[Dataset],
        classes: list[str],
        *,
        kind: SegmentationKind,
    ) -> None:
        super().__init__(datasets)
        self.classes = list(classes)
        self.kind = kind

    def as_segmentation_dataset(
        self, *, kind: SegmentationKind | None = None
    ) -> ConcatSegmentationDataset:
        if kind is not None and kind != self.kind:
            raise InvalidDatasetConfigError(
                f"requested {kind} segmentation from {self.kind} dataset"
            )
        return self


class YAMLSegmentationSplitDataset(Dataset[Any]):
    """Один YOLO-сплит с polygon/bbox labels для semantic или instance segmentation."""

    def __init__(
        self,
        split_root: Path,
        class_names: list[str],
        *,
        kind: SegmentationKind,
        layout: DetectionLayout = "auto",
    ) -> None:
        self.split_root = split_root.resolve()
        self.classes = list(class_names)
        self.kind = kind
        self.num_foreground_classes = len(class_names)
        self._layout: Literal["nested_class", "flat_yolo"]
        if layout == "auto":
            self._layout = self._infer_layout()
        else:
            self._layout = layout
        self.samples: list[tuple[Path, Path | None]] = []
        self._build_index()
        if not self.samples:
            raise InvalidDatasetConfigError(
                f"no segmentation samples found under {self.split_root}"
            )

    def as_segmentation_dataset(
        self, *, kind: SegmentationKind | None = None
    ) -> YAMLSegmentationSplitDataset:
        if kind is not None and kind != self.kind:
            raise InvalidDatasetConfigError(
                f"requested {kind} segmentation from {self.kind} dataset"
            )
        return self

    def _infer_layout(self) -> Literal["nested_class", "flat_yolo"]:
        for c in self.classes:
            if (self.split_root / c / "images").is_dir():
                return "nested_class"
        if "images" in self.split_root.parts:
            return "flat_yolo"
        raise InvalidDatasetConfigError(
            f"Cannot infer segmentation layout under {self.split_root}. "
            "Expected either …/<Class>/images/ (nested_class) or a path containing "
            "segment 'images' (flat_yolo). Pass layout= explicitly."
        )

    def _build_index(self) -> None:
        if self._layout == "nested_class":
            self._build_nested_class_index()
        else:
            self._build_flat_yolo_index()

    def _build_nested_class_index(self) -> None:
        for cls_name in self.classes:
            cls_dir = self.split_root / cls_name
            if not cls_dir.is_dir():
                continue
            img_root = cls_dir / "images" if (cls_dir / "images").is_dir() else cls_dir
            lbl_root = cls_dir / "labels" if (cls_dir / "labels").is_dir() else None
            for img_path in _iter_images_rglob(img_root):
                lbl_path: Path | None = None
                if lbl_root is not None:
                    cand = lbl_root / f"{img_path.stem}.txt"
                    if cand.is_file():
                        lbl_path = cand
                self.samples.append((img_path, lbl_path))

    def _build_flat_yolo_index(self) -> None:
        labels_root = _swap_images_dir_to_labels(self.split_root)
        for img_path in _iter_images_rglob(self.split_root):
            rel = img_path.relative_to(self.split_root)
            cand = labels_root / rel.with_suffix(".txt")
            self.samples.append((img_path, cand if cand.is_file() else None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        img_path, lbl_path = self.samples[index]
        img = read_image(str(img_path), mode=ImageReadMode.RGB).float() / 255.0
        _, h, w = img.shape
        text = lbl_path.read_text(encoding="utf-8") if lbl_path is not None else ""
        if self.kind == "instance":
            target = parse_yolo_segmentation_instance_target(
                text,
                img_w=w,
                img_h=h,
                num_foreground_classes=self.num_foreground_classes,
            )
            return img, target
        mask = parse_yolo_segmentation_semantic_mask(
            text,
            img_w=w,
            img_h=h,
            num_foreground_classes=self.num_foreground_classes,
        )
        return img, mask


def _dataset_for_split_root(
    resolved: Path,
    class_names: list[str],
    *,
    kind: SegmentationKind,
    layout: DetectionLayout,
) -> Dataset:
    return YAMLSegmentationSplitDataset(
        resolved,
        class_names,
        kind=kind,
        layout=layout,
    )


def concat_segmentation_split(
    entries: str | list[str],
    *,
    yaml_dir: Path,
    dataset_root: Path,
    class_names: list[str],
    kind: SegmentationKind | str,
    layout: DetectionLayout | str = "auto",
) -> Dataset:
    paths = [entries] if isinstance(entries, str) else entries
    seg_kind = cast(SegmentationKind, kind)
    lay = cast(DetectionLayout, layout)
    parts: list[Dataset] = []
    for entry in paths:
        resolved = resolve_ultralytics_split_entry(yaml_dir, dataset_root, entry)
        parts.append(
            _dataset_for_split_root(
                resolved,
                class_names,
                kind=seg_kind,
                layout=lay,
            )
        )
    if len(parts) == 1:
        return parts[0]
    return ConcatSegmentationDataset(parts, class_names, kind=seg_kind)
