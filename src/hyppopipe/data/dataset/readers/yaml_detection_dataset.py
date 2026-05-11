"""Датасеты детекции из YAML Ultralytics/YOLOv5 (разметка в формате YOLO bbox)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torchvision.io import ImageReadMode, read_image

from hyppopipe.data.dataset.errors import InvalidDatasetConfigError
from hyppopipe.data.dataset.readers.yaml_dataset import resolve_ultralytics_split_entry
from hyppopipe.data.image import SUPPORTED_FILE_TYPES

DetectionLayout = Literal["auto", "nested_class", "flat_yolo"]


def _swap_images_dir_to_labels(split_images_root: Path) -> Path:
    parts = list(split_images_root.parts)
    try:
        i = parts.index("images")
    except ValueError as e:
        raise InvalidDatasetConfigError(
            "flat_yolo layout requires 'images' in the split path "
            f"(got {split_images_root})"
        ) from e
    parts[i] = "labels"
    return Path(*parts)


def _iter_images(directory: Path) -> list[Path]:
    out: list[Path] = []
    for fname in sorted(os.listdir(directory)):
        p = directory / fname
        if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_TYPES:
            out.append(p)
    return out


def _iter_images_rglob(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_TYPES
    )


def _flat_yolo_roi_cls_from_label_file(
    lbl_path: Path | None, *, num_foreground_classes: int
) -> int:
    """0-based класс для ROI-классификации: YOLO-класс бокса с максимальной нормированной площадью."""
    if lbl_path is None or not lbl_path.is_file():
        return 0
    text = lbl_path.read_text(encoding="utf-8")
    best_area = -1.0
    best_cls = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        _cx, _cy, w, h = map(float, parts[1:5])
        area = w * h
        if area > best_area:
            best_area = area
            best_cls = cls_id
    if best_area < 0:
        return 0
    if not (0 <= best_cls < num_foreground_classes):
        raise ValueError(
            f"{lbl_path}: class id {best_cls} out of range "
            f"[0, {num_foreground_classes - 1}]"
        )
    return best_cls


def _parse_yolo_bbox_lines(
    text: str, *, img_w: int, img_h: int
) -> tuple[Tensor, Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        x1 = max(0.0, min(x1, float(img_w)))
        x2 = max(0.0, min(x2, float(img_w)))
        y1 = max(0.0, min(y1, float(img_h)))
        y2 = max(0.0, min(y2, float(img_h)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(cls_id)
    if not boxes:
        z = torch.zeros((0, 4), dtype=torch.float32)
        return z, torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
        labels, dtype=torch.int64
    )


class ConcatDetectionDataset(ConcatDataset):
    def __init__(self, datasets: list[Dataset], classes: list[str]) -> None:
        super().__init__(datasets)
        self.classes = list(classes)

    def as_detection_dataset(self) -> ConcatDetectionDataset:
        return self

    def roi_classification_label(self, index: int) -> int:
        i = int(index)
        for ds in self.datasets:
            n = len(ds)
            if i < n:
                fn = getattr(ds, "roi_classification_label", None)
                if fn is None:
                    msg = (
                        f"{type(ds).__name__} does not implement roi_classification_label; "
                        "use YAML/ConcatDetection datasets or rely on box-derived labels"
                    )
                    raise TypeError(msg)
                return int(fn(i))
            i -= n
        raise IndexError(index)


class YAMLDetectionSplitDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Один сплит: изображения и YOLO-.txt с bbox (нормализованные cx,cy,w,h)."""

    def __init__(
        self,
        split_root: Path,
        class_names: list[str],
        *,
        layout: DetectionLayout = "auto",
    ) -> None:
        self.split_root = split_root.resolve()
        self.classes = list(class_names)
        self.num_foreground_classes = len(class_names)
        self._layout: Literal["nested_class", "flat_yolo"]
        if layout == "auto":
            self._layout = self._infer_layout()
        else:
            self._layout = layout
        self.samples: list[tuple[Path, Path | None]] = []
        self._roi_cls_label_per_sample: list[int] = []
        self._build_index()
        if not self.samples:
            raise InvalidDatasetConfigError(
                f"no detection samples found under {self.split_root}"
            )

    def as_detection_dataset(self) -> YAMLDetectionSplitDataset:
        return self

    def roi_classification_label(self, index: int) -> int:
        return int(self._roi_cls_label_per_sample[index])

    def _infer_layout(self) -> Literal["nested_class", "flat_yolo"]:
        for c in self.classes:
            if (self.split_root / c / "images").is_dir():
                return "nested_class"
        if "images" in self.split_root.parts:
            return "flat_yolo"
        raise InvalidDatasetConfigError(
            f"Cannot infer detection layout under {self.split_root}. "
            "Expected either …/<Class>/images/ (nested_class) or a path containing "
            "segment 'images' (flat_yolo). Pass layout= explicitly."
        )

    def _build_index(self) -> None:
        if self._layout == "nested_class":
            self._build_nested_class_index()
        else:
            self._build_flat_yolo_index()

    def _build_nested_class_index(self) -> None:
        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = self.split_root / cls_name
            if not cls_dir.is_dir():
                continue
            img_root = cls_dir / "images" if (cls_dir / "images").is_dir() else cls_dir
            lbl_root = cls_dir / "labels" if (cls_dir / "labels").is_dir() else None
            for img_path in _iter_images(img_root):
                lbl_path: Path | None = None
                if lbl_root is not None:
                    cand = lbl_root / f"{img_path.stem}.txt"
                    if cand.is_file():
                        lbl_path = cand
                self.samples.append((img_path, lbl_path))
                self._roi_cls_label_per_sample.append(cls_idx)

    def _build_flat_yolo_index(self) -> None:
        labels_root = _swap_images_dir_to_labels(self.split_root)
        for img_path in _iter_images_rglob(self.split_root):
            rel = img_path.relative_to(self.split_root)
            cand = labels_root / rel.with_suffix(".txt")
            lbl_path = cand if cand.is_file() else None
            self.samples.append((img_path, lbl_path))
            cls_lbl = _flat_yolo_roi_cls_from_label_file(
                lbl_path, num_foreground_classes=self.num_foreground_classes
            )
            self._roi_cls_label_per_sample.append(cls_lbl)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        img_path, lbl_path = self.samples[index]
        img = read_image(str(img_path), mode=ImageReadMode.RGB).float() / 255.0
        _, h, w = img.shape
        text = lbl_path.read_text(encoding="utf-8") if lbl_path is not None else ""
        boxes, labels = _parse_yolo_bbox_lines(text, img_w=w, img_h=h)
        if labels.numel() > 0:
            bad = (labels < 0) | (labels >= self.num_foreground_classes)
            if bad.any():
                raise ValueError(
                    f"{img_path}: class id out of range [0, {self.num_foreground_classes - 1}]"
                )
            labels = labels + 1
        target = {"boxes": boxes, "labels": labels}
        return img, target


def _dataset_for_split_root(
    resolved: Path,
    class_names: list[str],
    *,
    layout: DetectionLayout,
) -> Dataset:
    ds = YAMLDetectionSplitDataset(resolved, class_names, layout=layout)
    return ds


def concat_detection_split(
    entries: str | list[str],
    *,
    yaml_dir: Path,
    dataset_root: Path,
    class_names: list[str],
    layout: DetectionLayout | str = "auto",
) -> Dataset:
    paths = [entries] if isinstance(entries, str) else entries
    lay = cast(DetectionLayout, layout)
    parts: list[Dataset] = []
    for entry in paths:
        resolved = resolve_ultralytics_split_entry(yaml_dir, dataset_root, entry)
        parts.append(_dataset_for_split_root(resolved, class_names, layout=lay))
    if len(parts) == 1:
        return parts[0]
    return ConcatDetectionDataset(parts, class_names)
