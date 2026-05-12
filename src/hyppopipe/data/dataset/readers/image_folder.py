from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from hyppopipe.data.dataset.splits import TrainVal, TrainValTest

from hyppopipe.data.dataset.base import ImageDataset
from hyppopipe.data.dataset.protocols import ClassificationConvertible
from hyppopipe.data.dataset.splits import split_random_fractions
from hyppopipe.data.image import SUPPORTED_FILE_TYPES, Image


class ImageFolderDataset(ImageDataset, ClassificationConvertible):
    def __init__(
        self,
        root: str | Path,
        absorb_folders: bool = False,
        *,
        strict: bool = True,
    ):
        self.root = Path(root)
        self._strict = strict
        if absorb_folders:
            self._load_with_nested_tops()
        else:
            self._load_class_dirs_at_root()

    def as_classification_dataset(self) -> Self:
        return self

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image, int]:
        path, target = self.samples[index]
        img = Image.from_path(path, strict=self._strict)
        return img, target

    def _load_class_dirs_at_root(self) -> None:
        r = self.root
        self.classes = sorted(p.name for p in r.iterdir() if p.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        c2i = self.class_to_idx
        self.samples = [
            (str(r / cls / fname), c2i[cls])
            for cls in self.classes
            for fname in os.listdir(r / cls)
            if (r / cls / fname).is_file()
            and (r / cls / fname).suffix.lower() in SUPPORTED_FILE_TYPES
        ]

    def _load_with_nested_tops(self) -> None:
        self.classes = sorted(
            {
                mid.name
                for top in self.root.iterdir()
                if top.is_dir()
                for mid in top.iterdir()
                if mid.is_dir()
            }
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        c2i = self.class_to_idx
        self.samples = [
            (str(d / fname), c2i[cls_name])
            for top in sorted(self.root.iterdir(), key=lambda p: p.name)
            if top.is_dir()
            for cls_name in self.classes
            for d in [top / cls_name]
            if d.is_dir()
            for fname in os.listdir(d)
            if (d / fname).is_file()
        ]


class PairedImageMaskFolderDataset(ImageDataset):
    def __init__(
        self,
        root: str | Path,
        image_folder: str | Path = "images",
        mask_folder: str | Path = "masks",
        *,
        class_names: list[str] | None = None,
        strict: bool = True,
    ):
        self.root = Path(root)
        self.image_folder = self.root / Path(image_folder)
        self.mask_folder = self.root / Path(mask_folder)
        if not self.image_folder.is_dir():
            raise ValueError(f"Image folder {image_folder} is not a directory")
        if not self.mask_folder.is_dir():
            raise ValueError(f"Mask folder {mask_folder} is not a directory")
        self._strict = strict
        if class_names is not None:
            self.classes = list(class_names)
        self.samples = self._build_pairs()

    def as_segmentation_dataset(self, *, kind: str = "semantic") -> Self:
        if kind != "semantic":
            raise ValueError(
                "PairedImageMaskFolderDataset supports only semantic segmentation"
            )
        return self

    def as_split_data(
        self,
        fractions: tuple[float, float] | tuple[float, float, float] = (0.8, 0.2),
        *,
        seed: int | None = None,
    ) -> TrainVal | TrainValTest:
        """Как у ``YAMLDataset.as_split_data``: возвращает ``TrainVal`` / ``TrainValTest`` для ``Trainer``.

        В отличие от YAML, здесь один каталог пар изображение–маска, поэтому сплит
        делается случайно через ``split_random_fractions`` (доли по умолчанию
        ``(0.8, 0.2)`` — train/val).
        """
        return split_random_fractions(self, fractions, seed=seed)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.from_path(image_path, strict=self._strict)
        mask = Image.from_path(mask_path, strict=self._strict)
        return image.body, self._mask_to_class_map(mask)

    def __len__(self) -> int:
        return len(self.samples)

    def _build_pairs(self) -> list[tuple[Path, Path]]:
        image_paths = [
            p
            for p in sorted(self.image_folder.rglob("*"), key=lambda p: p.name)
            if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_TYPES
        ]
        mask_by_stem = {
            p.stem: p
            for p in sorted(self.mask_folder.rglob("*"), key=lambda p: p.name)
            if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_TYPES
        }
        missing = [p.name for p in image_paths if p.stem not in mask_by_stem]
        if missing:
            raise ValueError(
                f"Missing masks for {len(missing)} image(s): {', '.join(missing[:5])}"
            )
        samples = [(p, mask_by_stem[p.stem]) for p in image_paths]
        if not samples:
            raise ValueError(f"No image/mask pairs found under {self.root}")
        return samples

    def _mask_to_class_map(self, mask: Image):
        out = mask.as_gray.squeeze(0).long()
        if out.numel() == 0:
            return out
        unique_values = {int(v) for v in out.unique().tolist()}
        has_single_foreground_class = (
            hasattr(self, "classes") and len(getattr(self, "classes")) == 1
        )
        if unique_values <= {0, 255} or (
            has_single_foreground_class and max(unique_values) > 1
        ):
            return (out > 0).long()
        return out
