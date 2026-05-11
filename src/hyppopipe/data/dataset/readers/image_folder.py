from __future__ import annotations

import os
from operator import attrgetter
from pathlib import Path
from typing import Self

from hyppopipe.data.dataset import ImageDataset
from hyppopipe.data.dataset.protocols import ClassificationConvertible
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
        strict: bool = True,
    ):
        self.root = Path(root)
        self.image_folder = self.root / Path(image_folder)
        self.mask_folder = self.root / Path(mask_folder)
        if not self.image_folder.is_dir():
            raise ValueError(f"Image folder {self.image_folder} is not a directory")
        if not self.mask_folder.is_dir():
            raise ValueError(f"Mask folder {self.mask_folder} is not a directory")
        self._strict = strict
        self.samples = list(
            zip(
                sorted(self.image_folder.rglob("*"), key=attrgetter("name")),
                sorted(self.mask_folder.rglob("*"), key=attrgetter("name")),
            )
        )

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.from_path(image_path, strict=self._strict)
        mask = Image.from_path(mask_path, strict=self._strict)
        return image.body, mask.as_gray

    def __len__(self) -> int:
        return len(self.samples)
