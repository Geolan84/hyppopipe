from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml
from torch.utils.data import ConcatDataset, Dataset

from hyppopipe.data.dataset.base import ImageDataset
from hyppopipe.data.dataset.errors import InvalidDatasetConfigError
from hyppopipe.data.dataset.splits import TrainVal, TrainValTest
from hyppopipe.data.image import SUPPORTED_FILE_TYPES, Image


def _is_absolute_or_uri(s: str) -> bool:
    return bool(s) and (s.startswith("/") or "://" in s)


def _resolve_path_field(yaml_dir: Path, path_field: str) -> Path:
    p = Path(path_field).expanduser()
    if p.is_absolute() or _is_absolute_or_uri(path_field):
        return p
    if path_field.startswith("./"):
        return (yaml_dir / path_field[2:]).resolve()
    return (yaml_dir / path_field).resolve()


def _resolve_split_entry(yaml_dir: Path, dataset_root: Path, entry: str) -> Path:
    e = Path(entry).expanduser()
    if e.is_absolute() or _is_absolute_or_uri(entry):
        return e
    if entry.startswith("./"):
        return (yaml_dir / entry[2:]).resolve()
    return (dataset_root / entry).resolve()


def _normalize_names(names: object) -> list[str]:
    if isinstance(names, list):
        if not names or not all(isinstance(x, str) for x in names):
            raise InvalidDatasetConfigError("names list must be non-empty list[str]")
        return list(cast(list[str], names))
    if isinstance(names, dict):
        parsed: dict[int, str] = {}
        for raw_k, raw_v in names.items():
            if isinstance(raw_k, str):
                try:
                    k = int(raw_k)
                except ValueError as e:
                    raise InvalidDatasetConfigError(
                        f"names dict keys must be int or numeric str, got {raw_k!r}"
                    ) from e
            elif isinstance(raw_k, int):
                k = raw_k
            else:
                raise InvalidDatasetConfigError(
                    f"names dict keys must be int or numeric str, got {type(raw_k)}"
                )
            if not isinstance(raw_v, str):
                raise InvalidDatasetConfigError("names dict values must be strings")
            parsed[k] = raw_v
        if not parsed:
            raise InvalidDatasetConfigError("names dict must be non-empty")
        max_k = max(parsed)
        expected = set(range(max_k + 1))
        if set(parsed.keys()) != expected:
            raise InvalidDatasetConfigError(
                "names dict must contain contiguous class ids 0..nc-1 "
                f"(got keys {sorted(parsed.keys())})"
            )
        return [parsed[i] for i in range(max_k + 1)]
    raise InvalidDatasetConfigError("names must be a dict or a list of class names")


def _coerce_split_field(raw: object, field_name: str) -> str | list[str]:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        if not raw:
            raise InvalidDatasetConfigError(f"{field_name} list must be non-empty")
        if not all(isinstance(x, str) for x in raw):
            raise InvalidDatasetConfigError(f"{field_name} must be str or list[str]")
        return cast(list[str], raw)
    raise InvalidDatasetConfigError(f"{field_name} must be str or list[str]")


def _parse_config_dict(raw: dict[str, object]) -> YAMLDatasetConfig:
    try:
        path = raw["path"]
        train = raw["train"]
        val = raw["val"]
        names = raw["names"]
    except KeyError as e:
        raise InvalidDatasetConfigError(f"missing required field {e.args[0]!r}") from e

    if not isinstance(path, str):
        raise InvalidDatasetConfigError("path must be a string")

    train_f = _coerce_split_field(train, "train")
    val_f = _coerce_split_field(val, "val")
    names_list = _normalize_names(names)

    test_raw = raw.get("test")
    test_f: str | list[str] | None
    if test_raw is None:
        test_f = None
    elif isinstance(test_raw, str):
        test_f = test_raw
    elif isinstance(test_raw, list):
        test_f = _coerce_split_field(test_raw, "test")
    else:
        raise InvalidDatasetConfigError("test must be str, list[str], or omitted")

    download = raw.get("download")
    if download is not None and not isinstance(download, str):
        raise InvalidDatasetConfigError("download must be a string or omitted")

    nc_raw = raw.get("nc")
    nc: int | None
    if nc_raw is None:
        nc = None
    elif isinstance(nc_raw, int) and nc_raw > 0:
        nc = nc_raw
    else:
        raise InvalidDatasetConfigError("nc must be a positive int or omitted")

    if nc is not None and nc != len(names_list):
        raise InvalidDatasetConfigError(
            f"nc={nc} does not match len(names)={len(names_list)}"
        )

    return YAMLDatasetConfig(
        path=path,
        train=train_f,
        val=val_f,
        names=names_list,
        test=test_f,
        download=download if isinstance(download, str) else None,
        nc=nc if nc is not None else len(names_list),
    )


@dataclass(frozen=True, slots=True)
class YAMLDatasetConfig:
    """Detailed structure is available at wiki: https://deepwiki.com/ultralytics/yolov5/7.1-dataset-configuration"""

    path: str
    train: str | list[str]
    val: str | list[str]
    names: list[str]
    test: str | list[str] | None = None
    download: str | None = None
    nc: int | None = None


class ConcatClassificationDataset(ConcatDataset):
    """``ConcatDataset`` с атрибутом ``classes`` для совместимости с ``infer_num_classes``."""

    def __init__(self, datasets: list[Dataset], classes: list[str]) -> None:
        super().__init__(datasets)
        self.classes = list(classes)

    def as_classification_dataset(self) -> ConcatClassificationDataset:
        return self


class YAMLClassificationSplitDataset(ImageDataset):
    """Один сплит YOLO-классификации: в корне лежат подпапки с именами из ``names``."""

    def __init__(
        self,
        root: Path,
        class_names: list[str],
        *,
        absorb_folders: bool = False,
        strict: bool = True,
    ) -> None:
        self.root = root
        self._strict = strict
        self.classes = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        if absorb_folders:
            self._load_nested_tops()
        else:
            self._load_flat_class_dirs()

    def _collect_ext_files(self, directory: Path) -> list[str]:
        out: list[str] = []
        for fname in os.listdir(directory):
            p = directory / fname
            if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_TYPES:
                out.append(str(p))
        return sorted(out)

    def _image_dir_for_class(self, cls_name: str) -> Path:
        base = self.root / cls_name
        if not base.is_dir():
            raise InvalidDatasetConfigError(
                f"split root {self.root}: missing class folder {cls_name!r}"
            )
        nested = base / "images"
        return nested if nested.is_dir() else base

    def _load_flat_class_dirs(self) -> None:
        c2i = self.class_to_idx
        self.samples = [
            (path, c2i[cls_name])
            for cls_name in self.classes
            for path in self._collect_ext_files(self._image_dir_for_class(cls_name))
        ]

    def _load_nested_tops(self) -> None:
        c2i = self.class_to_idx
        self.samples = []
        tops = sorted(self.root.iterdir(), key=lambda p: p.name)
        for top in tops:
            if not top.is_dir():
                continue
            for cls_name in self.classes:
                d = top / cls_name
                if not d.is_dir():
                    continue
                img_dir = d / "images" if (d / "images").is_dir() else d
                for path in self._collect_ext_files(img_dir):
                    self.samples.append((path, c2i[cls_name]))
        if not self.samples:
            raise InvalidDatasetConfigError(
                f"split root {self.root}: no samples found (absorb_folders=True)"
            )

    def as_classification_dataset(self) -> YAMLClassificationSplitDataset:
        return self

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image, int]:
        path, target = self.samples[index]
        return Image.from_path(path, strict=self._strict), int(target)


def _read_txt_paths(txt_file: Path) -> list[str]:
    lines: list[str] = []
    with txt_file.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def _dataset_for_entry(
    resolved: Path,
    class_names: list[str],
    *,
    absorb_folders: bool,
    dataset_root: Path,
    strict: bool,
) -> Dataset:
    if resolved.is_dir():
        ds = YAMLClassificationSplitDataset(
            resolved,
            class_names,
            absorb_folders=absorb_folders,
            strict=strict,
        )
        if len(ds) == 0:
            raise InvalidDatasetConfigError(f"no images under split path {resolved}")
        return ds

    if resolved.is_file() and resolved.suffix.lower() == ".txt":
        c2i = {c: i for i, c in enumerate(class_names)}
        samples: list[tuple[str, int]] = []
        for line in _read_txt_paths(resolved):
            img_path = Path(line).expanduser()
            if not img_path.is_absolute():
                candidate = (resolved.parent / line).resolve()
                if not candidate.is_file():
                    candidate = (dataset_root / line).resolve()
                img_path = candidate
            if not img_path.is_file():
                raise InvalidDatasetConfigError(
                    f"image path from {resolved.name} not found: {line}"
                )
            cls_name = img_path.parent.name
            if cls_name not in c2i:
                raise InvalidDatasetConfigError(
                    f"parent folder {cls_name!r} of {img_path} is not a class in names"
                )
            if img_path.suffix.lower() not in SUPPORTED_FILE_TYPES:
                continue
            samples.append((str(img_path), c2i[cls_name]))
        if not samples:
            raise InvalidDatasetConfigError(f"no valid images listed in {resolved}")

        class _ListDataset(ImageDataset, Dataset):
            def __init__(
                self,
                items: list[tuple[str, int]],
                classes: list[str],
                *,
                strict: bool,
            ) -> None:
                self._items = items
                self.classes = list(classes)
                self._strict = strict

            def as_classification_dataset(self) -> _ListDataset:
                return self

            def __len__(self) -> int:
                return len(self._items)

            def __getitem__(self, index: int) -> tuple[Image, int]:
                p, y = self._items[index]
                return Image.from_path(p, strict=self._strict), int(y)

        return _ListDataset(samples, class_names, strict=strict)

    raise InvalidDatasetConfigError(
        f"split path must be a directory or .txt file list, got {resolved}"
    )


def _concat_split(
    entries: str | list[str],
    *,
    yaml_dir: Path,
    dataset_root: Path,
    class_names: list[str],
    absorb_folders: bool,
    strict: bool,
) -> Dataset:
    paths = [entries] if isinstance(entries, str) else entries
    parts: list[Dataset] = []
    for entry in paths:
        resolved = _resolve_split_entry(yaml_dir, dataset_root, entry)
        parts.append(
            _dataset_for_entry(
                resolved,
                class_names,
                absorb_folders=absorb_folders,
                dataset_root=dataset_root,
                strict=strict,
            )
        )
    if len(parts) == 1:
        return parts[0]
    return ConcatClassificationDataset(parts, class_names)


def resolve_ultralytics_split_entry(
    yaml_dir: Path, dataset_root: Path, entry: str
) -> Path:
    """Разрешает поле ``train``/``val``/``test`` относительно корня датасета и YAML."""
    return _resolve_split_entry(yaml_dir, dataset_root, entry)


def load_ultralytics_dataset_yaml(
    path_to_yaml: str | Path,
) -> tuple[YAMLDatasetConfig, Path, Path]:
    """Читает YAML датасета в формате Ultralytics/YOLOv5 и возвращает конфиг и пути."""
    path_to_yaml = Path(path_to_yaml).expanduser().resolve()
    if not path_to_yaml.is_file():
        raise InvalidDatasetConfigError(f"YAML file not found: {path_to_yaml}")

    with path_to_yaml.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise InvalidDatasetConfigError("YAML root must be a mapping")

    try:
        config = _parse_config_dict(raw)
    except InvalidDatasetConfigError:
        raise
    except Exception as e:
        raise InvalidDatasetConfigError(str(e)) from e

    yaml_dir = path_to_yaml.parent.resolve()
    dataset_root = _resolve_path_field(yaml_dir, config.path)
    return config, yaml_dir, dataset_root


@dataclass(slots=True)
class YAMLSplitResource:
    """Один сплит из YAML Ultralytics: классификация и детекция строятся по запросу задачи."""

    entries: str | list[str]
    yaml_dir: Path
    dataset_root: Path
    class_names: list[str]
    absorb_folders: bool
    detection_layout: str = "auto"
    strict: bool = True

    def as_classification_dataset(self) -> Dataset:
        return _concat_split(
            self.entries,
            yaml_dir=self.yaml_dir,
            dataset_root=self.dataset_root,
            class_names=self.class_names,
            absorb_folders=self.absorb_folders,
            strict=self.strict,
        )

    def as_detection_dataset(self) -> Dataset:
        from hyppopipe.data.dataset.readers.yaml_detection_dataset import (
            concat_detection_split,
        )

        return concat_detection_split(
            self.entries,
            yaml_dir=self.yaml_dir,
            dataset_root=self.dataset_root,
            class_names=self.class_names,
            layout=self.detection_layout,
        )


class YAMLDataset:
    """Один YAML Ultralytics/YOLOv5: сплиты как ресурсы, задачи вызывают ``as_*_dataset()``."""

    def __init__(
        self,
        path_to_yaml: str | Path,
        *,
        absorb_folders: bool = False,
        detection_layout: str = "auto",
        strict: bool = True,
    ) -> None:
        self.path_to_yaml = Path(path_to_yaml).expanduser().resolve()
        self.config, yaml_dir, dataset_root = load_ultralytics_dataset_yaml(
            self.path_to_yaml
        )
        class_names = self.config.names

        self.classes = class_names
        self.detection_layout = detection_layout
        self.train = YAMLSplitResource(
            self.config.train,
            yaml_dir=yaml_dir,
            dataset_root=dataset_root,
            class_names=class_names,
            absorb_folders=absorb_folders,
            detection_layout=detection_layout,
            strict=strict,
        )
        self.val = YAMLSplitResource(
            self.config.val,
            yaml_dir=yaml_dir,
            dataset_root=dataset_root,
            class_names=class_names,
            absorb_folders=absorb_folders,
            detection_layout=detection_layout,
            strict=strict,
        )
        self.test: YAMLSplitResource | None
        if self.config.test is not None:
            self.test = YAMLSplitResource(
                self.config.test,
                yaml_dir=yaml_dir,
                dataset_root=dataset_root,
                class_names=class_names,
                absorb_folders=absorb_folders,
                detection_layout=detection_layout,
                strict=strict,
            )
        else:
            self.test = None

    def as_split_data(self) -> TrainVal | TrainValTest:
        """Возвращает структуру для ``Trainer`` / ``Pipeline.train``."""
        if self.test is not None:
            return TrainValTest(train=self.train, val=self.val, test=self.test)
        return TrainVal(train=self.train, val=self.val)
