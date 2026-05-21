"""Ultralytics/YOLOv5 YAML dataset configuration and classification loaders.

Reads ``data.yaml``-style configs, resolves split paths, and exposes train/val/test
resources that can be materialized as classification, detection, or segmentation
datasets via ``YAMLDataset`` and ``YAMLSplitResource``.
"""

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
    """Return True if ``s`` is an absolute path or URI-like string."""
    return bool(s) and (s.startswith("/") or "://" in s)


def _resolve_path_field(yaml_dir: Path, path_field: str) -> Path:
    """Resolve the YAML ``path`` field to an absolute dataset root."""
    p = Path(path_field).expanduser()
    if p.is_absolute() or _is_absolute_or_uri(path_field):
        return p
    if path_field.startswith("./"):
        return (yaml_dir / path_field[2:]).resolve()
    return (yaml_dir / path_field).resolve()


def _resolve_split_entry(yaml_dir: Path, dataset_root: Path, entry: str) -> Path:
    """Resolve a train/val/test entry path relative to YAML dir or dataset root."""
    e = Path(entry).expanduser()
    if e.is_absolute() or _is_absolute_or_uri(entry):
        return e
    if entry.startswith("./"):
        return (yaml_dir / entry[2:]).resolve()
    return (dataset_root / entry).resolve()


def _normalize_names(names: object) -> list[str]:
    """Parse ``names`` from YAML into an ordered list of class name strings.

    Accepts a non-empty ``list[str]`` or a dict mapping contiguous class ids
    ``0 .. nc-1`` to names (keys may be ``int`` or numeric strings).

    Args:
        names: Raw ``names`` field from the dataset YAML.

    Returns:
        Class names in index order.

    Raises:
        InvalidDatasetConfigError: If the value is empty or malformed.
    """
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
    """Normalize a YAML split field to a string or non-empty list of strings."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        if not raw:
            raise InvalidDatasetConfigError(f"{field_name} list must be non-empty")
        if not all(isinstance(x, str) for x in raw):
            raise InvalidDatasetConfigError(f"{field_name} must be str or list[str]")
        return cast(list[str], raw)
    raise InvalidDatasetConfigError(f"{field_name} must be str or list[str]")


def _parse_config_dict(raw: dict[str, object], path: Path) -> YAMLDatasetConfig:
    """Build ``YAMLDatasetConfig`` from a parsed YAML mapping.

    Validates required keys ``train``, ``val``, and ``names``, optional ``test``,
    ``download``, and ``nc``, and normalizes path and name fields.

    Args:
        raw: Root mapping loaded from the dataset YAML file.
        path: Default dataset root if ``path`` key is omitted.

    Returns:
        Frozen configuration dataclass.

    Raises:
        InvalidDatasetConfigError: On missing keys or invalid field types.
    """
    try:
        path = raw.get("path", str(path))
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
    """Parsed Ultralytics/YOLOv5 ``data.yaml`` fields.

    Field semantics match the upstream dataset configuration format; see
    `Ultralytics dataset configuration`_ for layout details.

    Attributes:
        path: Dataset root path string from YAML (may be relative).
        train: Train split path or list of paths.
        val: Validation split path or list of paths.
        names: Ordered class names.
        test: Optional test split path(s).
        download: Optional download script or URL hint.
        nc: Number of classes (defaults to ``len(names)`` when omitted).

    .. _Ultralytics dataset configuration: https://deepwiki.com/ultralytics/yolov5/7.1-dataset-configuration
    """

    path: str
    train: str | list[str]
    val: str | list[str]
    names: list[str]
    test: str | list[str] | None = None
    download: str | None = None
    nc: int | None = None


class ConcatClassificationDataset(ConcatDataset):
    """``ConcatDataset`` with a ``classes`` attribute for ``infer_num_classes``."""

    def __init__(self, datasets: list[Dataset], classes: list[str]) -> None:
        """Concatenate split datasets and expose shared class names.

        Args:
            datasets: Per-entry classification datasets to concatenate.
            classes: Class names shared by all parts.
        """
        super().__init__(datasets)
        self.classes = list(classes)

    def as_classification_dataset(self) -> ConcatClassificationDataset:
        """Return this concatenated dataset for classification training."""
        return self


class YAMLClassificationSplitDataset(ImageDataset):
    """One YAML classification split: class subfolders under ``root``."""

    def __init__(
        self,
        root: Path,
        class_names: list[str],
        *,
        absorb_folders: bool = False,
        strict: bool = True,
    ) -> None:
        """Index images for a single train/val/test entry.

        Args:
            root: Resolved split directory or list root.
            class_names: Class names from the YAML ``names`` field.
            absorb_folders: If True, allow ``root/<top>/<class>/`` nesting.
            strict: Passed to ``Image.from_path`` when loading samples.
        """
        self.root = root
        self._strict = strict
        self.classes = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        if absorb_folders:
            self._load_nested_tops()
        else:
            self._load_flat_class_dirs()

    def _collect_ext_files(self, directory: Path) -> list[str]:
        """List supported image file paths under ``directory``."""
        out: list[str] = []
        for fname in os.listdir(directory):
            p = directory / fname
            if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_TYPES:
                out.append(str(p))
        return sorted(out)

    def _image_dir_for_class(self, cls_name: str) -> Path:
        """Return the image directory for a class (``<class>/images`` or ``<class>``)."""
        base = self.root / cls_name
        if not base.is_dir():
            raise InvalidDatasetConfigError(
                f"split root {self.root}: missing class folder {cls_name!r}"
            )
        nested = base / "images"
        return nested if nested.is_dir() else base

    def _load_flat_class_dirs(self) -> None:
        """Index samples from class subfolders directly under ``root``."""
        c2i = self.class_to_idx
        self.samples = [
            (path, c2i[cls_name])
            for cls_name in self.classes
            for path in self._collect_ext_files(self._image_dir_for_class(cls_name))
        ]

    def _load_nested_tops(self) -> None:
        """Index samples from ``root/<top>/<class>/`` when ``absorb_folders`` is True."""
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
        """Return this split dataset for classification training."""
        return self

    def __len__(self) -> int:
        """Number of indexed samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image, int]:
        """Load image and class index for ``index``.

        Args:
            index: Sample index.

        Returns:
            Loaded ``Image`` and integer class index.
        """
        path, target = self.samples[index]
        return Image.from_path(path, strict=self._strict), int(target)


def _read_txt_paths(txt_file: Path) -> list[str]:
    """Read non-empty, non-comment lines from a split ``.txt`` file."""
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
    """Materialize one split entry as a classification ``Dataset``.

    Supports a class-folder directory, or a ``.txt`` file listing image paths
    (class inferred from the parent folder name).

    Args:
        resolved: Absolute path to the split entry.
        class_names: Allowed class names from YAML.
        absorb_folders: Forwarded to ``YAMLClassificationSplitDataset``.
        dataset_root: Dataset root for resolving relative paths in ``.txt`` lists.
        strict: Passed when loading images.

    Returns:
        A non-empty classification dataset.

    Raises:
        InvalidDatasetConfigError: If the entry is empty or paths are invalid.
    """
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
            """Classification dataset backed by a fixed list of path/label pairs."""

            def __init__(
                self,
                items: list[tuple[str, int]],
                classes: list[str],
                *,
                strict: bool,
            ) -> None:
                """Store indexed samples and class metadata."""
                self._items = items
                self.classes = list(classes)
                self._strict = strict

            def as_classification_dataset(self) -> _ListDataset:
                """Return this list-backed dataset for classification training."""
                return self

            def __len__(self) -> int:
                """Number of indexed samples."""
                return len(self._items)

            def __getitem__(self, index: int) -> tuple[Image, int]:
                """Load image and class index for ``index``."""
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
    """Build one classification dataset (or concat) for YAML split entries."""
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
    """Resolve a ``train``/``val``/``test`` path relative to YAML and dataset root.

    Args:
        yaml_dir: Directory containing the dataset YAML file.
        dataset_root: Resolved ``path`` field from the config.
        entry: Split path string from the YAML.

    Returns:
        Absolute filesystem path to the split entry.
    """
    return _resolve_split_entry(yaml_dir, dataset_root, entry)


def load_ultralytics_dataset_yaml(
    path_to_yaml: str | Path,
) -> tuple[YAMLDatasetConfig, Path, Path]:
    """Load and validate an Ultralytics/YOLOv5 dataset YAML file.

    Args:
        path_to_yaml: Path to ``data.yaml`` (or equivalent).

    Returns:
        A triple of parsed config, YAML parent directory, and resolved dataset root.

    Raises:
        InvalidDatasetConfigError: If the file is missing or fields are invalid.
    """
    path_to_yaml = Path(path_to_yaml).expanduser().resolve()
    if not path_to_yaml.is_file():
        raise InvalidDatasetConfigError(f"YAML file not found: {path_to_yaml}")

    with path_to_yaml.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise InvalidDatasetConfigError("YAML root must be a mapping")

    try:
        config = _parse_config_dict(raw, path_to_yaml.parent)
    except InvalidDatasetConfigError:
        raise
    except Exception as e:
        raise InvalidDatasetConfigError(str(e)) from e

    yaml_dir = path_to_yaml.parent.resolve()
    dataset_root = _resolve_path_field(yaml_dir, config.path)
    return config, yaml_dir, dataset_root


@dataclass(slots=True)
class YAMLSplitResource:
    """Lazy handle for one YAML split (train, val, or test).

    Classification, detection, and segmentation datasets are built on demand
    via ``as_classification_dataset``, ``as_detection_dataset``, and
    ``as_segmentation_dataset``.
    """

    entries: str | list[str]
    yaml_dir: Path
    dataset_root: Path
    class_names: list[str]
    absorb_folders: bool
    detection_layout: str = "auto"
    strict: bool = True

    def as_classification_dataset(self) -> Dataset:
        """Build a classification dataset for this split."""
        return _concat_split(
            self.entries,
            yaml_dir=self.yaml_dir,
            dataset_root=self.dataset_root,
            class_names=self.class_names,
            absorb_folders=self.absorb_folders,
            strict=self.strict,
        )

    def as_detection_dataset(self) -> Dataset:
        """Build a detection dataset for this split."""
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

    def as_segmentation_dataset(self, *, kind: str = "instance") -> Dataset:
        """Build a segmentation dataset for this split.

        Args:
            kind: ``"instance"`` or ``"semantic"`` segmentation target format.
        """
        from hyppopipe.data.dataset.readers.yaml_segmentation_dataset import (
            concat_segmentation_split,
        )

        return concat_segmentation_split(
            self.entries,
            yaml_dir=self.yaml_dir,
            dataset_root=self.dataset_root,
            class_names=self.class_names,
            kind=kind,
            layout=self.detection_layout,
        )


class YAMLDataset:
    """Ultralytics/YOLOv5 YAML facade with lazy per-task split materialization.

    Parses ``data.yaml``, exposes ``train``, ``val``, and optional ``test`` as
    ``YAMLSplitResource`` objects, and provides ``as_split_data`` for training
    pipelines.

    Examples:
        Load splits and build a classification training set::

            ds = YAMLDataset("datasets/my_task/data.yaml")
            split_data = ds.as_split_data()
            train_set = split_data.train.as_classification_dataset()
    """

    def __init__(
        self,
        path_to_yaml: str | Path,
        *,
        absorb_folders: bool = False,
        detection_layout: str = "auto",
        strict: bool = True,
    ) -> None:
        """Load dataset YAML and prepare split resources.

        Args:
            path_to_yaml: Path to the dataset YAML file.
            absorb_folders: Allow nested top folders in classification layout.
            detection_layout: ``"auto"``, ``"nested_class"``, or ``"flat_yolo"``
                for detection/segmentation path layout.
            strict: Passed when loading classification images.
        """
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
        """Return train/val (and optional test) resources for ``Trainer``.

        Returns:
            ``TrainVal`` when the YAML has no ``test`` split, else ``TrainValTest``.
        """
        if self.test is not None:
            return TrainValTest(train=self.train, val=self.val, test=self.test)
        return TrainVal(train=self.train, val=self.val)
