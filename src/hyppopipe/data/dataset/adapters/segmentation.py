"""Преобразование сплита в ``Dataset`` для задачи сегментации."""

from __future__ import annotations

from inspect import signature
from typing import Any, Literal, cast

from torch.utils.data import ConcatDataset, Dataset, Subset

from hyppopipe.data.dataset.errors import SegmentationDataUnsupportedError
from hyppopipe.data.dataset.readers.yaml_segmentation_dataset import (
    ConcatSegmentationDataset,
)

SegmentationKind = Literal["instance", "semantic"]


def _call_as_segmentation_dataset(
    ds: Any, kind: SegmentationKind
) -> Dataset[Any] | None:
    if not hasattr(ds, "as_segmentation_dataset"):
        return None

    fn = getattr(ds, "as_segmentation_dataset")
    if not callable(fn):
        return None

    params = signature(fn).parameters
    if "kind" in params:
        out = fn(kind=kind)
    else:
        out = fn()

    if not isinstance(out, Dataset):
        raise SegmentationDataUnsupportedError(
            f"{type(ds).__name__}.as_segmentation_dataset() must return a torch Dataset, "
            f"got {type(out).__name__}"
        )
    return cast(Dataset[Any], out)


def adapt_dataset_for_segmentation(
    ds: Any,
    *,
    kind: SegmentationKind,
) -> Dataset[Any]:
    """
    Возвращает torch-датасет для сегментации.

    ``kind="instance"`` ожидает элементы ``(image, target_dict)`` с
    ``boxes``, ``labels`` и ``masks``. ``kind="semantic"`` ожидает
    ``(image, class_map)``.
    """
    if isinstance(ds, Subset):
        inner = adapt_dataset_for_segmentation(cast(Any, ds.dataset), kind=kind)
        return Subset(inner, ds.indices)

    converted = _call_as_segmentation_dataset(ds, kind)
    if converted is not None:
        return converted

    if isinstance(ds, ConcatDataset):
        parts = [
            adapt_dataset_for_segmentation(cast(Any, x), kind=kind) for x in ds.datasets
        ]
        cls_lists = [getattr(p, "classes", None) for p in parts]
        if (
            cls_lists
            and cls_lists[0] is not None
            and all(c == cls_lists[0] for c in cls_lists)
        ):
            return ConcatSegmentationDataset(parts, list(cls_lists[0]), kind=kind)
        return ConcatDataset(parts)

    raise SegmentationDataUnsupportedError(
        f"{type(ds).__name__!r} cannot be used for {kind} segmentation: "
        "provide a dataset with as_segmentation_dataset(), a YAMLDataset split, "
        "PairedImageMaskFolderDataset for semantic segmentation, or "
        "Subset/ConcatDataset over compatible sources."
    )
