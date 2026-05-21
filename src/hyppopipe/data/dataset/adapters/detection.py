"""Adapt generic dataset splits for detection training."""

from __future__ import annotations

from typing import Any, cast

from torch.utils.data import ConcatDataset, Dataset, Subset

from hyppopipe.data.dataset.errors import DetectionDataUnsupportedError
from hyppopipe.data.dataset.readers.yaml_detection_dataset import ConcatDetectionDataset


def adapt_dataset_for_detection(ds: Any) -> Dataset[tuple[Any, dict[str, Any]]]:
    """Return a torch dataset yielding ``(image_tensor, target_dict)`` elements.

    Supports objects with ``as_detection_dataset()``, ``Subset``, and
    ``ConcatDataset`` over compatible parts.

    Args:
        ds: Dataset or wrapper to adapt.

    Returns:
        Detection-ready dataset.

    Raises:
        DetectionDataUnsupportedError: If the source cannot be converted.
    """
    if isinstance(ds, Subset):
        inner = adapt_dataset_for_detection(cast(Any, ds.dataset))
        return Subset(inner, ds.indices)

    if hasattr(ds, "as_detection_dataset"):
        fn = getattr(ds, "as_detection_dataset")
        if callable(fn):
            out = fn()
            if not isinstance(out, Dataset):
                raise DetectionDataUnsupportedError(
                    f"{type(ds).__name__}.as_detection_dataset() must return a torch Dataset, "
                    f"got {type(out).__name__}"
                )
            return cast(Dataset[tuple[Any, dict[str, Any]]], out)

    if isinstance(ds, ConcatDataset):
        parts = [adapt_dataset_for_detection(cast(Any, x)) for x in ds.datasets]
        cls_lists = [getattr(p, "classes", None) for p in parts]
        if (
            cls_lists
            and cls_lists[0] is not None
            and all(c == cls_lists[0] for c in cls_lists)
        ):
            return ConcatDetectionDataset(parts, list(cls_lists[0]))
        return ConcatDataset(parts)

    raise DetectionDataUnsupportedError(
        f"{type(ds).__name__!r} cannot be used for detection: "
        "use a YAMLDataset split (YAMLSplitResource), Subset/ConcatDataset over "
        "detection datasets with as_detection_dataset(), or build a detection "
        "Dataset explicitly."
    )
