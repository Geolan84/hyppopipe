"""Adapt generic dataset splits for classification training."""

from __future__ import annotations

from typing import Any, cast

from torch.utils.data import ConcatDataset, Dataset, Subset

from hyppopipe.data.dataset.errors import ClassificationDataUnsupportedError
from hyppopipe.data.dataset.protocols import ClassificationConvertible
from hyppopipe.data.dataset.readers.yaml_dataset import ConcatClassificationDataset


def adapt_dataset_for_classification(
    ds: Subset | ClassificationConvertible | ConcatDataset | Dataset,
) -> Dataset[tuple[Any, int]]:
    """Return a torch dataset yielding ``(sample, int_label)`` elements.

    Supports objects with ``as_classification_dataset()``, ``ImageFolderDataset``,
    ``Subset``, and ``ConcatDataset`` over compatible parts.

    Example:
        Wrap a folder dataset before training::

            train_ds = adapt_dataset_for_classification(folder.split.train)

    Args:
        ds: Dataset, subset, concat, or :class:`~hyppopipe.data.dataset.protocols.ClassificationConvertible`.

    Returns:
        Classification-ready dataset.

    Raises:
        ClassificationDataUnsupportedError: If the source cannot be converted.
    """
    if isinstance(ds, Subset):
        inner = adapt_dataset_for_classification(cast(Any, ds.dataset))
        return Subset(inner, ds.indices)

    if isinstance(ds, ClassificationConvertible):
        out = ds.as_classification_dataset()
        if not isinstance(out, Dataset):
            raise ClassificationDataUnsupportedError(
                f"{type(ds).__name__}.as_classification_dataset() must return a torch Dataset, "
                f"got {type(out).__name__}"
            )
        return cast(Dataset[tuple[Any, int]], out)

    if isinstance(ds, ConcatDataset):
        parts = [adapt_dataset_for_classification(x) for x in ds.datasets]
        cls_lists = [getattr(p, "classes", None) for p in parts]
        if (
            cls_lists
            and cls_lists[0] is not None
            and all(c == cls_lists[0] for c in cls_lists)
        ):
            return ConcatClassificationDataset(parts, list(cls_lists[0]))
        return ConcatDataset(parts)

    raise ClassificationDataUnsupportedError(
        f"{type(ds).__name__!r} cannot be used for classification: "
        "provide a dataset with as_classification_dataset(), ImageFolderDataset, "
        "Subset/ConcatDataset over such sources, or a YAMLDataset split "
        "(YAMLSplitResource)."
    )
