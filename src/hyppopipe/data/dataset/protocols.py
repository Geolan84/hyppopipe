"""Structural typing protocols for dataset adapters."""

from typing import Protocol, runtime_checkable

from hyppopipe.data.dataset import ImageDataset


@runtime_checkable
class ClassificationConvertible(Protocol):
    """Dataset that can expose a classification-ready :class:`ImageDataset`."""

    def as_classification_dataset(self) -> ImageDataset:
        """Return a torch dataset yielding ``(sample, int_label)`` pairs."""
