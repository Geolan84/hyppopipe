from typing import Protocol, runtime_checkable

from hyppopipe.data.dataset import ImageDataset


@runtime_checkable
class ClassificationConvertible(Protocol):
    def as_classification_dataset(self) -> ImageDataset: ...
