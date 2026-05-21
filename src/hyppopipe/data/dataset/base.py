"""Abstract dataset contract for hyppopipe training."""

from abc import ABC, abstractmethod
from collections.abc import Sized

from torch.utils.data import Dataset


class ImageDataset(Sized, ABC, Dataset):
    """Base class for image datasets used before task-specific adaptation."""

    __slots__ = ()

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
