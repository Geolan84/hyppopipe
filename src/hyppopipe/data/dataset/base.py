from abc import ABC, abstractmethod
from collections.abc import Sized

from torch.utils.data import Dataset


class ImageDataset(Sized, ABC, Dataset):
    __slots__ = ()

    @abstractmethod
    def __len__(self) -> int: ...
