"""Train/validation/test split containers and random splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, overload

import torch
from torch.utils.data import Dataset, random_split

from hyppopipe.data.dataset.base import ImageDataset


@dataclass(frozen=True, slots=True)
class TrainVal:
    """Holds training and validation dataset splits."""

    train: Dataset
    val: Dataset


@dataclass(frozen=True, slots=True)
class TrainValTest:
    """Holds training, validation, and test dataset splits."""

    train: Dataset
    val: Dataset
    test: Dataset


SplitData: TypeAlias = TrainVal | TrainValTest


def _lengths_for_split(n: int, fractions: tuple[float, ...]) -> list[int]:
    """Convert normalized fractions to integer split sizes that sum to ``n``."""
    if n <= 0:
        msg = "dataset must be non-empty"
        raise ValueError(msg)
    total = sum(fractions)
    if total <= 0:
        msg = "sum of fractions must be positive"
        raise ValueError(msg)
    norm = tuple(f / total for f in fractions)
    lengths = [int(n * f) for f in norm[:-1]]
    lengths.append(n - sum(lengths))
    if any(x < 0 for x in lengths):
        msg = "fractions yield negative split sizes; check dataset size and fractions"
        raise ValueError(msg)
    return lengths


@overload
def split_random_fractions(
    dataset: ImageDataset,
    fractions: tuple[float, float],
    *,
    seed: int | None = None,
) -> TrainVal: ...


@overload
def split_random_fractions(
    dataset: ImageDataset,
    fractions: tuple[float, float, float],
    *,
    seed: int | None = None,
) -> TrainValTest: ...


def split_random_fractions(
    dataset: ImageDataset,
    fractions: tuple[float, float] | tuple[float, float, float],
    *,
    seed: int | None = None,
) -> TrainVal | TrainValTest:
    """Randomly split one dataset by fractional sizes.

    Args:
        dataset: Source dataset to partition.
        fractions: Two values ``(train, val)`` or three ``(train, val, test)`` in ``(0, 1)``.
        seed: Optional RNG seed for reproducible splits.

    Returns:
        :class:`TrainVal` or :class:`TrainValTest` wrapping :func:`torch.utils.data.random_split` subsets.

    Raises:
        ValueError: If fractions are invalid or the dataset is empty.
    """
    if not isinstance(fractions, tuple) or not (2 <= len(fractions) <= 3):
        msg = "fractions must be a tuple of 2 or 3 positive floats"
        raise ValueError(msg)
    invalid_fractions = [f for f in fractions if not 0.0 < f < 1.0]
    if invalid_fractions:
        msg = f"each fraction must be (0, 1), but got {invalid_fractions}"
        raise ValueError(msg)
    n = len(dataset)
    lengths = _lengths_for_split(n, fractions)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    parts = random_split(dataset, lengths, generator=gen)
    if len(parts) == 2:
        return TrainVal(train=parts[0], val=parts[1])
    return TrainValTest(train=parts[0], val=parts[1], test=parts[2])
