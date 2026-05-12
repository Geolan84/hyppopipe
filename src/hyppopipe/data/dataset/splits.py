from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, overload

import torch
from torch.utils.data import Dataset, random_split

from hyppopipe.data.dataset.base import ImageDataset


@dataclass(frozen=True, slots=True)
class TrainVal:
    train: Dataset
    val: Dataset


@dataclass(frozen=True, slots=True)
class TrainValTest:
    train: Dataset
    val: Dataset
    test: Dataset


SplitData: TypeAlias = TrainVal | TrainValTest


def _lengths_for_split(n: int, fractions: tuple[float, ...]) -> list[int]:
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
    """Случайно делит один датасет по долям (train, val) или (train, val, test)."""
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
