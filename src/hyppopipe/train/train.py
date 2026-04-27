from typing import Any, Protocol

from hyppopipe.data.dataset import Dataset


class Trainable(Protocol):
    def train(self, dataset: Dataset, train_config: dict[str, Any]) -> None: ...
