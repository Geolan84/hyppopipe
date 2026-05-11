from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        verbose: bool = False,
        save_path: str = "best_model.pth",
        *,
        save_to_disk: bool = True,
    ):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = save_path
        self.save_to_disk = save_to_disk
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self._best_state: dict[str, Any] | None = None

    def __call__(self, model: Module, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: Module) -> None:
        state = model.state_dict()
        self._best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
        if self.save_to_disk:
            torch.save(state, self.save_path)
        if self.verbose:
            msg = (
                f"Saved new best model weights to {self.save_path}"
                if self.save_to_disk
                else "Saved new best model weights in memory"
            )
            print(msg)

    def load_best_model(self, model: Module) -> None:
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
        else:
            model.load_state_dict(torch.load(self.save_path, weights_only=True))
        if self.verbose:
            src = self.save_path if self._best_state is None else "memory"
            print(f"Loaded best model weights from {src}")
