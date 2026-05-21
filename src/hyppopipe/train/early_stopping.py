"""Validation-score early stopping with optional checkpoint persistence."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch.nn import Module

MonitorMode = Literal["min", "max"]


class EarlyStopping:
    """Stop training when a validation score stops improving for ``patience`` epochs."""

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        verbose: bool = False,
        save_path: str = "best_model.pth",
        *,
        save_to_disk: bool = True,
        mode: MonitorMode = "min",
    ):
        """Configure patience, improvement threshold, and checkpoint behavior.

        Args:
            patience: Epochs without sufficient improvement before stopping.
            delta: Minimum improvement over the best score to reset the counter.
            verbose: Print counter and checkpoint messages to stdout.
            save_path: File path when ``save_to_disk`` is True.
            save_to_disk: Persist best weights to disk; otherwise keep them in memory.
            mode: ``min`` for loss-like scores, ``max`` for accuracy/F1/Dice-like scores.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = save_path
        self.save_to_disk = save_to_disk
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.early_stop = False
        self._best_state: dict[str, Any] | None = None

    @property
    def best_loss(self) -> float:
        """Backward-compatible alias for :attr:`best_score` (loss-style naming)."""
        return self.best_score

    def __call__(self, model: Module, score: float) -> bool:
        """Update state after one validation epoch.

        Args:
            model: Model whose weights are checkpointed on improvement.
            score: Validation loss or monitor value for the epoch.

        Returns:
            True if training should stop.
        """
        if self._improved(score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _improved(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.delta
        return score > self.best_score + self.delta

    def save_checkpoint(self, model: Module) -> None:
        """Persist the current model as the best checkpoint."""
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
        """Restore the best weights into ``model`` from memory or disk."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
        else:
            model.load_state_dict(torch.load(self.save_path, weights_only=True))
        if self.verbose:
            src = self.save_path if self._best_state is None else "memory"
            print(f"Loaded best model weights from {src}")
