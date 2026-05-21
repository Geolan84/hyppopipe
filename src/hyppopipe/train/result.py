"""Training outcome types and artifact export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyppopipe.pipeline.pipeline import Pipeline


@dataclass(slots=True)
class RunHistory:
    """Per-epoch training curves for one model run."""

    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    monitor: list[float | None] = field(default_factory=list)

    def append_epoch(
        self,
        epoch: int,
        *,
        train_loss: float,
        val_loss: float,
        monitor: float | None = None,
    ) -> None:
        """Record metrics for one completed epoch."""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.monitor.append(monitor)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "monitor": self.monitor,
        }


@dataclass(slots=True)
class ModelRunResult:
    """Metrics and paths for one trained model variant."""

    model_label: str
    best_val_loss: float
    epochs_ran: int
    stopped_early: bool
    checkpoint_path: str
    """Path to weights loadable with :meth:`torch.nn.Module.load_state_dict`."""
    model_spec: dict[str, Any]
    inference_meta: dict[str, Any]
    train_loss_last: float | None = None
    val_loss_last: float | None = None
    monitor_name: str | None = None
    best_monitor_value: float | None = None
    monitor_mode: str | None = None
    """``min`` or ``max`` when a validation monitor drove checkpointing."""
    history: RunHistory | None = None


@dataclass(slots=True)
class StepTrainResult:
    """All model runs completed for one pipeline step."""

    step_name: str
    runs: list[ModelRunResult] = field(default_factory=list)


@dataclass(slots=True)
class TrainResult:
    """Aggregated training output for multiple pipeline steps."""

    steps: dict[str, StepTrainResult] = field(default_factory=dict)

    def export_artifacts(
        self,
        root: Path | str,
        pipeline: Pipeline,
        *,
        run_index_by_step: dict[str, int] | None = None,
        class_names_by_step: dict[str, list[str]] | None = None,
        reports: bool = True,
    ) -> None:
        """Write a :class:`~hyppopipe.train.bundle.PredictBundle` directory under ``root``.

        Args:
            root: Export directory (created if needed).
            pipeline: Pipeline whose step actions define task kinds.
            run_index_by_step: Per-step index into ``runs``; best val loss when omitted.
            class_names_by_step: Optional class name lists stored in the manifest.
            reports: When True, write loss/monitor plots and ``training_history.json``
                under ``reports/<step_name>/``.
        """
        from hyppopipe.train.bundle import export_train_result

        export_train_result(
            Path(root),
            self,
            pipeline.steps,
            run_index_by_step=run_index_by_step,
            class_names_by_step=class_names_by_step,
            reports=reports,
        )
