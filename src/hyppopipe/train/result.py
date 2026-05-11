from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyppopipe.pipeline.pipeline import Pipeline


@dataclass(slots=True)
class ModelRunResult:
    model_label: str
    best_val_loss: float
    epochs_ran: int
    stopped_early: bool
    """Always points to a weights file that can be loaded with ``load_state_dict``."""
    checkpoint_path: str
    model_spec: dict[str, Any]
    inference_meta: dict[str, Any]
    train_loss_last: float | None = None
    val_loss_last: float | None = None


@dataclass(slots=True)
class StepTrainResult:
    step_name: str
    runs: list[ModelRunResult] = field(default_factory=list)


@dataclass(slots=True)
class TrainResult:
    steps: dict[str, StepTrainResult] = field(default_factory=dict)

    def export_artifacts(
        self,
        root: Path | str,
        pipeline: Pipeline,
        *,
        run_index_by_step: dict[str, int] | None = None,
        class_names_by_step: dict[str, list[str]] | None = None,
    ) -> None:
        from hyppopipe.train.bundle import export_train_result

        export_train_result(
            Path(root),
            self,
            pipeline.steps,
            run_index_by_step=run_index_by_step,
            class_names_by_step=class_names_by_step,
        )
