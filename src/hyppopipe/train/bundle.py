from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from collections.abc import Mapping

from hyppopipe.pipeline.step import Step
from hyppopipe.train.result import ModelRunResult, StepTrainResult, TrainResult
from hyppopipe.train.tasks.dispatch import dispatch_training_task

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
WEIGHTS_SUBDIR = "weights"


@dataclass(slots=True)
class StepArtifact:
    """One trained step inside a ``PredictBundle``."""

    task: str
    weights_path: Path
    model_spec: dict[str, Any]
    inference_meta: dict[str, Any]
    class_names: list[str] | None = None


@dataclass(slots=True)
class PredictBundle:
    """Loaded export directory or ephemeral view built from ``TrainResult``."""

    root: Path | None
    steps: dict[str, StepArtifact]

    @classmethod
    def load(cls, root: Path | str) -> PredictBundle:
        root_path = Path(root).resolve()
        manifest_path = root_path / MANIFEST_NAME
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = raw.get("version")
        if version != MANIFEST_VERSION:
            msg = (
                f"Unsupported manifest version {version!r}, expected {MANIFEST_VERSION}"
            )
            raise ValueError(msg)
        steps_out: dict[str, StepArtifact] = {}
        for name, entry in raw["steps"].items():
            wp = Path(entry["weights"])
            if not wp.is_absolute():
                wp = root_path / wp
            steps_out[name] = StepArtifact(
                task=entry["task"],
                weights_path=wp.resolve(),
                model_spec=entry["model_spec"],
                inference_meta=entry["inference_meta"],
                class_names=entry.get("class_names"),
            )
        return cls(root=root_path, steps=steps_out)

    @classmethod
    def from_train_result(
        cls,
        train_result: TrainResult,
        steps: Mapping[str, Step],
        *,
        run_index_by_step: dict[str, int] | None = None,
    ) -> PredictBundle:
        steps_out: dict[str, StepArtifact] = {}
        for step_name, step_tr in train_result.steps.items():
            run = _pick_run(
                step_tr, run_index_by_step.get(step_name) if run_index_by_step else None
            )
            if step_name not in steps:
                msg = f"Step {step_name!r} not found on pipeline"
                raise KeyError(msg)
            dispatch_training_task(steps[step_name].action)
            task_kind = run.inference_meta.get("task", "unknown")
            steps_out[step_name] = StepArtifact(
                task=str(task_kind),
                weights_path=Path(run.checkpoint_path).resolve(),
                model_spec=run.model_spec,
                inference_meta=run.inference_meta,
                class_names=None,
            )
        return cls(root=None, steps=steps_out)


def _pick_run(step_result: StepTrainResult, run_index: int | None) -> ModelRunResult:
    if not step_result.runs:
        msg = f"Step {step_result.step_name!r} has no completed runs"
        raise ValueError(msg)
    if run_index is not None:
        return step_result.runs[run_index]
    return min(step_result.runs, key=lambda r: r.best_val_loss)


def export_train_result(
    root: Path,
    train_result: TrainResult,
    steps: Mapping[str, Step],
    *,
    run_index_by_step: dict[str, int] | None = None,
    class_names_by_step: dict[str, list[str]] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    weights_dir = root / WEIGHTS_SUBDIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest_steps: dict[str, Any] = {}
    class_names_by_step = class_names_by_step or {}

    for step_name, step_tr in train_result.steps.items():
        run = _pick_run(
            step_tr, run_index_by_step.get(step_name) if run_index_by_step else None
        )
        if step_name not in steps:
            msg = f"Step {step_name!r} not found on pipeline"
            raise KeyError(msg)
        dispatch_training_task(steps[step_name].action)

        dest = weights_dir / f"{step_name}.pth"
        shutil.copy2(run.checkpoint_path, dest)
        rel_weights = f"{WEIGHTS_SUBDIR}/{dest.name}"
        task_kind = run.inference_meta.get("task", "unknown")
        manifest_steps[step_name] = {
            "task": task_kind,
            "weights": rel_weights,
            "model_spec": run.model_spec,
            "inference_meta": run.inference_meta,
            "class_names": class_names_by_step.get(step_name),
            "model_label": run.model_label,
            "best_val_loss": run.best_val_loss,
        }

    manifest = {"version": MANIFEST_VERSION, "steps": manifest_steps}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
