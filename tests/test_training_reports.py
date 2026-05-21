from __future__ import annotations

import json
from pathlib import Path

import torch

from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.train.bundle import export_train_result
from hyppopipe.train.reporting import (
    HISTORY_FILENAME,
    LOSS_PLOT_FILENAME,
    export_step_reports,
    training_history_payload,
)
from hyppopipe.train.result import (
    ModelRunResult,
    RunHistory,
    StepTrainResult,
    TrainResult,
)


def _run_with_history(
    label: str,
    *,
    train_losses: list[float],
    val_losses: list[float],
    monitor: list[float | None] | None = None,
    monitor_name: str | None = "f1_macro",
    checkpoint_path: str | None = None,
) -> ModelRunResult:
    n = len(train_losses)
    history = RunHistory(
        epochs=list(range(1, n + 1)),
        train_loss=train_losses,
        val_loss=val_losses,
        monitor=monitor if monitor is not None else [None] * n,
    )
    return ModelRunResult(
        model_label=label,
        best_val_loss=min(val_losses),
        epochs_ran=n,
        stopped_early=False,
        checkpoint_path=checkpoint_path or f"{label}.pth",
        model_spec={},
        inference_meta={"task": "classification"},
        monitor_name=monitor_name,
        best_monitor_value=max(v for v in history.monitor if v is not None)
        if monitor and any(v is not None for v in monitor)
        else None,
        monitor_mode="max" if monitor_name else None,
        history=history,
    )


def test_run_history_append_epoch() -> None:
    h = RunHistory()
    h.append_epoch(1, train_loss=0.9, val_loss=0.8, monitor=0.5)
    h.append_epoch(2, train_loss=0.7, val_loss=0.6)
    assert h.epochs == [1, 2]
    assert h.train_loss == [0.9, 0.7]
    assert h.val_loss == [0.8, 0.6]
    assert h.monitor == [0.5, None]


def test_training_history_payload() -> None:
    step = StepTrainResult(
        step_name="cls",
        runs=[
            _run_with_history("a", train_losses=[1.0, 0.5], val_losses=[0.9, 0.4]),
            _run_with_history("b", train_losses=[0.8, 0.3], val_losses=[0.7, 0.2]),
        ],
    )
    payload = training_history_payload(step)
    assert payload["step_name"] == "cls"
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["history"]["epochs"] == [1, 2]


def test_export_step_reports_writes_files(tmp_path: Path) -> None:
    step = StepTrainResult(
        step_name="cls",
        runs=[
            _run_with_history(
                "m1",
                train_losses=[1.0, 0.6],
                val_losses=[0.9, 0.5],
                monitor=[0.4, 0.7],
            ),
            _run_with_history(
                "m2",
                train_losses=[0.9, 0.4],
                val_losses=[0.8, 0.3],
                monitor=[0.5, 0.8],
            ),
        ],
    )
    out = tmp_path / "reports" / "cls"
    export_step_reports(step, out)

    history_path = out / HISTORY_FILENAME
    assert history_path.is_file()
    data = json.loads(history_path.read_text(encoding="utf-8"))
    assert data["runs"][0]["model_label"] == "m1"

    assert (out / LOSS_PLOT_FILENAME).is_file()
    assert (out / "monitor_curves.png").is_file()


def test_export_train_result_with_reports(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pth"
    torch.save({"w": torch.zeros(1)}, ckpt)
    run = _run_with_history(
        "tiny",
        train_losses=[1.0],
        val_losses=[0.5],
        monitor=[0.6],
        checkpoint_path=str(ckpt),
    )
    tr = TrainResult(steps={"only": StepTrainResult(step_name="only", runs=[run])})
    pipe_steps = {"only": Step(ImageClassifier(num_classes=2), inputs={"__input__"})}
    bundle_root = tmp_path / "bundle"
    export_train_result(bundle_root, tr, pipe_steps, reports=True)

    reports_dir = bundle_root / "reports" / "only"
    assert (reports_dir / HISTORY_FILENAME).is_file()
    assert (reports_dir / LOSS_PLOT_FILENAME).is_file()
    assert (bundle_root / "manifest.json").is_file()


def test_export_artifacts_reports_flag(tmp_path: Path) -> None:
    ckpt = tmp_path / "x.pth"
    torch.save({"w": torch.zeros(1)}, ckpt)
    run = _run_with_history(
        "x",
        train_losses=[1.0],
        val_losses=[0.5],
        checkpoint_path=str(ckpt),
    )
    tr = TrainResult(steps={"s": StepTrainResult(step_name="s", runs=[run])})
    pipeline = Pipeline({"s": Step(ImageClassifier(num_classes=2))})
    bundle_root = tmp_path / "no_reports"
    tr.export_artifacts(bundle_root, pipeline, reports=False)
    assert not (bundle_root / "reports").exists()
