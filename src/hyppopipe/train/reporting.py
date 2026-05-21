"""Training curve plots and raw history export for artifact bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyppopipe.train.result import ModelRunResult, StepTrainResult

REPORTS_SUBDIR = "reports"
HISTORY_FILENAME = "training_history.json"
LOSS_PLOT_FILENAME = "loss_curves.png"


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def _runs_with_history(runs: list[ModelRunResult]) -> list[ModelRunResult]:
    return [r for r in runs if r.history is not None and r.history.epochs]


def training_history_payload(step_result: StepTrainResult) -> dict[str, Any]:
    """Build a JSON-serializable snapshot of all runs on one step."""
    runs_out: list[dict[str, Any]] = []
    for run in step_result.runs:
        entry: dict[str, Any] = {
            "model_label": run.model_label,
            "epochs_ran": run.epochs_ran,
            "stopped_early": run.stopped_early,
            "best_val_loss": run.best_val_loss,
            "monitor_name": run.monitor_name,
            "best_monitor_value": run.best_monitor_value,
            "monitor_mode": run.monitor_mode,
        }
        if run.history is not None:
            entry["history"] = run.history.to_dict()
        runs_out.append(entry)
    return {"step_name": step_result.step_name, "runs": runs_out}


def write_training_history_json(
    path: Path,
    step_result: StepTrainResult,
) -> None:
    """Write ``training_history.json`` for one pipeline step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(training_history_payload(step_result), indent=2),
        encoding="utf-8",
    )


def plot_loss_curves(
    runs: list[ModelRunResult],
    path: Path,
    *,
    step_name: str | None = None,
) -> bool:
    """Plot train/val loss vs epoch for every run that has history.

    Returns:
        True if a figure was written, False when there is nothing to plot.
    """
    plotted = _runs_with_history(runs)
    if not plotted:
        return False

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    title = f"Loss — {step_name}" if step_name else "Loss"
    fig.suptitle(title)

    for run in plotted:
        assert run.history is not None
        h = run.history
        ax_train.plot(h.epochs, h.train_loss, marker="o", label=run.model_label)
        ax_val.plot(h.epochs, h.val_loss, marker="o", label=run.model_label)

    ax_train.set_xlabel("epoch")
    ax_train.set_ylabel("train loss")
    ax_train.legend(loc="best", fontsize="small")
    ax_train.grid(True, alpha=0.3)

    ax_val.set_xlabel("epoch")
    ax_val.set_ylabel("val loss")
    ax_val.legend(loc="best", fontsize="small")
    ax_val.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_monitor_curves(
    runs: list[ModelRunResult],
    path: Path,
    *,
    step_name: str | None = None,
) -> bool:
    """Plot validation monitor vs epoch (one figure per monitor name).

    Returns:
        True if a figure was written, False when no monitor history exists.
    """
    by_monitor: dict[str, list[ModelRunResult]] = {}
    for run in _runs_with_history(runs):
        if run.monitor_name is None or run.history is None:
            continue
        values = [v for v in run.history.monitor if v is not None]
        if not values:
            continue
        by_monitor.setdefault(run.monitor_name, []).append(run)

    if not by_monitor:
        return False

    if len(by_monitor) == 1:
        monitor_name, monitor_runs = next(iter(by_monitor.items()))
        _save_monitor_figure(
            monitor_runs,
            monitor_name,
            path,
            step_name=step_name,
        )
        return True

    stem = path.stem
    suffix = path.suffix or ".png"
    parent = path.parent
    wrote = False
    for monitor_name, monitor_runs in by_monitor.items():
        out = parent / f"{stem}_{_safe_filename(monitor_name)}{suffix}"
        _save_monitor_figure(
            monitor_runs,
            monitor_name,
            out,
            step_name=step_name,
        )
        wrote = True
    return wrote


def _save_monitor_figure(
    runs: list[ModelRunResult],
    monitor_name: str,
    path: Path,
    *,
    step_name: str | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    title = f"{monitor_name} — {step_name}" if step_name else monitor_name
    ax.set_title(title)

    for run in runs:
        assert run.history is not None
        h = run.history
        epochs: list[int] = []
        values: list[float] = []
        for ep, val in zip(h.epochs, h.monitor, strict=True):
            if val is None:
                continue
            epochs.append(ep)
            values.append(val)
        if epochs:
            ax.plot(epochs, values, marker="o", label=run.model_label)

    ax.set_xlabel("epoch")
    ax.set_ylabel(monitor_name)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def export_step_reports(step_result: StepTrainResult, reports_dir: Path) -> None:
    """Write history JSON and loss/monitor PNGs for one step under ``reports_dir``."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_training_history_json(reports_dir / HISTORY_FILENAME, step_result)
    plot_loss_curves(
        step_result.runs,
        reports_dir / LOSS_PLOT_FILENAME,
        step_name=step_result.step_name,
    )
    monitor_path = reports_dir / "monitor_curves.png"
    plot_monitor_curves(
        step_result.runs,
        monitor_path,
        step_name=step_result.step_name,
    )


def export_training_reports(
    bundle_root: Path,
    train_result: dict[str, StepTrainResult],
) -> None:
    """Write per-step report directories under ``bundle_root/reports/``."""
    for step_name, step_tr in train_result.items():
        export_step_reports(
            step_tr,
            bundle_root / REPORTS_SUBDIR / step_name,
        )
