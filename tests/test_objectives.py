from __future__ import annotations

import torch
from torch import nn

from hyppopipe.train.config import TrainingConfig
from hyppopipe.train.early_stopping import EarlyStopping
from hyppopipe.train.objectives import (
    ClassificationObjectives,
    MonitorSpec,
    _ClassificationAccuracyMetric,
    _ClassificationF1Metric,
    resolve_loss,
)
from hyppopipe.train.result import ModelRunResult, StepTrainResult
from hyppopipe.train.bundle import _pick_run  # noqa: PLC2701


def test_classification_accuracy_metric() -> None:
    metric = _ClassificationAccuracyMetric()
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1])
    metric.update(logits, targets)
    assert metric.compute() == 1.0


def test_classification_f1_perfect() -> None:
    metric = _ClassificationF1Metric()
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    targets = torch.tensor([0, 1, 0])
    metric.update(logits, targets)
    assert metric.compute() == 1.0


def test_resolve_loss_module_and_factory() -> None:
    device = torch.device("cpu")
    config = TrainingConfig()

    def factory(d: torch.device, c: TrainingConfig) -> nn.Module:
        del c
        return nn.CrossEntropyLoss(label_smoothing=0.1).to(d)

    module = nn.CrossEntropyLoss()
    assert isinstance(
        resolve_loss(module, device, config, default=_noop), nn.CrossEntropyLoss
    )
    built = resolve_loss(factory, device, config, default=_noop)
    assert isinstance(built, nn.CrossEntropyLoss)


def _noop(device: torch.device, config: TrainingConfig) -> nn.Module:
    return nn.MSELoss().to(device)


def test_early_stopping_max_mode() -> None:
    model = nn.Linear(2, 2)
    early = EarlyStopping(patience=2, mode="max", save_to_disk=False)
    assert not early(model, 0.5)
    assert not early(model, 0.7)
    assert not early(model, 0.6)
    assert early(model, 0.65)
    assert early.early_stop
    assert early.best_score == 0.7


def test_pick_run_prefers_monitor_max() -> None:
    runs = [
        ModelRunResult(
            model_label="a",
            best_val_loss=0.5,
            epochs_ran=1,
            stopped_early=False,
            checkpoint_path="a.pth",
            model_spec={},
            inference_meta={},
            monitor_name="f1_macro",
            best_monitor_value=0.8,
            monitor_mode="max",
        ),
        ModelRunResult(
            model_label="b",
            best_val_loss=0.1,
            epochs_ran=1,
            stopped_early=False,
            checkpoint_path="b.pth",
            model_spec={},
            inference_meta={},
            monitor_name="f1_macro",
            best_monitor_value=0.9,
            monitor_mode="max",
        ),
    ]
    picked = _pick_run(StepTrainResult(step_name="s", runs=runs), None)
    assert picked.model_label == "b"


def test_monitor_spec_custom() -> None:
    spec = MonitorSpec.custom(
        "half",
        lambda: _ClassificationAccuracyMetric(),
        mode="max",
    )
    assert spec.name == "half"
    assert spec.mode == "max"
    m = spec.factory()
    assert isinstance(m, _ClassificationAccuracyMetric)


def test_classification_objectives_registry() -> None:
    loss = ClassificationObjectives.loss(label_smoothing=0.05)
    mod = loss(torch.device("cpu"), TrainingConfig())
    assert isinstance(mod, nn.CrossEntropyLoss)
    mon = ClassificationObjectives.monitor("f1_macro")
    assert mon.name == "f1_macro"
    assert mon.mode == "max"
