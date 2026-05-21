"""Orchestrates model training for a single pipeline step."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torchvision.models import WeightsEnum
from tqdm import tqdm

from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.package_logging import LogConfig, run_logging
from hyppopipe.pipeline.step import Step
from hyppopipe.train.config import TrainingConfig, apply_seed, resolve_device
from hyppopipe.train.early_stopping import EarlyStopping
from hyppopipe.train.model_spec import model_spec_from_module
from hyppopipe.train.objectives import (
    EpochMetric,
    LossFactory,
    MonitorSpec,
    resolve_loss,
)
from hyppopipe.train.result import ModelRunResult, RunHistory, StepTrainResult
from hyppopipe.train.tasks import (
    TrainingTask,
    dispatch_training_task,
    model_label_for_module,
)

logger = logging.getLogger(__name__)


class ModelCandidate:
    """Torchvision model factory evaluated with one or more pretrained weight enums."""

    __slots__ = ("model", "weights")

    def __init__(
        self,
        model: Callable[..., Module],
        weights: WeightsEnum | Sequence[WeightsEnum],
    ):
        """Register a factory and its weight variants to train sequentially.

        Args:
            model: Callable such as ``torchvision.models.resnet50`` accepting ``weights=``.
            weights: Single enum or sequence of enums passed to ``model``.
        """
        self.model = model
        if isinstance(weights, list | tuple):
            self.weights = list(weights)
        else:
            self.weights = [weights]

    def iter_models(self) -> Iterator[tuple[str, Module, WeightsEnum]]:
        """Yield ``(label, model, weights_enum)`` for each weight variant."""
        for w in self.weights:
            m = self.model(weights=w)
            wname = getattr(w, "name", None) or str(w).split(".")[-1]
            label = f"{self.model.__name__}_{wname}"
            yield label, m, w


def model_spec_from_candidate_weights(
    candidate: ModelCandidate, *, weights_member: WeightsEnum
) -> dict[str, Any]:
    """Build a serializable spec for a torchvision factory + weights pair.

    Args:
        candidate: Source factory wrapper.
        weights_member: The weights enum member used for this run.

    Returns:
        Dict with ``kind``, ``factory``, and ``weights_enum`` keys.
    """
    factory = candidate.model
    weights_fqn = (
        f"{weights_member.__class__.__module__}."
        f"{weights_member.__class__.__qualname__}.{weights_member.name}"
    )
    return {
        "kind": "torchvision_factory",
        "factory": f"{factory.__module__}.{factory.__name__}",
        "weights_enum": weights_fqn,
    }


class Trainer:
    """Trains one or more models for a pipeline step action.

    Example:
        Train a classification step on a split dataset::

            trainer = Trainer([resnet50], data=splits, config=TrainingConfig(epochs=5))
            step_result = trainer.train(step=classify_step, step_name="classify")
    """

    def __init__(
        self,
        model_candidates: Sequence[Module | ModelCandidate],
        data: SplitData | None = None,
        *,
        config: TrainingConfig | None = None,
        transforms: Any | None = None,
        loss: LossFactory | None = None,
        monitor: MonitorSpec | None = None,
        ignore_fails: bool = False,
    ):
        """Configure models and defaults for :meth:`train`.

        Args:
            model_candidates: Ready modules and/or :class:`ModelCandidate` factories.
            data: Optional default :class:`~hyppopipe.data.dataset.splits.SplitData`.
            config: Training hyperparameters; defaults to a new :class:`TrainingConfig`.
            transforms: Task-specific train/val data transforms (see ``hyppopipe.train.transforms``).
            loss: Loss override; falls back to ``config.loss``, then the task default.
            monitor: Validation monitor for early stopping; falls back to ``config.monitor``,
                then validation loss (minimize).
            ignore_fails: If True, log and skip failed candidates instead of raising.
        """
        self.model_candidates = model_candidates
        self.data = data
        self.config = config if config is not None else TrainingConfig()
        self.transforms = transforms
        self.loss = loss
        self.monitor = monitor
        self.ignore_fails = ignore_fails

    def train(
        self,
        *,
        step: Step,
        step_name: str,
        config: TrainingConfig | None = None,
        log_to: Path | str | LogConfig | None = None,
    ) -> StepTrainResult:
        """Train all model candidates for one pipeline step.

        Args:
            step: Step whose ``action`` selects the :class:`~hyppopipe.train.tasks.base.TrainingTask`.
            step_name: Key used in logs and result artifacts.
            config: Optional override of ``self.config`` for this call.
            log_to: Per-run logging configuration.

        Returns:
            One :class:`~hyppopipe.train.result.ModelRunResult` per successful candidate.

        Raises:
            ValueError: If no training data is available.
        """
        if self.data is None:
            raise ValueError("Training is impossible without splitted data")

        if config is not None:
            self.config = config

        with run_logging(log_to):
            return self._train_step(step=step, step_name=step_name)

    def _train_step(self, *, step: Step, step_name: str) -> StepTrainResult:
        """Run training for every candidate on ``step.action``."""
        apply_seed(self.config.seed)
        task = dispatch_training_task(step.action)

        n_train, n_val = task.split_lengths(self.data)
        logger.info(
            "Step %r: starting (%s); train=%d val=%d epochs=%d batch_size=%d val_batch_size=%d device=%s",
            step_name,
            step.action.__class__.__name__,
            n_train,
            n_val,
            self.config.epochs,
            self.config.batch_size,
            self.config.resolve_val_batch_size(),
            resolve_device(self.config.device),
        )

        out = StepTrainResult(step_name=step_name)

        for model_candidate in self.model_candidates:
            try:
                if isinstance(model_candidate, Module):
                    run = self._train_model(
                        model_candidate,
                        task=task,
                        step_name=step_name,
                        model_label=model_label_for_module(model_candidate),
                        model_spec=model_spec_from_module(model_candidate),
                    )
                    out.runs.append(run)
                else:
                    for label, sub_model, w_enum in model_candidate.iter_models():
                        run = self._train_model(
                            sub_model,
                            task=task,
                            step_name=step_name,
                            model_label=label,
                            model_spec=model_spec_from_candidate_weights(
                                model_candidate,
                                weights_member=w_enum,
                            ),
                            weights_enum=w_enum,
                        )
                        out.runs.append(run)
            except Exception:
                if self.ignore_fails:
                    logger.exception(
                        "Step %r: training failed for a model candidate; ignoring (ignore_fails=True)",
                        step_name,
                    )
                    continue
                raise

        logger.info(
            "Step %r: finished %d model run(s)",
            step_name,
            len(out.runs),
        )
        return out

    def _train_model(
        self,
        model: Module,
        *,
        task: TrainingTask,
        step_name: str,
        model_label: str,
        model_spec: dict[str, Any],
        weights_enum: WeightsEnum | None = None,
    ) -> ModelRunResult:
        """Train a single model through epochs with optional early stopping."""
        assert self.data is not None
        device = resolve_device(self.config.device)
        run_t0 = time.perf_counter()

        prepared, train_loader, val_loader = task.prepare(
            model,
            self.data,
            self.config,
            weights_enum=weights_enum,
            transforms=self.transforms,
        )
        inference_meta = task.inference_meta_from_prepared(prepared)
        prepared = prepared.to(device)
        loss_spec = self.loss if self.loss is not None else self.config.loss
        criterion = resolve_loss(
            loss_spec,
            device,
            self.config,
            default=task.create_criterion,
        )
        monitor_spec = self.monitor if self.monitor is not None else self.config.monitor
        val_metric: EpochMetric | None = (
            monitor_spec.factory() if monitor_spec is not None else None
        )
        checkpoint_mode = monitor_spec.mode if monitor_spec is not None else "min"
        optimizer = self.config.build_optimizer(prepared.parameters())

        es_cfg = self.config.early_stopping
        early: EarlyStopping | None = None
        if es_cfg is not None and es_cfg.enabled:
            ckpt_path = es_cfg.save_path
            if ckpt_path is None:
                safe_step = "".join(
                    c if c.isalnum() or c in "-._" else "_" for c in step_name
                )
                safe_model = "".join(
                    c if c.isalnum() or c in "-._" else "_" for c in model_label
                )
                ckpt_path = f"{safe_step}_{safe_model}_best.pth"
            early = EarlyStopping(
                patience=es_cfg.patience,
                delta=es_cfg.delta,
                verbose=es_cfg.verbose,
                save_path=ckpt_path,
                save_to_disk=es_cfg.save_to_disk,
                mode=checkpoint_mode,
            )

        best_val = float("inf")
        best_monitor: float | None = None
        train_last = 0.0
        val_last = 0.0
        monitor_last: float | None = None
        epochs_ran = 0
        stopped_early = False
        history = RunHistory()

        logger.info(
            "Step %r model %r: training on %s (%d epochs, early_stopping=%s, monitor=%s)",
            step_name,
            model_label,
            device,
            self.config.epochs,
            "on" if early is not None else "off",
            monitor_spec.name if monitor_spec is not None else "val_loss",
        )

        epoch_pbar = tqdm(
            range(self.config.epochs),
            desc=f"{step_name} · {model_label}",
            unit="epoch",
            leave=True,
        )
        for _epoch in epoch_pbar:
            epochs_ran = _epoch + 1
            epoch_t0 = time.perf_counter()
            train_last = self._run_epoch_train(
                prepared,
                train_loader,
                task,
                criterion,
                optimizer,
                device,
                epoch=epochs_ran,
                total_epochs=self.config.epochs,
            )
            val_last, monitor_last = self._run_epoch_eval(
                prepared,
                val_loader,
                task,
                criterion,
                device,
                epoch=epochs_ran,
                total_epochs=self.config.epochs,
                val_metric=val_metric,
            )
            best_val = min(best_val, val_last)
            checkpoint_score = (
                monitor_last
                if val_metric is not None and monitor_last is not None
                else val_last
            )
            if monitor_last is not None:
                if checkpoint_mode == "max":
                    best_monitor = (
                        monitor_last
                        if best_monitor is None
                        else max(best_monitor, monitor_last)
                    )
                else:
                    best_monitor = (
                        monitor_last
                        if best_monitor is None
                        else min(best_monitor, monitor_last)
                    )
            history.append_epoch(
                epochs_ran,
                train_loss=train_last,
                val_loss=val_last,
                monitor=monitor_last,
            )
            epoch_dt = time.perf_counter() - epoch_t0
            postfix: dict[str, str] = {
                "train": f"{train_last:.4f}",
                "val": f"{val_last:.4f}",
                "t": f"{epoch_dt:.1f}s",
            }
            if monitor_last is not None and monitor_spec is not None:
                postfix[monitor_spec.name] = f"{monitor_last:.4f}"
            epoch_pbar.set_postfix(**postfix)

            stop_now = False
            if early is not None:
                stop_now = early(prepared, checkpoint_score)
                best_display = early.best_score
            else:
                best_display = checkpoint_score

            log_monitor = (
                f" {monitor_spec.name}={monitor_last:.6f}"
                if monitor_spec is not None and monitor_last is not None
                else ""
            )
            logger.info(
                "Step %r %s: epoch %d/%d train_loss=%.6f val_loss=%.6f%s best=%.6f (epoch %.2fs)",
                step_name,
                model_label,
                epochs_ran,
                self.config.epochs,
                train_last,
                val_last,
                log_monitor,
                best_display,
                epoch_dt,
            )

            if early is not None and stop_now:
                stopped_early = True
                metric_label = (
                    monitor_spec.name if monitor_spec is not None else "val_loss"
                )
                logger.info(
                    "Step %r %s: early stopping after %d epochs (best %s=%.6f)",
                    step_name,
                    model_label,
                    epochs_ran,
                    metric_label,
                    early.best_score,
                )
                early.load_best_model(prepared)
                break
        else:
            if early is not None:
                early.load_best_model(prepared)

        safe_step = "".join(c if c.isalnum() or c in "-._" else "_" for c in step_name)
        safe_model = "".join(
            c if c.isalnum() or c in "-._" else "_" for c in model_label
        )
        final_name = f"{safe_step}_{safe_model}_final.pth"
        if early is not None and es_cfg is not None and es_cfg.save_to_disk:
            persistent_path = str(Path(early.save_path).resolve())
            persistent_msg = persistent_path
        else:
            cpu_state = {k: v.detach().cpu() for k, v in prepared.state_dict().items()}
            torch.save(cpu_state, final_name)
            persistent_path = str(Path(final_name).resolve())
            persistent_msg = persistent_path

        total_dt = time.perf_counter() - run_t0
        if early is not None:
            reported_checkpoint = early.best_score
        elif val_metric is not None and best_monitor is not None:
            reported_checkpoint = best_monitor
        else:
            reported_checkpoint = best_val
        logger.info(
            "Step %r %s: done in %.1fs — epochs=%d/%s checkpoint_score=%.6f "
            "train_loss=%.6f val_loss=%.6f checkpoint=%s",
            step_name,
            model_label,
            total_dt,
            epochs_ran,
            "early" if stopped_early else str(self.config.epochs),
            reported_checkpoint,
            train_last,
            val_last,
            persistent_msg,
        )

        return ModelRunResult(
            model_label=model_label,
            best_val_loss=best_val,
            epochs_ran=epochs_ran,
            stopped_early=stopped_early,
            checkpoint_path=persistent_path,
            model_spec=model_spec,
            inference_meta=inference_meta,
            train_loss_last=train_last,
            val_loss_last=val_last,
            monitor_name=monitor_spec.name if monitor_spec is not None else None,
            best_monitor_value=(
                early.best_score
                if early is not None and monitor_spec is not None
                else best_monitor
            ),
            monitor_mode=checkpoint_mode if monitor_spec is not None else None,
            history=history,
        )

    def _run_epoch_train(
        self,
        model: Module,
        loader: torch.utils.data.DataLoader[Any],
        task: TrainingTask,
        criterion: Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Run one training epoch and return mean loss per sample."""
        model.train()
        total = 0.0
        n = 0
        batch_pbar = tqdm(
            loader,
            desc=f"Train {epoch}/{total_epochs}",
            unit="batch",
            leave=False,
        )
        for batch in batch_pbar:
            loss_sum, bs = task.train_batch(model, batch, criterion, optimizer, device)
            total += loss_sum
            n += bs
            batch_pbar.set_postfix(loss=f"{total / max(n, 1):.4f}")
        return total / max(n, 1)

    def _run_epoch_eval(
        self,
        model: Module,
        loader: torch.utils.data.DataLoader[Any],
        task: TrainingTask,
        criterion: Module,
        device: torch.device,
        *,
        epoch: int,
        total_epochs: int,
        val_metric: EpochMetric | None = None,
    ) -> tuple[float, float | None]:
        """Run one validation epoch; return mean loss and optional monitor value."""
        model.eval()
        total = 0.0
        n = 0
        if val_metric is not None:
            val_metric.reset()
        batch_pbar = tqdm(
            loader,
            desc=f"Val {epoch}/{total_epochs}",
            unit="batch",
            leave=False,
        )
        with torch.no_grad():
            for batch in batch_pbar:
                loss_sum, bs = task.eval_batch(model, batch, criterion, device)
                total += loss_sum
                n += bs
                if val_metric is not None:
                    task.update_monitor(val_metric, model, batch, device)
                batch_pbar.set_postfix(loss=f"{total / max(n, 1):.4f}")
        val_loss = total / max(n, 1)
        monitor_value = val_metric.compute() if val_metric is not None else None
        return val_loss, monitor_value
