"""Declarative multi-step ML pipeline execution and training."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

import torch
from torch.nn import Module

from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.data.image import Image
from hyppopipe.inference.run import (
    run_step_inference,
    run_step_without_artifact,
    step_needs_artifact,
)
from hyppopipe.inference.types import PipelinePrediction
from hyppopipe.package_logging import LogConfig, ensure_default_logging, run_logging
from hyppopipe.pipeline.errors import MissingInputsError
from hyppopipe.pipeline.step import Step
from hyppopipe.train.bundle import PredictBundle
from hyppopipe.train.config import TrainingConfig, resolve_device
from hyppopipe.train.result import StepTrainResult, TrainResult

if TYPE_CHECKING:
    from hyppopipe.train.trainer import Trainer

logger = logging.getLogger(__name__)


def _topological_step_order(steps: Mapping[str, Step]) -> list[str]:
    """Return step names in dependency order (Kahn topological sort).

    Args:
        steps: Named pipeline steps and their ``Step.inputs`` edges.

    Returns:
        Step names such that every dependency appears before its dependents.

    Raises:
        MissingInputsError: If a step references an unknown step name.
        ValueError: If the dependency graph has a cycle.
    """
    names = list(steps.keys())
    name_set = set(names)
    adj: dict[str, list[str]] = {n: [] for n in names}
    indeg: dict[str, int] = {n: 0 for n in names}
    for v, step in steps.items():
        ins = step.inputs
        if ins is None:
            continue
        for u in ins:
            if u == "__input__":
                continue
            if u not in name_set:
                raise MissingInputsError(v, {u})
            adj[u].append(v)
            indeg[v] += 1
    q = deque([n for n in names if indeg[n] == 0])
    out: list[str] = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(out) != len(names):
        msg = "Pipeline graph has a cycle or invalid dependencies"
        raise ValueError(msg)
    return out


class Pipeline:
    """Ordered graph of :class:`~hyppopipe.pipeline.step.Step` for train and predict.

    Example:
        Build from a mapping, train selected steps, then run inference::

            pipeline = Pipeline({"detect": detect_step, "classify": classify_step})
            pipeline.train({"detect": trainer_det, "classify": trainer_cls}, data=splits)
            pred = pipeline.predict(image, train_result=result)  # or predict(image) for transform-only steps
    """

    @overload
    def __init__(self, steps: Iterable[Step], *, shift_result: bool = True): ...

    @overload
    def __init__(self, steps: Mapping[str, Step], *, shift_result: bool = True): ...

    @overload
    def __init__(self, steps: Pipeline, *, shift_result: bool = True): ...

    def __init__(
        self,
        steps: Iterable[Step] | Mapping[str, Step] | Pipeline,
        *,
        shift_result: bool = True,
    ):
        """Build a pipeline from steps in dependency order.

        Args:
            steps: Named mapping, iterable of ``Step``, or another ``Pipeline`` to copy.
            shift_result: If True, ``predict`` chains each step's output into the next
                step instead of resolving inputs from ``Step.inputs``.
        """
        self.shift_result = shift_result
        self.registry = {}
        self.steps: Mapping[str, Step]
        if isinstance(steps, Pipeline):
            self.steps = deepcopy(steps.steps)
        elif isinstance(steps, Mapping):
            self.steps = cast(Mapping[str, Step], steps)
        elif isinstance(steps, Iterable):
            self.steps = {
                step.name or f"step_{i}": step
                for i, step in enumerate(cast(Iterable[Step], steps))
            }

    def train(
        self,
        step_config: Mapping[str, "Trainer"],
        data: SplitData | None = None,
        config: TrainingConfig | None = None,
        *,
        log_to: Path | str | LogConfig | None = None,
    ) -> TrainResult:
        """Train one or more pipeline steps with the given trainers.

        Args:
            step_config: Map from step name to :class:`~hyppopipe.train.trainer.Trainer`.
            data: Default split data when a trainer has no ``data`` set.
            config: Optional override of each trainer's :class:`~hyppopipe.train.config.TrainingConfig`.
            log_to: Per-run logging target (see :func:`~hyppopipe.package_logging.run_logging`).

        Returns:
            Aggregated training results per step.

        Raises:
            KeyError: If ``step_config`` references unknown step names.
            ValueError: If training data is missing for a step.
        """
        missing_keys = set(step_config.keys()) - set(self.steps.keys())
        if missing_keys:
            raise KeyError(f"Keys {missing_keys} are not present in pipeline steps")

        with run_logging(log_to):
            step_names = ", ".join(step_config.keys())
            logger.info("Pipeline training started for steps: %s", step_names)

            steps_out: dict[str, StepTrainResult] = {}
            for step_name, trainer in step_config.items():
                if trainer.data is None and data is None:
                    raise ValueError(
                        "Training data (TrainVal / TrainValTest) is required"
                    )
                if trainer.data is None:
                    trainer.data = data
                step = self.steps[step_name]
                steps_out[step_name] = trainer.train(
                    step=step,
                    step_name=step_name,
                    config=config,
                )
                n_runs = len(steps_out[step_name].runs)
                logger.info(
                    "Pipeline step %r completed (%d model run(s))", step_name, n_runs
                )

            logger.info("Pipeline training finished (%d step(s))", len(steps_out))
            return TrainResult(steps=steps_out)

    def predict(
        self,
        image: Image,
        *,
        bundle: PredictBundle | None = None,
        bundle_path: Path | str | None = None,
        train_result: TrainResult | None = None,
        run_index_by_step: dict[str, int] | None = None,
        device: str | torch.device | None = None,
        score_thresh: float = 0.5,
        step_base_models: Mapping[str, Module] | None = None,
        return_all: bool = False,
    ) -> PipelinePrediction:
        """Run inference for every step in topological order.

        Args:
            image: Input image stored under registry key ``__input__``.
            bundle: Pre-exported :class:`~hyppopipe.train.bundle.PredictBundle`.
            bundle_path: Directory containing ``manifest.json`` and weights.
            train_result: Ephemeral bundle built from a recent :class:`~hyppopipe.train.result.TrainResult`.
            run_index_by_step: Per-step index into ``StepTrainResult.runs`` when using ``train_result``.
            device: Torch device string or instance; auto-detected if None.
            score_thresh: Minimum detection/segmentation score for ROI selection.
            step_base_models: Optional pre-built base modules keyed by step name.
            return_all: If True, include ``__input__`` in returned outputs.

        Returns:
            Predictions keyed by step name (and optionally ``__input__``).

        Raises:
            ValueError: If trained steps lack exactly one artifact source, or artifact
                sources are passed when no step needs them.
            KeyError: If a trained step has no artifacts in the bundle.
        """
        ordered = _topological_step_order(self.steps)
        trained_steps = [
            name for name in ordered if step_needs_artifact(self.steps[name].action)
        ]
        n_sources = sum(1 for x in (bundle, bundle_path, train_result) if x is not None)

        resolved: PredictBundle | None
        if trained_steps:
            if n_sources != 1:
                msg = (
                    "Exactly one of bundle, bundle_path, train_result must be provided "
                    f"for trained steps: {trained_steps!r}"
                )
                raise ValueError(msg)
            if bundle is not None:
                resolved = bundle
            elif bundle_path is not None:
                resolved = PredictBundle.load(bundle_path)
            else:
                assert train_result is not None
                resolved = PredictBundle.from_train_result(
                    train_result,
                    self.steps,
                    run_index_by_step=run_index_by_step,
                )
            for name in trained_steps:
                if name not in resolved.steps:
                    msg = (
                        f"No artifacts for step {name!r}; train this step or export a bundle "
                        "that includes it."
                    )
                    raise KeyError(msg)
        else:
            if n_sources != 0:
                msg = (
                    "This pipeline has no steps that require trained weights; omit "
                    "bundle, bundle_path, and train_result"
                )
                raise ValueError(msg)
            resolved = None

        dev = resolve_device(device)
        ensure_default_logging()
        models_cache: dict[str, Module] = {}
        base_dict = dict(step_base_models) if step_base_models is not None else None

        self.registry = {}
        self.registry["__input__"] = image

        for step_index, name in enumerate(ordered):
            step = self.steps[name]
            inputs = self._predict_step_inputs(ordered, step_index, name, step)
            if step_needs_artifact(step.action):
                assert resolved is not None
                out = run_step_inference(
                    name,
                    step,
                    inputs,
                    resolved.steps[name],
                    device=dev,
                    score_thresh=score_thresh,
                    models_cache=models_cache,
                    step_base_models=base_dict,
                )
            else:
                out = run_step_without_artifact(step, inputs)
            self.registry[name] = out

        if return_all:
            outputs = dict(self.registry)
        else:
            outputs = {k: v for k, v in self.registry.items() if k != "__input__"}
        return PipelinePrediction(outputs=outputs)

    def _predict_step_inputs(
        self,
        ordered: list[str],
        step_index: int,
        name: str,
        step: Step,
    ) -> tuple[Any, ...]:
        """Resolve inputs for one predict step (chained or explicit mode)."""
        if self.shift_result:
            if step.inputs is not None:
                logger.debug(
                    "Pipeline.shift_result=True: ignoring explicit inputs %s for step %r",
                    step.inputs,
                    name,
                )
            if step_index == 0:
                return (self.registry["__input__"],)
            prev = ordered[step_index - 1]
            return (self.registry[prev],)
        return self._get_inputs(name, step)

    def _get_inputs(self, name: str, step: Step) -> tuple[Any, ...]:
        """Collect inputs for a step from the runtime registry.

        Args:
            name: Step name (for error messages).
            step: Step whose ``inputs`` keys are looked up in ``self.registry``.

        Returns:
            Values from the registry in ``step.inputs`` order.

        Raises:
            MissingInputsError: If any required registry key is missing.
        """
        if step.inputs is None:
            return tuple()
        missing_inputs = set(step.inputs) - self.registry.keys()
        if missing_inputs:
            raise MissingInputsError(name, missing_inputs)
        return tuple(self.registry[key] for key in step.inputs)

    def _filter_outputs(self, output_names: set[str] | None = None) -> dict[str, Any]:
        """Return registry entries, optionally restricted to ``output_names``.

        Args:
            output_names: Keys to keep; if None, return the full registry.

        Returns:
            Subset of ``self.registry``.
        """
        if output_names is None:
            return self.registry
        return {
            key: self.registry[key]
            for key in self.registry.keys()
            if key in output_names
        }
