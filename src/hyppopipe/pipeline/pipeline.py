from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast, overload

import torch
from torch.nn import Module

from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.data.image import Image
from hyppopipe.inference.run import run_step_inference
from hyppopipe.inference.types import PipelinePrediction
from hyppopipe.package_logging import ensure_default_logging
from hyppopipe.pipeline.errors import MissingInputsError
from hyppopipe.pipeline.step import Step
from hyppopipe.train.bundle import PredictBundle
from hyppopipe.train.config import resolve_device
from hyppopipe.train.result import StepTrainResult, TrainResult
from hyppopipe.train.trainer import Trainer

logger = logging.getLogger(__name__)


def _topological_step_order(steps: Mapping[str, Step]) -> list[str]:
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
    @overload
    def __init__(
        self, steps: Iterable[Step | Pipeline], *, shift_result: bool = True
    ): ...

    @overload
    def __init__(
        self, steps: Mapping[str, Step | Pipeline], *, shift_result: bool = True
    ): ...

    @overload
    def __init__(self, steps: Pipeline, *, shift_result: bool = True): ...

    def __init__(
        self,
        steps: Iterable[Step] | Mapping[str, Step] | Pipeline,
        *,
        shift_result: bool = True,
    ):
        """
        Args:
            shift_result: Если ``True``, при ``predict`` входы шагов не берутся из
                ``Step.inputs``, а выстраиваются в цепочку: первый шаг (в топологическом
                порядке) получает ``__input__``, каждый следующий — вывод предыдущего шага.
                Явные ``inputs`` в этом режиме игнорируются. Если ``False``, используются
                только ``Step.inputs`` (как в ``_get_inputs``).
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
        step_config: Mapping[str, Trainer],
        data: SplitData | None = None,
    ) -> TrainResult:
        missing_keys = set(step_config.keys()) - set(self.steps.keys())
        if missing_keys:
            raise KeyError(f"Keys {missing_keys} are not present in pipeline steps")

        ensure_default_logging()
        step_names = ", ".join(step_config.keys())
        logger.info("Pipeline training started for steps: %s", step_names)

        steps_out: dict[str, StepTrainResult] = {}
        for step_name, trainer in step_config.items():
            if trainer.data is None and data is None:
                raise ValueError("Training data (TrainVal / TrainValTest) is required")
            if trainer.data is None:
                trainer.data = data
            step = self.steps[step_name]
            steps_out[step_name] = trainer.train(step=step, step_name=step_name)
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
        """Runs inference for every step in topological order."""
        n_sources = sum(1 for x in (bundle, bundle_path, train_result) if x is not None)
        if n_sources != 1:
            msg = "Exactly one of bundle, bundle_path, train_result must be provided"
            raise ValueError(msg)

        resolved: PredictBundle
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

        ordered = _topological_step_order(self.steps)
        for name in ordered:
            if name not in resolved.steps:
                msg = (
                    f"No artifacts for step {name!r}; train this step or export a bundle "
                    "that includes it."
                )
                raise KeyError(msg)

        dev = resolve_device(device)
        ensure_default_logging()
        models_cache: dict[str, Module] = {}
        base_dict = dict(step_base_models) if step_base_models is not None else None

        self.registry = {}
        self.registry["__input__"] = image

        for step_index, name in enumerate(ordered):
            step = self.steps[name]
            artifact = resolved.steps[name]
            inputs = self._predict_step_inputs(ordered, step_index, name, step)
            out = run_step_inference(
                name,
                step,
                inputs,
                artifact,
                device=dev,
                score_thresh=score_thresh,
                models_cache=models_cache,
                step_base_models=base_dict,
            )
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
        """
        Gets inputs from registry for a step.

        Args:
            name (str): Name of the step.
            step (Step): Pipeline step.

        Raises:
            MissingInputsError: If step has inputs that are not in registry.

        Returns:
            tuple[Any, ...]: Inputs for a step.
        """
        if step.inputs is None:
            return tuple()
        missing_inputs = set(step.inputs) - self.registry.keys()
        if missing_inputs:
            raise MissingInputsError(name, missing_inputs)
        return tuple(self.registry[key] for key in step.inputs)

    def _filter_outputs(self, output_names: set[str] | None = None) -> dict[str, Any]:
        """
        Filters outputs from registry by given names.
        If output_names is None, returns all outputs.

        Args:
            output_names (set[str] | None, optional): Names of outputs to filter. Defaults to None.

        Returns:
            dict[str, Any]: Filtered outputs from registry.
        """
        if output_names is None:
            return self.registry
        return {
            key: self.registry[key]
            for key in self.registry.keys()
            if key in output_names
        }
