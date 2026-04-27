from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, overload

from hyppopipe.data.base import DataResource
from hyppopipe.pipeline.errors import MissingInputsError
from hyppopipe.pipeline.step import Step
from hyppopipe.types import HyppopipeError


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
        self.shift_result = shift_result
        self.registry = {}
        if isinstance(steps, Pipeline):
            self.steps = deepcopy(steps.steps)
        elif isinstance(steps, Mapping):
            self.steps = steps
        elif isinstance(steps, Iterable):
            self.steps = {
                step.name or f"step_{i}": step for i, step in enumerate(steps)
            }

    def predict(
        self,
        input_data: DataResource,
        output_names: set[str] | None = None,
    ) -> dict[str, Any]:
        previous_result = input_data
        self.registry["__input__"] = input_data
        for name, step in self.steps.items():
            step_inputs = self._get_inputs(step)
            self.logger.info(f"Running step {name}")
            try:
                if self.shift_result:
                    previous_result = step(previous_result, *step_inputs)
                else:
                    previous_result = step(*step_inputs)
            except Exception as e:
                self.logger.error(f"Error in step {name}: {e}")
                raise HyppopipeError(f"Error in step {name}: {e}") from e
            else:
                self.registry[name] = previous_result
                self.logger.info(f"Step {name} completed successfully")
        return self._filter_outputs(output_names)

    def _get_inputs(self, step: Step) -> tuple[Any, ...]:
        """
        Gets inputs from registry for a step.

        Args:
            step (Step): Pipeline step.

        Raises:
            MissingInputsError: If step has inputs that are not in registry.

        Returns:
            tuple[Any, ...]: Inputs for a step.
        """
        if step.inputs is None:
            return tuple()
        missing_inputs = step.inputs - self.registry.keys()
        if missing_inputs:
            raise MissingInputsError(step.name, missing_inputs)
        return tuple(self.registry[key] for key in step.inputs)

    def _filter_outputs(self, output_names: set[str] | None = None) -> dict[str, Any]:
        """
        Filters outputs from registry by given names.
        If output_names is None, returns all outputs.

        Args:
            output_names (set[str] | None, optional): Names of outputs to filter. Defaults to None.

        Returns:
            dict[str, Any]: Filtered outputs.
        """
        if output_names is None:
            return self.registry
        return {
            key: self.registry[key]
            for key in self.registry.keys()
            if key in output_names
        }

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.__module__)
