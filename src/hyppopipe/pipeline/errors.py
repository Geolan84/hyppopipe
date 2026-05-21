"""Pipeline-specific exceptions."""

from hyppopipe.types import HyppopipeError


class MissingInputsError(HyppopipeError):
    """Raised when a step's declared inputs are not available in the registry."""

    def __init__(self, step_name: str, missing_inputs: set[str]):
        """Record which inputs were missing for a step.

        Args:
            step_name: Name of the step that could not run.
            missing_inputs: Registry keys that were required but absent.
        """
        self.step_name = step_name
        self.missing_inputs = missing_inputs

    def __str__(self):
        return f"Missing inputs for step {self.step_name}: {self.missing_inputs}"
