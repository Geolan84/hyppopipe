from hyppopipe.types import HyppopipeError


class MissingInputsError(HyppopipeError):
    def __init__(self, step_name: str, missing_inputs: set[str]):
        self.step_name = step_name
        self.missing_inputs = missing_inputs

    def __str__(self):
        return f"Missing inputs for step {self.step_name}: {self.missing_inputs}"
