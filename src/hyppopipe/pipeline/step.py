from collections.abc import Callable, Sequence
from typing import Any

from hyppopipe.types import NO_VALUE


class Step:
    def __init__(
        self,
        action: Callable,
        action_args: tuple[Any, ...] | None = None,
        action_kwargs: dict[str, Any] | None = None,
        *,
        inputs: Sequence[str] | None = None,
        name: str | None = None,
        description: str | None = None,
        input_prepare: Callable[[tuple[Any, ...]], tuple[Any, ...]] | None = None,
    ):
        self.action = action
        self.action_args = action_args or ()
        self.action_kwargs = action_kwargs or {}
        self.inputs = inputs
        self.input_prepare = input_prepare
        self.name = name
        self.description = description

    def __call__(self, *args, previous_result: Any = NO_VALUE, **kwargs) -> Any:
        if previous_result is NO_VALUE:
            return self.action(*args, **kwargs)
        return self.action(previous_result, *args, **kwargs)

    def __str__(self):
        return f"Step {self.name}: {self.description}"
