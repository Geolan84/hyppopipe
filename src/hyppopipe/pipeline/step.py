"""Single executable unit inside a :class:`~hyppopipe.pipeline.pipeline.Pipeline`."""

from collections.abc import Callable, Sequence
from typing import Any

from hyppopipe.types import NO_VALUE


class Step:
    """Wraps a callable action with optional named inputs and metadata."""

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
        """Configure a pipeline step.

        Args:
            action: Functor (e.g. ``ImageClassifier``) or any callable executed at runtime.
            action_args: Positional arguments forwarded to ``action`` when invoked directly.
            action_kwargs: Keyword arguments forwarded to ``action`` when invoked directly.
            inputs: Registry keys required before this step runs (``"__input__"`` for the raw image).
            name: Stable step identifier; auto-generated when the pipeline is built from a list.
            description: Human-readable label for logging and ``str(step)``.
            input_prepare: Optional transform applied to resolved inputs before inference.
        """
        self.action = action
        self.action_args = action_args or ()
        self.action_kwargs = action_kwargs or {}
        self.inputs = inputs
        self.input_prepare = input_prepare
        self.name = name
        self.description = description

    def __call__(self, *args, previous_result: Any = NO_VALUE, **kwargs) -> Any:
        """Invoke ``action``, optionally threading a prior step result.

        Args:
            *args: Positional arguments when ``previous_result`` is not set.
            previous_result: Output of the previous step in chained predict mode.
            **kwargs: Keyword arguments forwarded to ``action``.

        Returns:
            Whatever ``action`` returns.
        """
        if previous_result is NO_VALUE:
            return self.action(*args, **kwargs)
        return self.action(previous_result, *args, **kwargs)

    def __str__(self):
        return f"Step {self.name}: {self.description}"
