from __future__ import annotations

from typing import Protocol, runtime_checkable

from hyppopipe.pipeline.context import PipelineContext


@runtime_checkable
class PipelineStep(Protocol):
    """One pipeline stage that mutates :class:`PipelineContext` (sklearn-like ``fit``)."""

    def fit(self, ctx: PipelineContext) -> None: ...


class Pipeline:
    """Ordered named steps; :meth:`fit` runs the context through each step."""

    def __init__(self, steps: list[tuple[str, PipelineStep]]) -> None:
        """Store a shallow copy of the step list.

        Args:
            steps (list[tuple[str, PipelineStep]]): ``(name, step)`` pairs executed
                in order by :meth:`fit`.
        """
        self.steps = list(steps)

    def named_steps(self) -> list[tuple[str, PipelineStep]]:
        """Return the configured steps.

        Returns:
            list[tuple[str, PipelineStep]]: Current ``(name, step)`` sequence.
        """
        return list(self.steps)

    def fit(self, ctx: PipelineContext | None = None) -> PipelineContext:
        """Run ``fit`` on each step in order.

        Args:
            ctx (PipelineContext | None, optional): Starting context; if ``None``,
                a new :class:`PipelineContext` is created. Defaults to ``None``.

        Returns:
            PipelineContext: The same instance after all steps have run.
        """
        context = ctx if ctx is not None else PipelineContext()
        for _name, step in self.steps:
            step.fit(context)
        return context

    def run(self, ctx: PipelineContext | None = None) -> PipelineContext:
        """Alias for :meth:`fit` (same behavior).

        Args:
            ctx (PipelineContext | None, optional): Passed to :meth:`fit`.

        Returns:
            PipelineContext: Result of :meth:`fit`.
        """
        return self.fit(ctx)
