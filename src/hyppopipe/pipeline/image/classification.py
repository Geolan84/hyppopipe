"""Classification step functor."""

from __future__ import annotations

from typing import Literal

SourceMode = Literal["full", "roi"]


class ImageClassifier:
    """Classification step functor; fields configure `classification.ClassificationTrainingTask`."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        source_mode: SourceMode = "full",
    ) -> None:
        """Store classification training options.

        Args:
            num_classes: Output classes; inferred from the dataset when None.
            source_mode: ``"full"`` uses the whole image; ``"roi"`` expects a crop from a prior localize step.
        """
        self.num_classes = num_classes
        self.source_mode = source_mode

    def __call__(self) -> None:
        """Marker callable; training uses attributes, not runtime invocation."""
        ...
