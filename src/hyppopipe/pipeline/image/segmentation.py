"""Semantic and instance segmentation step functor."""

from __future__ import annotations

from typing import Literal

SegmentationKind = Literal["instance", "semantic"]


class ImageSegmentator:
    """Pipeline step for semantic or instance segmentation."""

    def __init__(
        self,
        *,
        kind: SegmentationKind = "instance",
        num_classes: int | None = None,
        input_channels: int = 3,
        image_size: tuple[int, int] | None = (224, 224),
    ) -> None:
        """Store segmentation training options.

        Args:
            kind: Desired **label format**: ``"instance"`` (Mask R-CNN targets) or
                ``"semantic"`` (class map). Training may adapt when the model architecture
                differs (see ``SegmentationTrainingTask``).
            num_classes: Classes including background; inferred from data when None.
            input_channels: Channel count after default semantic input preparation.
            image_size: ``(H, W)`` for semantic batching; instance models resize internally.

        Raises:
            ValueError: If ``kind`` is not ``"instance"`` or ``"semantic"``.
        """
        if kind not in ("instance", "semantic"):
            raise ValueError("kind must be 'instance' or 'semantic'")
        self.kind = kind
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size

    def __call__(self) -> None:
        """Marker callable; training uses attributes, not runtime invocation."""
        ...
