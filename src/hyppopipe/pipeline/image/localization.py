"""Object detection / localization step functor."""

from __future__ import annotations


class ImageLocalizer:
    """Pipeline step for localization and detection (torchvision Faster R-CNN training)."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
    ) -> None:
        """Store detection training options.

        Args:
            num_classes: Total classes **including background** for torchvision Faster R-CNN.
                When None, uses ``len(dataset.classes) + 1``.
        """
        self.num_classes = num_classes

    def __call__(self) -> None:
        """Marker callable; training uses attributes, not runtime invocation."""
        ...
