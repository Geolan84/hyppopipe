from dataclasses import replace
from typing import Any, Self, Sequence

from torch import Tensor
from torchvision import transforms

from hyppopipe.data.image import Image


class ImageTransformer:
    __slots__ = ("transformations", "_composed")

    def __init__(self) -> None:
        self.transformations: list[Any] = []
        self._composed = transforms.Compose(self.transformations)

    def __call__(self, image: Image | Tensor) -> Image | Tensor:
        original_image = image
        if isinstance(image, Image):
            image = image.tensor
        image = self._composed(image)
        if isinstance(original_image, Image):
            return replace(original_image, body=image)
        return image

    def resize(
        self,
        size: int | Sequence[int],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Append ``torchvision.transforms.Resize`` to the torchvision stage.

        Args:
            size (int | Sequence[int]): Target ``(H, W)``; if int, becomes
                ``(size, size)``.
            *args: Forwarded to ``Resize``.
            **kwargs: Forwarded to ``Resize``. ``antialias`` defaults to ``True``
                if not given.

        Returns:
            Self: This recipe for chaining.
        """
        hw = (size, size) if isinstance(size, int) else tuple(size)
        antialias = kwargs.pop("antialias", True)
        self.transformations.append(
            transforms.Resize(hw, antialias=antialias, *args, **kwargs)
        )
        return self

    def rotate(self, degrees: float | tuple[float, float]) -> Self:
        """Append ``torchvision.transforms.RandomRotation``.

        Args:
            degrees (float | tuple[float, float]): Single angle or ``(min, max)``
                range in degrees.

        Returns:
            Self: This recipe for chaining.
        """
        if isinstance(degrees, float):
            degrees = (degrees, degrees)
        self.transformations.append(transforms.RandomRotation(degrees))
        return self

    def clear(self) -> None:
        self.transformations.clear()
