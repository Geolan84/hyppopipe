from collections.abc import Sequence
from typing import Any, Protocol, Self


class Rotatable(Protocol):
    def rotate(self, degrees: float | tuple[float, float]) -> Self: ...


class Croppable(Protocol):
    def crop(self, *args: Any, **kwargs: Any) -> Self: ...


class CircularCroppable(Protocol):
    def circle_crop(self) -> Self: ...


class Resizable(Protocol):
    def resize(
        self,
        size: int | Sequence[int],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...


class CenterCroppable(Protocol):
    def center_crop(
        self,
        size: int | Sequence[int],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...


class RandomCroppable(Protocol):
    def random_crop(
        self,
        size: int | Sequence[int],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...


class RandomResizedCroppable(Protocol):
    def random_resized_crop(
        self,
        size: int | Sequence[int],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...


class HorizontallyFlippable(Protocol):
    def horizontal_flip(self, p: float = 0.5) -> Self: ...


class VerticallyFlippable(Protocol):
    def vertical_flip(self, p: float = 0.5) -> Self: ...


class Normalizable(Protocol):
    def normalize(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...


class TensorConvertible(Protocol):
    def to_tensor(self, *args: Any, **kwargs: Any) -> Self: ...


class Grayscalable(Protocol):
    def grayscale(self, num_output_channels: int = 1) -> Self: ...


class ColorJitterable(Protocol):
    def color_jitter(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
    ) -> Self: ...


class Paddable(Protocol):
    def pad(
        self,
        padding: int | Sequence[int],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...


class GaussianBlurrable(Protocol):
    def gaussian_blur(
        self,
        kernel_size: int | Sequence[int],
        sigma: float | Sequence[float] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...


class AffinelyTransformable(Protocol):
    def affine(
        self,
        degrees: float | Sequence[float],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...


class ImageTransformable(Rotatable, CircularCroppable, Resizable, Protocol):
    """Fluent image preprocessing API shared by datasets and recipes.

    See :class:`~hyppopipe.data.torch.transforms.TorchImageTransformRecipe`.
    """
