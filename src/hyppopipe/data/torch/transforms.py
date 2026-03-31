"""
Image transform pipeline mixing numpy manipulations and ``torchvision`` ops.

The same recipe can drive a dataset and a single file from disk so inference
matches training. Datasets store a copy in ``transform_recipe``; for disk files:

.. code-block:: python

    from hyppopipe.data.image import Image

    recipe = TorchImageTransformRecipe().resize((224, 224)).circle_crop()
    x = recipe.to_model_tensor(recipe.apply(Image.from_path(path)))
    # x is CHW, like a DataLoader batch without the batch dimension
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np
import torch
from torch import from_numpy
from torchvision import transforms

from hyppopipe.data.image import Image
from hyppopipe.utils import _numpy_circle_crop, to_chw

Manipulator = Callable[[np.ndarray], np.ndarray]


def _circle_crop_array(arr: np.ndarray) -> np.ndarray:
    """Apply circle crop via :func:`hyppopipe.utils._numpy_circle_crop`.

    Args:
        arr (np.ndarray): Image array in PIL-style layout.

    Returns:
        np.ndarray: Cropped array.
    """
    return _numpy_circle_crop(arr)


class TorchImageTransformRecipe:
    """Fluent image preprocessing; matches the former ``get_image`` dataset path.

    Structurally satisfies :class:`~hyppopipe.common.protocols.ImageTransformable`.
    """

    def __init__(self) -> None:
        """Initialize empty manipulation and torchvision transform lists."""
        self.manipulations: list[Manipulator] = []
        self.transformations: list[Any] = []

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

    def circle_crop(self) -> Self:
        """Append minimum enclosing-circle crop on the numpy stage (before TV).

        Returns:
            Self: This recipe for chaining.
        """
        self.manipulations.append(_circle_crop_array)
        return self

    def apply(self, source: Image) -> Image:
        """Run numpy manipulations then torchvision; return a new ``Image``.

        Preserves ``source.kwargs`` on the result.

        Args:
            source (Image): Input wrapper around a numpy array.

        Returns:
            Image: Transformed image as float32 array scaled to ``[0, 1]`` (typ.
            ``HW`` or ``HWC`` after inverse CHW layout).
        """
        arr = np.asarray(source.image, copy=True)
        for fn in self.manipulations:
            arr = fn(arr)

        if not self.transformations:
            return Image(arr, **source.kwargs)

        tensor_image = from_numpy(arr.copy())
        if tensor_image.ndim == 3 and tensor_image.shape[2] in (1, 3, 4):
            tensor_image = tensor_image.permute(2, 0, 1).contiguous()
        elif tensor_image.ndim == 2:
            tensor_image = tensor_image.unsqueeze(0)
        else:
            tensor_image = to_chw(tensor_image)

        tensor_image = tensor_image.float()
        if tensor_image.numel() and tensor_image.max() > 1.0:
            tensor_image = tensor_image / 255.0

        transform = transforms.Compose(self.transformations)
        tensor_image = transform(tensor_image)
        out = tensor_image.detach().cpu().numpy()

        if out.ndim == 3 and out.shape[0] in (1, 3, 4):
            out = np.transpose(out, (1, 2, 0))
            if out.shape[2] == 1:
                out = np.squeeze(out, axis=2)

        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        return Image(out, **source.kwargs)

    def to_model_tensor(self, image: Image) -> torch.Tensor:
        """Convert ``image`` to CHW ``float`` tensor (dataset ``__getitem__`` format).

        Args:
            image (Image): Post-``apply`` image (or compatible array layout).

        Returns:
            torch.Tensor: Shape ``(C, H, W)``, same convention as
            ``TorchImageFolderDataset`` items before batching.
        """
        arr = np.asarray(image.image, dtype=np.float32)
        tensor = torch.from_numpy(arr.copy())
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()
        else:
            tensor = to_chw(tensor).contiguous()
        return tensor
