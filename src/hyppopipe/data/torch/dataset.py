from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self, get_args

import torch

from hyppopipe.common.types import FileType
from hyppopipe.data.image import Image
from hyppopipe.data.torch.transforms import TorchImageTransformRecipe


class TorchImageFolderDataset():
    """Image folder layout: ``train_dir`` / ``test_dir`` with per-class subfolders."""

    def __init__(
        self,
        root_dir: str,
        train_dir: str,
        test_dir: str,
        file_type: FileType | None = None,
        **kwargs,
    ):
        """Scan train and test trees and build an in-memory list of ``Image`` objects.

        Args:
            root_dir (str): Dataset root; ``train_dir`` and ``test_dir`` are
                resolved under this path.
            train_dir (str): Subfolder name for training images.
            test_dir (str): Subfolder name for test / validation images.
            file_type (FileType | None): Single extension filter; if ``None``,
                all literal file types in ``FileType`` are used.
            **kwargs: Passed through; ``transform_recipe`` (if present) replaces
                the default :class:`TorchImageTransformRecipe`.
        """
        transform_recipe = kwargs.pop("transform_recipe", None)
        self.root_dir = Path(root_dir)
        self.train_dir = self.root_dir / train_dir
        self.test_dir = self.root_dir / test_dir
        self.file_type = file_type
        self.kwargs = kwargs

        self._transforms = (
            transform_recipe
            if transform_recipe is not None
            else TorchImageTransformRecipe()
        )

        self.classes = sorted(d.name for d in self.train_dir.iterdir() if d.is_dir())
        exts = (
            frozenset({file_type})
            if file_type is not None
            else frozenset(get_args(FileType))
        )
        train_images = list(self.get_images_generator(self.train_dir, exts))
        test_images = list(self.get_images_generator(self.test_dir, exts))
        self._train_len = len(train_images)
        self.images: list[Image] = train_images + test_images

    @property
    def transform_recipe(self) -> TorchImageTransformRecipe:
        """Recipe used for ``__getitem__`` and fluent dataset methods.

        Returns:
            TorchImageTransformRecipe: The active recipe instance.
        """
        return self._transforms

    def get_images_generator(self, folder: Path, exts: frozenset[str]):
        """Yield ``Image`` instances for every matching file under ``folder/<class>``.

        Args:
            folder (Path): Usually ``train_dir`` or ``test_dir``.
            exts (frozenset[str]): Allowed file extensions (without dot).

        Returns:
            Generator[Image, None, None]: Lazy generator of ``Image.from_path`` results
            with ``class_`` set from folder name.
        """
        return (
            Image.from_path(f, class_=class_)
            for class_ in self.classes
            for ext in exts
            for f in sorted((folder / class_).glob(f"*.{ext}"))
            if f.is_file()
        )

    def num_classes(self) -> int:
        """Number of class subfolders discovered under ``train_dir``.

        Returns:
            int: ``len(self.classes)``.
        """
        return len(self.classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Map class directory name to integer label.

        Returns:
            dict[str, int]: Name to index in ``range(num_classes)``.
        """
        return {name: i for i, name in enumerate(self.classes)}

    @property
    def train_indices(self) -> range:
        """Indices in ``self.images`` that belong to the train split.

        Returns:
            range: ``0 .. len(train) - 1``.
        """
        return range(self._train_len)

    @property
    def test_indices(self) -> range:
        """Indices in ``self.images`` that belong to the test split.

        Returns:
            range: From end of train through ``len(self.images) - 1``.
        """
        return range(self._train_len, len(self.images))

    def __len__(self) -> int:
        """Total number of images (train + test).

        Returns:
            int: Length of ``self.images``.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return transformed CHW tensor and integer class index.

        Args:
            index (int): Index into ``self.images`` (after transforms in recipe).

        Returns:
            tuple[torch.Tensor, int]: Model input tensor and label.
        """
        image = self.get_image(index)
        class_name = image.kwargs["class_"]
        label = self.class_to_idx[class_name]
        tensor = self._transforms.to_model_tensor(image)
        return tensor, int(label)

    def resize(
        self,
        size: int | Sequence[int],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Forward ``resize`` to the internal :class:`TorchImageTransformRecipe`.

        Args:
            size (int | Sequence[int]): Passed to ``transform_recipe.resize``.
            *args: Forwarded.
            **kwargs: Forwarded.

        Returns:
            Self: This dataset for chaining.
        """
        self._transforms.resize(size, *args, **kwargs)
        return self

    def rotate(self, degrees: float | tuple[float, float]) -> Self:
        """Forward ``rotate`` to the internal recipe.

        Args:
            degrees (float | tuple[float, float]): Rotation range.

        Returns:
            Self: This dataset for chaining.
        """
        self._transforms.rotate(degrees)
        return self

    def circle_crop(self) -> Self:
        """Forward ``circle_crop`` to the internal recipe.

        Returns:
            Self: This dataset for chaining.
        """
        self._transforms.circle_crop()
        return self

    def get_image(self, index: int) -> Image:
        """Load and apply the full recipe to ``self.images[index]``.

        Args:
            index (int): Dataset index.

        Returns:
            Image: Transformed image wrapper.
        """
        return self._transforms.apply(self.images[index])
