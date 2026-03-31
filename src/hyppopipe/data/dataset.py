from typing import Any, Iterator, Protocol

from hyppopipe.common.protocols import ImageTransformable
from hyppopipe.common.types import Backend, FileType


class ImageDataset(ImageTransformable, Protocol):
    """Protocol for image datasets with class labels and iteration."""

    def __iter__(self) -> Iterator[Any]: ...
    def num_classes(self) -> int | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


def make_folder_dataset(
    root_dir: str,
    file_type: FileType,
    backend: Backend = Backend.TORCH,
    train_dir: str = "train",
    test_dir: str = "test",
    **kwargs,
) -> ImageDataset:
    """Construct a folder-based image dataset for the given backend.

    Args:
        root_dir (str): Dataset root directory.
        file_type (FileType): File extension filter for image files.
        backend (Backend): Data stack; currently only ``Backend.TORCH`` is supported.
        train_dir (str): Train subdirectory name under ``root_dir``.
        test_dir (str): Test subdirectory name under ``root_dir``.
        **kwargs: Forwarded to the concrete dataset (e.g. ``transform_recipe``).

    Raises:
        ValueError: If ``backend`` is not supported.

    Returns:
        ImageDataset: A backend-specific dataset implementing ``ImageDataset``.
    """
    match backend:
        case Backend.TORCH:
            from hyppopipe.data.torch.dataset import TorchImageFolderDataset

            return TorchImageFolderDataset(
                root_dir, train_dir, test_dir, file_type, **kwargs
            )  # ty:ignore[invalid-return-type]
        case _:
            raise ValueError(f"Unsupported backend: {backend}")
