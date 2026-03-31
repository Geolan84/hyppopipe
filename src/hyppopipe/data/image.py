from pathlib import Path

import pydicom
from numpy import asarray, ndarray
from PIL import Image as PILImage


class Image:
    """Wrapper around a numpy image array with optional keyword metadata."""

    def __init__(self, image: ndarray, **kwargs):
        """Store the array and arbitrary keyword arguments (e.g. ``class_``).

        Args:
            image (ndarray): Raw pixel data.
            **kwargs: Metadata attached to this image (e.g. label / class name).
        """
        self.image = image
        self.kwargs = kwargs

    def __array__(self, dtype=None) -> ndarray:
        """Numpy array protocol: return ``asarray(self.image)``.

        Args:
            dtype: Optional dtype for :func:`numpy.asarray`.

        Returns:
            ndarray: A view or copy of ``self.image``.
        """
        return asarray(self.image, dtype=dtype)

    @staticmethod
    def from_path(path: Path | str, file_type=None, **kwargs) -> "Image":
        """Load an image from disk (PIL-supported formats or DICOM ``.dcm``).

        Args:
            path (Path): File path. ``.dcm`` is read with pydicom; other suffixes
                go through PIL.
            file_type: Reserved for callers; not used internally.
            **kwargs: Passed to :class:`Image` (e.g. ``class_``).

        Returns:
            Self: New ``Image`` instance.
        """
        if isinstance(path, str):
            path = Path(path)
        if Path(path).suffix.lower() == ".dcm":
            dicom_file = pydicom.dcmread(path)
            img = asarray(dicom_file.pixel_array)
        else:
            img = asarray(PILImage.open(path))
        return Image(img, **kwargs)
