from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import cv2
import filetype
import numpy as np
from matplotlib import pyplot as plt
from pydicom import dcmread
from torch import Tensor, from_numpy
from torchvision.io import decode_image

from hyppopipe.data.base import DataResource

SUPPORTED_FILE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".dcm": "application/dicom",
}


def _ensure_content_matches_suffix(path: Path, suffix: str) -> None:
    expected = SUPPORTED_FILE_TYPES[suffix]
    kind = filetype.guess(str(path))
    if kind is None:
        raise ValueError(
            f"File {path} could not be identified as a valid {suffix} payload (unknown signature)"
        )
    if kind.mime != expected:
        raise ValueError(
            f"File {path} is detected as {kind.mime!r}, expected {expected} for extension {suffix}"
        )


def _decode_tiff(path: Path) -> Tensor:
    """
    Loads TIFF as uint8 CHW RGB to align with ``torchvision.io.decode_image``.
    Uses OpenCV (already a project dependency); ``decode_image`` does not support TIFF.
    """
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"File {path} could not be decoded as TIFF")
    if arr.dtype != np.uint8:
        arr = cv2.normalize(
            arr,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        return from_numpy(arr).unsqueeze(0)
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 3:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(
                f"File {path} has unsupported TIFF channel layout (shape {arr.shape})"
            )
        return from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    raise ValueError(f"File {path} has unsupported TIFF array shape {arr.shape}")


@dataclass(frozen=True, slots=True)
class Image(DataResource):
    body: Tensor
    sample_id: str | None = None
    legend: dict[str, Any] | None = None

    def show(self, **kwargs) -> None:
        """
        Shows image using matplotlib pyplot graph.
        """
        x = self.body.detach().cpu()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
        if x.shape[0] == 1:
            img = x[0].numpy()
            plt.imshow(img, cmap="gray")
        else:
            img = x[:3].permute(1, 2, 0).numpy()
            plt.imshow(img)
        if self.sample_id:
            plt.title(self.sample_id)
        plt.axis("off")
        plt.show(**kwargs)

    @classmethod
    def from_path(cls, path: Path | str) -> Self:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"File {path} is not a file")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"File {path} is not a supported image file")
        _ensure_content_matches_suffix(path, suffix)
        sample_id = path.with_suffix("").name
        match suffix:
            case ".dcm":
                return cls(from_numpy(dcmread(path).pixel_array).float(), sample_id)
            case ".tif" | ".tiff":
                return cls(_decode_tiff(path), sample_id)
            case _:
                return cls(decode_image(str(path)), sample_id)

    @property
    def tensor(self) -> Tensor:
        return self.body

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.body.shape)
