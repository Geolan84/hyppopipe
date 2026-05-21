"""Medical and general-purpose image loading as torch tensors."""

import base64
import binascii
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import filetype
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from pydicom import dcmread
from torch import Tensor, from_numpy, frombuffer, uint8
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


@dataclass(frozen=True, slots=True)
class Image(DataResource):
    """Immutable image container with optional sample metadata.

    Example:
        Load from disk and display::

            img = Image.from_path("scan.png")
            img.show()

            other_img = Image.from_base64("...")
            other_img.show()

    Attributes:
        body: Image tensor, typically uint8 CHW from decoders.
        sample_id: Optional identifier (filename stem when loaded from a path).
        legend: Optional arbitrary metadata attached to the sample.
    """

    body: Tensor
    sample_id: str | None = None
    legend: dict[str, Any] | None = None

    @classmethod
    def from_base64(cls, payload: str | bytes) -> Self:
        """Load an image from base64.

        Accepts payloads produced by :attr:`base64` (numpy array archive) or
        standard base64-encoded image files (PNG, JPEG, etc.).
        """
        if isinstance(payload, str):
            payload = payload.encode("ascii")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError("Invalid base64 payload") from e
        if raw.startswith(b"\x93NUMPY"):
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            return cls(from_numpy(np.ascontiguousarray(arr)))
        try:
            return cls(decode_image(frombuffer(raw, dtype=uint8).clone()))
        except RuntimeError as e:
            raise ValueError(
                "Base64 payload is neither a tensor archive nor a decodable image"
            ) from e

    @classmethod
    def from_path(cls, path: Path | str, *, strict: bool = True) -> Self:
        """Load an image from a supported file path.

        Args:
            path: File path (JPEG, PNG, TIFF, or DICOM).
            strict: If True, verify content type matches the extension.

        Returns:
            Loaded image with ``sample_id`` set to the path stem.

        Raises:
            ValueError: If the path is invalid or the format is unsupported.
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"File {path} is not a file")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"File {path} is not a supported image file")
        try:
            if strict:
                _ensure_content_matches_suffix(path, suffix)
            sample_id = path.with_suffix("").name
            match suffix:
                case ".dcm":
                    return cls(from_numpy(dcmread(path).pixel_array).float(), sample_id)
                case ".tif" | ".tiff":
                    return cls(_decode_tiff(path), sample_id)
                case _:
                    return cls(decode_image(str(path)), sample_id)
        except PermissionError:
            raise ValueError(f"File {path} is not accessible")

    def show(self, **kwargs) -> None:
        """Display the image with matplotlib."""
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

    def save(self, path: Path | str) -> None:
        """
        Save the image to a file.

        Args:
            path (Path | str): Path to save the image to.
        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FILE_TYPES or suffix == ".dcm":
            raise ValueError(f"File {path} is not a supported image file")
        pil_img = _body_to_pil_image(self.body)
        if suffix in {".jpg", ".jpeg"} and pil_img.mode == "RGBA":
            pil_img = pil_img.convert("RGB")
        pil_img.save(path)

    @property
    def as_gray(self) -> Tensor:
        """Single-channel grayscale tensor derived from ``body``."""
        return transforms.Grayscale(num_output_channels=1)(self.body)

    @property
    def tensor(self) -> Tensor:
        """Underlying image tensor."""
        return self.body

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of ``body``."""
        return tuple(self.body.shape)

    @property
    def base64(self) -> bytes:
        """Base64-encoded tensor archive (round-trips with :meth:`from_base64`)."""
        buf = io.BytesIO()
        np.save(buf, self.body.detach().cpu().contiguous().numpy(), allow_pickle=False)
        return base64.b64encode(buf.getvalue())

    @property
    def base64png(self) -> bytes:
        """Base64-encoded PNG (decodable with OpenCV, Pillow, etc.)."""
        return base64.b64encode(_tensor_to_png_bytes(self.body))


def _ensure_content_matches_suffix(path: Path, suffix: str) -> None:
    """Verify file magic bytes match the path extension.

    Raises:
        ValueError: If the file type cannot be identified or does not match.
    """
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
    """Load TIFF as uint8 CHW RGB (Pillow; ``decode_image`` does not support TIFF)."""
    try:
        pil_img = PILImage.open(path)
        pil_img.load()
    except OSError as e:
        raise ValueError(f"File {path} could not be decoded as TIFF") from e

    if pil_img.mode == "P":
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode == "PA":
        pil_img = pil_img.convert("RGBA")
    elif pil_img.mode == "LA":
        pil_img = pil_img.convert("L")
    elif pil_img.mode == "CMYK":
        pil_img = pil_img.convert("RGB")

    arr = np.asarray(pil_img)
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float64, copy=False)
        arr_min = arr_f.min()
        arr_max = arr_f.max()
        if arr_max > arr_min:
            arr = ((arr_f - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
        else:
            arr = np.zeros(arr.shape, dtype=np.uint8)

    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        return from_numpy(arr.copy()).unsqueeze(0)
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 3:
            rgb = arr
        elif channels == 4:
            rgb = arr[:, :, :3]
        else:
            raise ValueError(
                f"File {path} has unsupported TIFF channel layout (shape {arr.shape})"
            )
        return from_numpy(np.ascontiguousarray(rgb).copy()).permute(2, 0, 1)
    raise ValueError(f"File {path} has unsupported TIFF array shape {arr.shape}")


def _body_to_pil_image(body: Tensor) -> PILImage.Image:
    """Convert a CHW (or HW/HWC) tensor to a Pillow image for saving."""
    x = body.detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        x = x.permute(2, 0, 1)
    if x.ndim != 3:
        raise ValueError(
            f"Cannot encode tensor with shape {tuple(body.shape)} as image"
        )

    arr: np.ndarray = x.numpy()
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float64, copy=False)
        arr_min = arr_f.min()
        arr_max = arr_f.max()
        if arr_max > arr_min:
            arr = ((arr_f - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
        else:
            arr = np.zeros(arr.shape, dtype=np.uint8)

    channels = arr.shape[0]
    if channels == 1:
        return PILImage.fromarray(arr[0], mode="L")
    if channels == 3:
        return PILImage.fromarray(arr.transpose(1, 2, 0), mode="RGB")
    if channels == 4:
        return PILImage.fromarray(arr.transpose(1, 2, 0), mode="RGBA")
    raise ValueError(f"Unsupported channel count {channels} for image encoding")


def _tensor_to_png_bytes(body: Tensor) -> bytes:
    """Encode a CHW (or HW/HWC) tensor as PNG file bytes."""
    buf = io.BytesIO()
    _body_to_pil_image(body).save(buf, format="PNG")
    return buf.getvalue()
