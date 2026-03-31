"""Shared helpers not tied to a single pipeline submodule."""

from __future__ import annotations

import numpy as np
import torch


def _normalize_for_imshow(arr: np.ndarray) -> np.ndarray:
    """Scale array values to a range valid for ``matplotlib.pyplot.imshow``.

    Args:
        arr (np.ndarray): Image data, often float or ``uint8``.

    Returns:
        np.ndarray: ``uint8`` passed through unchanged; floats clipped to ``[0, 1]``
        or normalized from ``0..255`` or min-max to ``[0, 1]``.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.size == 0:
        return arr
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    lo, hi = float(a.min()), float(a.max())
    if hi <= 1.0 + 1e-5 and lo >= -1e-5:
        return np.clip(a, 0.0, 1.0)
    if hi <= 255.0 + 1e-3 and lo >= -1e-3:
        return np.clip(a / 255.0, 0.0, 1.0)
    if hi > lo:
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return np.zeros_like(a, dtype=np.float32)


def to_pyplot_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Prepare a single image for ``matplotlib.pyplot.imshow``.

    Supports ``torch.Tensor`` or ``numpy.ndarray`` in **CHW** (as in
    ``TorchImageFolderDataset`` / ``to_model_tensor``) or **HWC**. If batched,
    shape must be ``(1, ...)``.

    Args:
        image (torch.Tensor | np.ndarray): Tensor or array; for CHW, channel
            count must be in ``{1, 3, 4}``.

    Raises:
        ValueError: If ``ndim == 4`` and batch size is not ``1``.

    Returns:
        np.ndarray: Shape ``(H, W)`` or ``(H, W, C)``; ``uint8`` unchanged,
        floats in ``[0, 1]`` including rescaling from ``0..255``.
    """
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            msg = (
                f"Expected a single sample (batch=1), got shape {arr.shape}"
            )
            raise ValueError(msg)
        arr = arr[0]

    if arr.ndim == 3:
        c_first, c_last = arr.shape[0], arr.shape[-1]
        # CHW: few channels on first axis, last axis is spatial
        if c_first in (1, 3, 4) and c_last not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)

    return _normalize_for_imshow(arr)
