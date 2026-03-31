import cv2
import torch
import numpy as np


def to_chw(x: torch.Tensor) -> torch.Tensor:
    """Reshape a tensor to channel-first ``(C, H, W)``.

    Args:
        x (torch.Tensor): Tensor of shape ``(H, W)`` or ``(C, H, W)``.

    Raises:
        ValueError: If ``ndim`` is neither 2 nor 3.

    Returns:
        torch.Tensor: Tensor of shape ``(C, H, W)``.
    """
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected (H,W) or (C,H,W), got {tuple(x.shape)}")


def _crop_image_bgr(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Crop image to the minimum enclosing circle of the largest contour.

    Input and output are BGR ``uint8`` arrays as used by OpenCV.

    Args:
        image (np.ndarray): BGR image.

    Returns:
        tuple[np.ndarray, int]: Cropped BGR image and a flag ``1`` if a contour
        was found and used, ``0`` if the original image is returned unchanged.
    """
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("no contours!")
        flag = 0
        return image, flag
    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x)
    y = int(y)
    r = int(r)
    flag = 1
    h, w = output.shape[:2]

    pad_left = max(0, r - x)
    pad_top = max(0, r - y)
    pad_right = max(0, (x + r) - w)
    pad_bottom = max(0, (y + r) - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        output = cv2.copyMakeBorder(
            output, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT
        )
        x, y = x + pad_left, y + pad_top
    cropped = output[y - r : y + r, x - r : x + r]
    return cropped, flag


def _numpy_circle_crop(arr: np.ndarray) -> np.ndarray:
    """Crop a numpy image to the minimum enclosing circle (RGB/RGBA or gray).

    Accepts typical PIL-style ``HW``, ``HWC`` with 3 or 4 channels; converts
    internally to BGR for OpenCV, then maps the result back to the input layout.

    Args:
        arr (np.ndarray): Grayscale ``(H, W)``, RGB ``(H, W, 3)``, or RGBA
            ``(H, W, 4)``. Other shapes are returned unchanged.

    Returns:
        np.ndarray: Circle-cropped array in the same channel layout as the input,
        or the original ``arr`` if no valid contour was found or shape is unsupported.
    """
    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        as_gray = True
        alpha = False
    elif arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        as_gray = False
        alpha = False
    elif arr.ndim == 3 and arr.shape[2] == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        as_gray = False
        alpha = True
    else:
        return arr

    cropped, flag = _crop_image_bgr(bgr)
    if flag == 0:
        return arr

    if as_gray:
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    if alpha:
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA)
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
