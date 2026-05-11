from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from hyppopipe.data.image import Image

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_IMAGES = ROOT / "examples" / "images"


@pytest.mark.parametrize(
    ("filename", "expected_sample_id"),
    [
        ("10_left.jpeg", "10_left"),
        ("140_G.png", "140_G"),
        ("image74prime.tif", "image74prime"),
        ("IM000001.dcm", "IM000001"),
    ],
)
def test_from_path_loads_supported_examples(
    filename: str, expected_sample_id: str
) -> None:
    path = EXAMPLES_IMAGES / filename
    img = Image.from_path(path)
    t = img.body
    assert (
        isinstance(img, Image)
        and img.sample_id == expected_sample_id
        and isinstance(t, torch.Tensor)
        and t.ndim >= 2
        and t.numel() > 0
    )


def test_from_path_accepts_str_path() -> None:
    path = EXAMPLES_IMAGES / "140_G.png"
    img = Image.from_path(str(path))
    assert img.sample_id == "140_G"


def test_from_path_missing_file_raises() -> None:
    missing = EXAMPLES_IMAGES / "does_not_exist_ever.png"
    with pytest.raises(ValueError, match="is not a file"):
        Image.from_path(missing)


def test_from_path_directory_raises() -> None:
    with pytest.raises(ValueError, match="is not a file"):
        Image.from_path(EXAMPLES_IMAGES)


def test_from_path_unsupported_suffix_raises(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("not an image", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="not a supported image file",
    ):
        Image.from_path(p)


def test_from_path_mime_mismatch_raises(tmp_path: Path) -> None:
    """JPEG bytes with .png extension — filetype detects jpeg, code expects png."""
    src = EXAMPLES_IMAGES / "10_left.jpeg"
    dst = tmp_path / "disguised.png"
    shutil.copyfile(src, dst)
    with pytest.raises(ValueError, match="detected as"):
        Image.from_path(dst)


def test_from_path_strict_false_allows_extension_mime_mismatch(tmp_path: Path) -> None:
    """PNG payload under .jpg — типичный случай «ложного» расширения в датасетах."""
    src = EXAMPLES_IMAGES / "140_G.png"
    dst = tmp_path / "wrong.jpg"
    shutil.copyfile(src, dst)
    img = Image.from_path(dst, strict=False)
    assert isinstance(img.body, torch.Tensor) and img.body.numel() > 0


def test_from_path_unknown_signature_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.png"
    p.write_bytes(b"not a real png")
    with pytest.raises(
        ValueError,
        match="unknown signature|could not be identified",
    ):
        Image.from_path(p)
