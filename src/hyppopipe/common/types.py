from enum import StrEnum
from typing import Literal

BackendType = Literal["torch"]


class Backend(StrEnum):
    TORCH = "torch"


FileType = Literal["png", "jpg", "dcm", "tif"]
