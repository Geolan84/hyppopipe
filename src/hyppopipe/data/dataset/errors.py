from hyppopipe.types import HyppopipeError


class InvalidDatasetConfigError(HyppopipeError):
    """Invalid dataset config."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f"Invalid dataset config: {self.message}"


class ClassificationDataUnsupportedError(HyppopipeError):
    """Split or dataset does not provide data for classification."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


class DetectionDataUnsupportedError(HyppopipeError):
    """Split or dataset does not provide detection data."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message
