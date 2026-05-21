"""Dataset configuration and task-compatibility exceptions."""

from hyppopipe.types import HyppopipeError


class InvalidDatasetConfigError(HyppopipeError):
    """Raised when a dataset YAML or config block is invalid."""

    def __init__(self, message: str):
        """Store a human-readable configuration error.

        Args:
            message: Description of what is wrong with the config.
        """
        self.message = message

    def __str__(self):
        return f"Invalid dataset config: {self.message}"


class ClassificationDataUnsupportedError(HyppopipeError):
    """Raised when a split cannot be adapted for classification training."""

    def __init__(self, message: str):
        """Store why classification adaptation failed.

        Args:
            message: Explanation for the caller.
        """
        self.message = message

    def __str__(self) -> str:
        return self.message


class DetectionDataUnsupportedError(HyppopipeError):
    """Raised when a split cannot be adapted for detection training."""

    def __init__(self, message: str):
        """Store why detection adaptation failed.

        Args:
            message: Explanation for the caller.
        """
        self.message = message

    def __str__(self) -> str:
        return self.message


class SegmentationDataUnsupportedError(HyppopipeError):
    """Raised when a split cannot be adapted for segmentation training."""

    def __init__(self, message: str):
        """Store why segmentation adaptation failed.

        Args:
            message: Explanation for the caller.
        """
        self.message = message

    def __str__(self) -> str:
        return self.message
