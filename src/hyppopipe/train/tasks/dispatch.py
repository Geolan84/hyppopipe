from __future__ import annotations

from typing import Any

from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.pipeline.image.localization import ImageLocalizer
from hyppopipe.pipeline.image.segmentation import ImageSegmentator
from hyppopipe.train.tasks.base import TrainingTask
from hyppopipe.train.tasks.classification import ClassificationTrainingTask
from hyppopipe.train.tasks.detection import DetectionTrainingTask
from hyppopipe.train.tasks.segmentation import SegmentationTrainingTask


def dispatch_training_task(step_action: Any) -> TrainingTask:
    """Resolve the training strategy from the pipeline step functor."""
    match step_action:
        case ImageClassifier():
            return ClassificationTrainingTask(step_action)
        case ImageLocalizer():
            return DetectionTrainingTask(step_action)
        case ImageSegmentator():
            return SegmentationTrainingTask(step_action)
        case _:
            msg = (
                f"Step type {type(step_action).__name__!r} is not supported for training; "
                "add a branch in dispatch_training_task or use a supported step functor"
            )

    raise TypeError(msg)
