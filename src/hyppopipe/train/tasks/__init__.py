from hyppopipe.train.tasks.base import TrainingTask, model_label_for_module
from hyppopipe.train.tasks.detection import DetectionTrainingTask
from hyppopipe.train.tasks.dispatch import dispatch_training_task
from hyppopipe.train.tasks.segmentation import SegmentationTrainingTask

__all__ = [
    "DetectionTrainingTask",
    "SegmentationTrainingTask",
    "TrainingTask",
    "dispatch_training_task",
    "model_label_for_module",
]
