from hyppopipe.train.tasks.base import TrainingTask, model_label_for_module
from hyppopipe.train.tasks.detection import DetectionTrainingTask
from hyppopipe.train.tasks.dispatch import dispatch_training_task

__all__ = [
    "DetectionTrainingTask",
    "TrainingTask",
    "dispatch_training_task",
    "model_label_for_module",
]
