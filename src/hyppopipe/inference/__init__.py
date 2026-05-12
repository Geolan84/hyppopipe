from typing import Any

from hyppopipe.inference.types import (
    ClassificationPrediction,
    LocalizationPrediction,
    PipelinePrediction,
    SegmentationPrediction,
)

__all__ = [
    "ClassificationPrediction",
    "LocalizationPrediction",
    "PipelinePrediction",
    "SegmentationPrediction",
    "build_and_load_step_model",
    "run_step_inference",
]


def __getattr__(name: str) -> Any:
    if name == "build_and_load_step_model":
        from hyppopipe.inference.model_builder import build_and_load_step_model

        return build_and_load_step_model
    if name == "run_step_inference":
        from hyppopipe.inference.run import run_step_inference

        return run_step_inference
    raise AttributeError(name)
