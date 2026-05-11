from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module

from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.pipeline.image.localization import ImageLocalizer
from hyppopipe.train.bundle import StepArtifact
from hyppopipe.train.model_spec import instantiate_base_from_spec
from hyppopipe.train.tasks.classification import prepare_classification_model_from_meta
from hyppopipe.train.tasks.detection import prepare_detection_model_from_meta


def build_and_load_step_model(
    step_name: str,
    artifact: StepArtifact,
    step_action: Any,
    *,
    device: torch.device,
    step_base_models: dict[str, Module] | None = None,
) -> Module:
    meta = artifact.inference_meta
    task = meta.get("task")
    spec = artifact.model_spec

    if spec.get("kind") == "torchvision_factory":
        base = instantiate_base_from_spec(spec)
    elif step_base_models is not None and step_name in step_base_models:
        base = step_base_models[step_name]
    else:
        msg = (
            f"Step {step_name!r}: cannot rebuild base model from spec {spec.get('kind')!r}; "
            "use ModelCandidate(torchvision_fn, weights=[...]) when training, "
            f"or pass step_base_models[{step_name!r}] when calling predict."
        )
        raise ValueError(msg)

    if task == "classification":
        if not isinstance(step_action, ImageClassifier):
            msg = f"Step {step_name!r}: expected ImageClassifier action"
            raise TypeError(msg)
        num_classes = meta.get("num_classes")
        in_ch = meta.get("canonical_in_channels")
        if num_classes is None or in_ch is None:
            msg = f"Step {step_name!r}: inference_meta missing num_classes or canonical_in_channels"
            raise ValueError(msg)
        prepared = prepare_classification_model_from_meta(
            base,
            num_classes=int(num_classes),
            canonical_in_channels=int(in_ch),
        )
    elif task == "detection":
        if not isinstance(step_action, ImageLocalizer):
            msg = f"Step {step_name!r}: expected ImageLocalizer action"
            raise TypeError(msg)
        n_cls = meta.get("num_classes")
        if n_cls is None:
            msg = f"Step {step_name!r}: inference_meta missing num_classes"
            raise ValueError(msg)
        prepared = prepare_detection_model_from_meta(base, num_classes=int(n_cls))
    else:
        msg = f"Unsupported inference task {task!r} for step {step_name!r}"
        raise ValueError(msg)

    state = torch.load(artifact.weights_path, weights_only=True, map_location=device)
    prepared.load_state_dict(state)
    return prepared.to(device).eval()
