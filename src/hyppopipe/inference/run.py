from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from hyppopipe.data.image import Image
from hyppopipe.inference.model_builder import build_and_load_step_model
from hyppopipe.inference.types import (
    ClassificationPrediction,
    LocalizationPrediction,
    SegmentationPrediction,
)
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.pipeline.image.localization import ImageLocalizer
from hyppopipe.pipeline.image.segmentation import ImageSegmentator
from hyppopipe.pipeline.step import Step
from hyppopipe.train.bundle import StepArtifact
from hyppopipe.train.tasks.classification import (
    _ensure_channel_count,
    _normalize_tensor_imagenet_style,
    default_classification_transform,
)

logger = logging.getLogger(__name__)


def _tensor_for_detection(image: Image, device: torch.device) -> Tensor:
    x = image.body
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    else:
        x = x.float()
        if x.max() > 1.5:
            x = x.div_(255.0)
    return x.to(device)


def _crop_xyxy(chw: Tensor, box: Tensor) -> Tensor:
    """CHW tensor, box xyxy in pixel coords (same space as ``chw`` spatial dims)."""
    x1, y1, x2, y2 = box.detach().cpu().tolist()
    x1i = max(int(x1), 0)
    y1i = max(int(y1), 0)
    x2i = min(int(x2), chw.shape[2])
    y2i = min(int(y2), chw.shape[1])
    if x2i <= x1i or y2i <= y1i:
        return chw
    return chw[:, y1i:y2i, x1i:x2i].contiguous()


def run_localization(
    model: Module,
    image: Image,
    *,
    device: torch.device,
    score_thresh: float,
    class_names: list[str] | None = None,
    sample_id_suffix: str = "_crop",
) -> LocalizationPrediction:
    model.eval()
    inp = _tensor_for_detection(image, device)
    with torch.no_grad():
        preds = model([inp])
    pred0 = preds[0]
    boxes = pred0["boxes"]
    scores = pred0["scores"]
    labels = pred0["labels"]
    det = {"boxes": boxes, "scores": scores, "labels": labels}
    used_fallback = False
    if boxes.numel() == 0:
        crop_tensor = inp.detach().cpu()
        used_fallback = True
        logger.warning(
            "Localization: empty predictions; using full image for downstream step"
        )
    else:
        keep = scores >= score_thresh
        if not bool(keep.any()):
            crop_tensor = inp.detach().cpu()
            used_fallback = True
            logger.warning(
                "Localization: no boxes above score_thresh=%s; using full image",
                score_thresh,
            )
        else:
            valid_idx = torch.where(keep)[0]
            best_sub = int(torch.argmax(scores[valid_idx]).item())
            idx = int(valid_idx[best_sub].item())
            crop_tensor = _crop_xyxy(inp, boxes[idx]).detach().cpu()
    sid = image.sample_id or "sample"
    crop_image = Image(
        crop_tensor, sample_id=f"{sid}{sample_id_suffix}", legend=image.legend
    )
    return LocalizationPrediction(
        detections=det,
        crop=crop_image,
        used_full_image_fallback=used_fallback,
        source_image=image,
        class_names=class_names,
        score_thresh=score_thresh,
    )


def run_classification(
    model: Module,
    image: Image,
    artifact: StepArtifact,
    *,
    device: torch.device,
) -> ClassificationPrediction:
    meta = artifact.inference_meta
    cc = meta.get("canonical_in_channels")
    if cc is None:
        msg = "classification inference_meta missing canonical_in_channels"
        raise ValueError(msg)
    tf = default_classification_transform(canonical_channels=int(cc))
    x = image.body
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    else:
        x = x.float()
        if x.max() > 1.5:
            x = x.div_(255.0)
    x = tf(x).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)[0]
    probs = logits.softmax(dim=-1)
    class_index = int(torch.argmax(probs).item())
    names = artifact.class_names
    class_name = names[class_index] if names and 0 <= class_index < len(names) else None
    return ClassificationPrediction(
        logits=logits,
        probs=probs,
        class_index=class_index,
        class_name=class_name,
    )


def run_segmentation(
    model: Module,
    image: Image,
    artifact: StepArtifact,
    *,
    device: torch.device,
    score_thresh: float,
) -> SegmentationPrediction:
    meta = artifact.inference_meta
    kind = meta.get("kind")
    if kind == "instance":
        inp = _tensor_for_detection(image, device)
        model.eval()
        with torch.no_grad():
            pred = model([inp])[0]
        return SegmentationPrediction(
            kind="instance",
            masks=pred["masks"],
            boxes=pred.get("boxes"),
            labels=pred.get("labels"),
            scores=pred.get("scores"),
            source_image=image,
            class_names=artifact.class_names,
            score_thresh=score_thresh,
        )

    if kind != "semantic":
        msg = f"Unsupported segmentation kind {kind!r}"
        raise ValueError(msg)

    input_channels = int(meta.get("input_channels", 3))
    image_size = meta.get("image_size")
    original_size = tuple(int(x) for x in image.body.shape[-2:])
    x = image.body
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if x.numel() > 0 and x.max() > 1.5:
            x = x / 255.0
    x = _ensure_channel_count(x, input_channels)
    if image_size is not None:
        x = F.interpolate(
            x.unsqueeze(0),
            size=tuple(image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    x = _normalize_tensor_imagenet_style(x).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)
    logits = output["out"] if isinstance(output, dict) else output
    if logits.shape[-2:] != original_size:
        logits = F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
    probs = logits.softmax(dim=1)[0]
    class_map = probs.argmax(dim=0)
    return SegmentationPrediction(
        kind="semantic",
        masks=class_map,
        source_image=image,
        class_names=artifact.class_names,
    )


def image_from_step_inputs(inputs: tuple[Any, ...]) -> Image:
    if not inputs:
        msg = "Step has no inputs"
        raise ValueError(msg)
    head = inputs[0]
    if isinstance(head, LocalizationPrediction):
        return head.crop
    if isinstance(head, Image):
        return head
    msg = f"Cannot resolve Image from pipeline inputs ({type(head).__name__})"
    raise TypeError(msg)


def run_step_inference(
    step_name: str,
    step: Step,
    inputs: tuple[Any, ...],
    artifact: StepArtifact,
    *,
    device: torch.device,
    score_thresh: float,
    models_cache: dict[str, Module],
    step_base_models: dict[str, Module] | None,
) -> LocalizationPrediction | ClassificationPrediction | SegmentationPrediction:
    if step.input_prepare is not None:
        inputs = step.input_prepare(inputs)
    if step_name not in models_cache:
        models_cache[step_name] = build_and_load_step_model(
            step_name,
            artifact,
            step.action,
            device=device,
            step_base_models=step_base_models,
        )
    model = models_cache[step_name]

    if isinstance(step.action, ImageLocalizer):
        base_image = inputs[0] if inputs else None
        if not isinstance(base_image, Image):
            msg = "ImageLocalizer expects an Image from __input__"
            raise TypeError(msg)
        return run_localization(
            model,
            base_image,
            device=device,
            score_thresh=score_thresh,
            class_names=artifact.class_names,
        )

    if isinstance(step.action, ImageClassifier):
        img = image_from_step_inputs(inputs)
        return run_classification(model, img, artifact, device=device)

    if isinstance(step.action, ImageSegmentator):
        img = image_from_step_inputs(inputs)
        return run_segmentation(
            model,
            img,
            artifact,
            device=device,
            score_thresh=score_thresh,
        )

    msg = f"Unsupported step action {type(step.action).__name__} for inference"
    raise TypeError(msg)
