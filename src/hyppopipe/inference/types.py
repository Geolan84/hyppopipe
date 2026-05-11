from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from torch import Tensor

from hyppopipe.data.image import Image


_BACKGROUND_CLASS_NAMES = {"background", "__background__", "bg"}


def _image_to_display_array(image: Image) -> np.ndarray[Any, Any]:
    x = image.body.detach().cpu()
    if x.ndim == 2:
        return x.numpy()
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        x = x.permute(2, 0, 1)
    if x.shape[0] == 1:
        return x[0].numpy()
    return x[:3].permute(1, 2, 0).numpy()


def _resolve_class_name(label: int, class_names: list[str] | None) -> str:
    if not class_names:
        return f"class {label}"

    has_background = class_names[0].lower() in _BACKGROUND_CLASS_NAMES
    if has_background and 0 <= label < len(class_names):
        return class_names[label]
    if label == 0:
        return "background"
    if 1 <= label <= len(class_names):
        return class_names[label - 1]
    return f"class {label}"


@dataclass(slots=True)
class LocalizationPrediction:
    """Выход шага детекции: сырые предсказания и ROI для следующего шага."""

    detections: dict[str, Tensor]
    crop: Image
    used_full_image_fallback: bool
    source_image: Image | None = None
    class_names: list[str] | None = None
    score_thresh: float | None = None

    def show(
        self,
        image: Image | None = None,
        *,
        score_thresh: float | None = None,
        top_k: int | None = None,
        show_labels: bool = True,
        show_scores: bool = True,
        class_names: list[str] | None = None,
        color: str = "lime",
        linewidth: float = 2.0,
        figsize: tuple[float, float] | None = None,
        ax: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Показывает исходное изображение с bbox, классами и confidence."""
        image_to_show = image or self.source_image
        if image_to_show is None:
            msg = (
                "LocalizationPrediction.show() requires image=... "
                "when source image is absent"
            )
            raise ValueError(msg)

        boxes = self.detections.get("boxes")
        if boxes is None:
            msg = "detections must contain 'boxes'"
            raise ValueError(msg)

        scores = self.detections.get("scores")
        labels = self.detections.get("labels")
        boxes_cpu = boxes.detach().cpu()
        scores_cpu = scores.detach().cpu() if scores is not None else None
        labels_cpu = labels.detach().cpu() if labels is not None else None

        idxs = list(range(int(boxes_cpu.shape[0])))
        threshold = self.score_thresh if score_thresh is None else score_thresh
        if threshold is not None and scores_cpu is not None:
            idxs = [i for i in idxs if float(scores_cpu[i].item()) >= threshold]
        if top_k is not None:
            if scores_cpu is None:
                idxs = idxs[:top_k]
            else:
                idxs = sorted(
                    idxs, key=lambda i: float(scores_cpu[i].item()), reverse=True
                )[:top_k]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        display = _image_to_display_array(image_to_show)
        if display.ndim == 2:
            ax.imshow(display, cmap="gray")
        else:
            ax.imshow(display)

        names = class_names if class_names is not None else self.class_names
        for i in idxs:
            x1, y1, x2, y2 = [float(v) for v in boxes_cpu[i].tolist()]
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    width,
                    height,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                )
            )

            text_parts: list[str] = []
            if show_labels and labels_cpu is not None:
                text_parts.append(_resolve_class_name(int(labels_cpu[i].item()), names))
            if show_scores and scores_cpu is not None:
                text_parts.append(f"{float(scores_cpu[i].item()):.2f}")
            if text_parts:
                ax.text(
                    x1,
                    y1,
                    " ".join(text_parts),
                    color="black",
                    fontsize=9,
                    bbox={
                        "facecolor": color,
                        "alpha": 0.75,
                        "edgecolor": "none",
                        "pad": 1,
                    },
                )

        if image_to_show.sample_id:
            ax.set_title(image_to_show.sample_id)
        ax.axis("off")
        plt.show(**kwargs)


@dataclass(slots=True)
class ClassificationPrediction:
    """Выход классификации."""

    logits: Tensor
    probs: Tensor
    class_index: int
    class_name: str | None


@dataclass(slots=True)
class PipelinePrediction:
    """Результат ``Pipeline.predict``: значения по именам шагов."""

    outputs: dict[str, Any]
