from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from hyppopipe.data.dataset.adapters import adapt_dataset_for_detection
from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.pipeline.image.localization import ImageLocalizer
from hyppopipe.train.config import TrainingConfig
from hyppopipe.train.tasks.base import TrainingTask


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def infer_detection_num_classes(dataset: Dataset[Any]) -> int:
    """Число классов для ``torchvision`` detection (включая фон)."""
    if hasattr(dataset, "classes"):
        classes = getattr(dataset, "classes")
        if isinstance(classes, list) and classes:
            return len(classes) + 1
    msg = (
        "Detection dataset must expose non-empty ``classes`` "
        "(list of foreground class names)"
    )
    raise ValueError(msg)


def adapt_fasterrcnn_heads(model: Module, num_classes: int) -> Module:
    box_predictor = model.roi_heads.box_predictor
    in_features = box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _first_bn_norm_factory(module: Module) -> Callable[..., nn.Module]:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            return partial(nn.BatchNorm2d, eps=m.eps, momentum=m.momentum)
    return partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)


def _retinanet_norm_layer(head: Module) -> Callable[..., nn.Module] | None:
    cls_head = getattr(head, "classification_head", None)
    if cls_head is None:
        return None
    conv_seq = getattr(cls_head, "conv", None)
    if conv_seq is None:
        return None
    for m in conv_seq.modules():
        if isinstance(m, nn.BatchNorm2d):
            return partial(nn.BatchNorm2d, eps=m.eps, momentum=m.momentum)
        if isinstance(m, nn.GroupNorm):
            num_groups = m.num_groups
            eps = m.eps

            def _group_norm_factory(
                num_channels: int, *, ng: int = num_groups, e: float = eps
            ) -> nn.GroupNorm:
                return nn.GroupNorm(ng, num_channels, eps=e)

            return _group_norm_factory
    return None


def _fcos_num_convs(head: Module) -> int:
    cls_head = getattr(head, "classification_head", None)
    conv_seq = getattr(cls_head, "conv", None) if cls_head is not None else None
    if conv_seq is None:
        return 4
    return sum(1 for layer in conv_seq.children() if isinstance(layer, nn.Conv2d))


def adapt_torchvision_ssd_heads(model: Module, num_classes: int) -> Module:
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.ssd import SSD, SSDHead
    from torchvision.models.detection.ssdlite import SSDLiteHead

    if not isinstance(model, SSD):
        msg = "Internal error: expected torchvision SSD instance"
        raise TypeError(msg)
    fixed_size = model.transform.fixed_size
    if fixed_size is None:
        msg = "SSD transform must define fixed_size to rebuild detection heads"
        raise ValueError(msg)
    backbone = model.backbone
    out_channels = getattr(backbone, "out_channels", None)
    if out_channels is None:
        out_channels = det_utils.retrieve_out_channels(backbone, fixed_size)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    head_name = type(model.head).__name__
    if head_name == "SSDLiteHead":
        norm_layer = _first_bn_norm_factory(model.head)
        model.head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)
    elif head_name == "SSDHead":
        model.head = SSDHead(out_channels, num_anchors, num_classes)
    else:
        msg = (
            f"Unsupported SSD head {head_name!r}. Extend adapt_torchvision_ssd_heads "
            "for custom torchvision SSD variants."
        )
        raise NotImplementedError(msg)
    return model


def adapt_torchvision_retinanet_head(model: Module, num_classes: int) -> Module:
    from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead

    if not isinstance(model, RetinaNet):
        msg = "Internal error: expected torchvision RetinaNet instance"
        raise TypeError(msg)
    na = model.anchor_generator.num_anchors_per_location()
    num_anchors = na[0]
    in_channels = model.backbone.out_channels
    norm_layer = _retinanet_norm_layer(model.head)
    model.head = RetinaNetHead(
        in_channels, num_anchors, num_classes, norm_layer=norm_layer
    )
    return model


def adapt_torchvision_fcos_head(model: Module, num_classes: int) -> Module:
    from torchvision.models.detection.fcos import FCOS, FCOSHead

    if not isinstance(model, FCOS):
        msg = "Internal error: expected torchvision FCOS instance"
        raise TypeError(msg)
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    in_channels = model.backbone.out_channels
    num_convs = _fcos_num_convs(model.head)
    model.head = FCOSHead(in_channels, num_anchors, num_classes, num_convs=num_convs)
    return model


def adapt_detection_model(model: Module, num_classes: int) -> Module:
    """Подгоняет головы типичных моделей ``torchvision.detection`` под число классов."""
    if hasattr(model, "roi_heads"):
        return adapt_fasterrcnn_heads(model, num_classes)

    try:
        from torchvision.models.detection.ssd import SSD

        if isinstance(model, SSD):
            return adapt_torchvision_ssd_heads(model, num_classes)
    except ImportError:
        pass

    try:
        from torchvision.models.detection.retinanet import RetinaNet

        if isinstance(model, RetinaNet):
            return adapt_torchvision_retinanet_head(model, num_classes)
    except ImportError:
        pass

    try:
        from torchvision.models.detection.fcos import FCOS

        if isinstance(model, FCOS):
            return adapt_torchvision_fcos_head(model, num_classes)
    except ImportError:
        pass

    supported = (
        "Faster R-CNN / Mask R-CNN–style (roi_heads), SSD/SSDLite, RetinaNet, FCOS"
    )
    msg = (
        f"Unsupported detection model type {type(model)!r}. "
        f"Supported torchvision builders: {supported}."
    )
    raise NotImplementedError(msg)


def prepare_detection_model_from_meta(model: Module, *, num_classes: int) -> Module:
    """Rebuild detection heads using ``num_classes`` (including background)."""
    return adapt_detection_model(model, num_classes)


def prepare_detection_model(
    model: Module,
    train_dataset: Dataset[Any],
    localizer: ImageLocalizer,
) -> Module:
    n_cls = (
        localizer.num_classes
        if localizer.num_classes is not None
        else infer_detection_num_classes(train_dataset)
    )
    return adapt_detection_model(model, n_cls)


class _TransformedDetectionDataset(
    Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]
):
    def __init__(
        self,
        base: Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]],
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        self.base = base
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img, target = self.base[index]
        if self.image_transform is not None:
            img = self.image_transform(img)
        return img, target


def detection_train_val_loaders(
    data: SplitData,
    config: TrainingConfig,
    localizer: ImageLocalizer,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_core = adapt_dataset_for_detection(data.train)
    val_core = adapt_dataset_for_detection(data.val)
    train_ds = _TransformedDetectionDataset(train_core, localizer.train_transform)
    val_ds = _TransformedDetectionDataset(val_core, localizer.val_transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin,
        collate_fn=detection_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.resolve_val_batch_size(),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
        collate_fn=detection_collate_fn,
    )
    return train_loader, val_loader


class DetectionTrainingTask(TrainingTask):
    def __init__(self, localizer: ImageLocalizer) -> None:
        self._localizer = localizer

    def inference_meta_from_prepared(self, prepared: Module) -> dict[str, Any]:
        meta: dict[str, Any] = {"task": "detection"}
        if hasattr(prepared, "roi_heads"):
            cls_score = prepared.roi_heads.box_predictor.cls_score
            if hasattr(cls_score, "out_features"):
                meta["num_classes"] = int(cls_score.out_features)  # type: ignore[attr-defined]
                return meta
        cls_head = getattr(getattr(prepared, "head", None), "classification_head", None)
        num_columns = getattr(cls_head, "num_columns", None)
        if num_columns is not None:
            meta["num_classes"] = int(num_columns)
            return meta
        num_cls = getattr(cls_head, "num_classes", None)
        if num_cls is not None:
            meta["num_classes"] = int(num_cls)
            return meta
        return meta

    def split_lengths(self, data: SplitData) -> tuple[int, int]:
        return (
            len(adapt_dataset_for_detection(data.train)),
            len(adapt_dataset_for_detection(data.val)),
        )

    def prepare(
        self,
        model: Module,
        data: SplitData,
        config: TrainingConfig,
        *,
        weights_enum: Any | None = None,
    ) -> tuple[Module, DataLoader[Any], DataLoader[Any]]:
        train_det = adapt_dataset_for_detection(data.train)
        prepared = prepare_detection_model(model, train_det, self._localizer)
        train_ld, val_ld = detection_train_val_loaders(data, config, self._localizer)
        return prepared, train_ld, val_ld

    def create_criterion(self, device: torch.device, config: TrainingConfig) -> Module:
        return torch.nn.Identity().to(device)

    def train_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> tuple[float, int]:
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        losses.backward()
        optimizer.step()
        n = len(images)
        return losses.detach().item() * n, n

    def eval_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        device: torch.device,
    ) -> tuple[float, int]:
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
        n = len(images)
        return losses.detach().item() * n, n
