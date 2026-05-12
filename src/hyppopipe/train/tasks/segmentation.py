from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from hyppopipe.data.dataset.adapters import adapt_dataset_for_segmentation
from hyppopipe.data.dataset.errors import SegmentationDataUnsupportedError
from hyppopipe.data.dataset.splits import SplitData
from hyppopipe.data.image import Image
from hyppopipe.pipeline.image.segmentation import ImageSegmentator, SegmentationKind
from hyppopipe.train.config import TrainingConfig
from hyppopipe.train.tasks.base import TrainingTask
from hyppopipe.train.tasks.classification import (
    _ensure_channel_count,
    _normalize_tensor_imagenet_style,
)

logger = logging.getLogger(__name__)

SegmentationBackend = Literal["mask_rcnn", "semantic"]


def infer_segmentation_backend(model: Module) -> SegmentationBackend:
    """Определяет семейство модели по структуре ``nn.Module`` (без хардкода имён классов)."""
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is not None and getattr(roi_heads, "mask_predictor", None) is not None:
        return "mask_rcnn"
    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        for mod in classifier.modules():
            if isinstance(mod, nn.Conv2d):
                return "semantic"
    msg = (
        "Неподдерживаемая архитектура для сегментации: ожидается Mask R-CNN "
        "(``roi_heads.mask_predictor``) или semantic-сеть с ``classifier`` из Conv2d "
        "(FCN / DeepLab / LRASPP и т.п.)."
    )
    raise NotImplementedError(msg)


def resolve_segmentation_data_kind(
    *,
    user_kind: SegmentationKind,
    backend: SegmentationBackend,
) -> SegmentationKind:
    """Согласует формат датасета с возможностями модели."""
    if backend == "mask_rcnn":
        if user_kind == "semantic":
            raise ValueError(
                "Mask R-CNN обучается на instance-таргетах (boxes, labels, masks). "
                "Укажите ImageSegmentator(kind='instance') и датасет в instance-формате, "
                "либо используйте semantic-модель (например deeplabv3_resnet50) с "
                "kind='semantic'."
            )
        return "instance"
    if user_kind == "instance":
        logger.info(
            "Модель — semantic segmentation (FCN/DeepLab и т.д.); "
            "данные загружаются как semantic (class map), несмотря на kind='instance'."
        )
    return "semantic"


def segmentation_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def _as_tensor(raw: Any) -> Tensor:
    x = raw.body if isinstance(raw, Image) else raw
    if not isinstance(x, torch.Tensor):
        msg = "Expected torch.Tensor or hyppopipe.data.image.Image"
        raise TypeError(msg)
    return x


def _image_to_float_tensor(raw: Any, *, channels: int | None = None) -> Tensor:
    x = _as_tensor(raw)
    if x.ndim != 3:
        msg = f"Expected CHW image tensor, got shape {tuple(x.shape)}"
        raise ValueError(msg)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if x.numel() > 0 and x.max() > 1.5:
            x = x / 255.0
    if channels is not None:
        x = _ensure_channel_count(x, channels)
    return x


def _mask_to_class_map(raw: Any) -> Tensor:
    mask = raw.as_gray if isinstance(raw, Image) else raw
    if not isinstance(mask, torch.Tensor):
        msg = "Expected torch.Tensor or hyppopipe.data.image.Image mask"
        raise TypeError(msg)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        msg = f"Expected HW semantic mask tensor, got shape {tuple(mask.shape)}"
        raise ValueError(msg)
    out = mask.long()
    if out.numel() == 0:
        return out
    unique_values = {int(v) for v in out.unique().tolist()}
    if unique_values <= {0, 255}:
        return (out > 0).long()
    return out


def _resize_image_and_mask(
    image: Tensor,
    mask: Tensor,
    size: tuple[int, int] | None,
) -> tuple[Tensor, Tensor]:
    if size is None:
        return image, mask
    image = F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask = (
        F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=size,
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
    )
    return image, mask.long()


class _SemanticSegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        base: Dataset[Any],
        *,
        image_size: tuple[int, int] | None,
        input_channels: int,
        image_transform: Callable[[Tensor], Tensor] | None,
    ) -> None:
        self.base = base
        self.image_size = image_size
        self.input_channels = input_channels
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        raw_image, raw_mask = self.base[index]
        image = _image_to_float_tensor(raw_image, channels=self.input_channels)
        mask = _mask_to_class_map(raw_mask)
        image, mask = _resize_image_and_mask(image, mask, self.image_size)
        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = _normalize_tensor_imagenet_style(image)
        return image, mask


class _InstanceSegmentationDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    def __init__(
        self,
        base: Dataset[Any],
        image_transform: Callable[[Tensor], Tensor] | None,
    ) -> None:
        self.base = base
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        raw_image, target = self.base[index]
        image = _image_to_float_tensor(raw_image)
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image, target


def _scan_targets_for_num_classes(
    dataset: Dataset[Any], *, kind: SegmentationKind
) -> int:
    n = len(dataset)
    if n == 0:
        msg = "Cannot infer segmentation classes from an empty dataset"
        raise ValueError(msg)
    max_label = 0
    for i in range(min(n, 2048)):
        _, target = dataset[i]
        if kind == "instance":
            labels = target.get("labels")
            if labels is not None and labels.numel() > 0:
                max_label = max(max_label, int(labels.max().item()))
        else:
            mask = _mask_to_class_map(target)
            if mask.numel() > 0:
                max_label = max(max_label, int(mask.max().item()))
    return max_label + 1


def infer_segmentation_num_classes(
    dataset: Dataset[Any],
    *,
    kind: SegmentationKind,
) -> int:
    if hasattr(dataset, "classes"):
        classes = getattr(dataset, "classes")
        if isinstance(classes, list) and classes:
            return len(classes) + 1
    return _scan_targets_for_num_classes(dataset, kind=kind)


def _replace_conv2d(conv: nn.Conv2d, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        conv.in_channels,
        out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )


def _replace_last_conv2d(module: Module, out_channels: int) -> bool:
    if isinstance(module, nn.Sequential):
        for name, child in reversed(list(module.named_children())):
            if isinstance(child, nn.Conv2d):
                module[int(name)] = _replace_conv2d(child, out_channels)
                return True
            if _replace_last_conv2d(child, out_channels):
                return True
    return False


def adapt_instance_segmentation_model(model: Module, num_classes: int) -> Module:
    if not hasattr(model, "roi_heads") or not hasattr(
        model.roi_heads, "mask_predictor"
    ):
        msg = (
            "Instance segmentation supports torchvision Mask R-CNN-style models "
            "with roi_heads.mask_predictor."
        )
        raise NotImplementedError(msg)

    box_predictor = model.roi_heads.box_predictor
    in_features = box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    mask_predictor = model.roi_heads.mask_predictor
    in_features_mask = mask_predictor.conv5_mask.in_channels
    hidden_layer = mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    return model


def adapt_semantic_segmentation_model(model: Module, num_classes: int) -> Module:
    replaced = False
    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        for attr in ("high_classifier", "low_classifier"):
            child = getattr(classifier, attr, None)
            if isinstance(child, nn.Conv2d):
                setattr(classifier, attr, _replace_conv2d(child, num_classes))
                replaced = True
        replaced = _replace_last_conv2d(classifier, num_classes) or replaced

    aux_classifier = getattr(model, "aux_classifier", None)
    if aux_classifier is not None:
        replaced = _replace_last_conv2d(aux_classifier, num_classes) or replaced

    if not replaced:
        msg = (
            "Semantic segmentation supports torchvision FCN/DeepLab/LRASPP-style "
            "models with classifier Conv2d heads."
        )
        raise NotImplementedError(msg)
    return model


def prepare_segmentation_model_from_meta(
    model: Module,
    *,
    kind: SegmentationKind,
    num_classes: int,
) -> Module:
    if kind == "instance":
        return adapt_instance_segmentation_model(model, num_classes)
    return adapt_semantic_segmentation_model(model, num_classes)


def prepare_segmentation_model(
    model: Module,
    train_dataset: Dataset[Any],
    segmentator: ImageSegmentator,
    *,
    backend: SegmentationBackend,
    effective_kind: SegmentationKind,
) -> tuple[Module, int]:
    num_classes = (
        segmentator.num_classes
        if segmentator.num_classes is not None
        else infer_segmentation_num_classes(train_dataset, kind=effective_kind)
    )
    meta_kind: SegmentationKind = "instance" if backend == "mask_rcnn" else "semantic"
    prepared = prepare_segmentation_model_from_meta(
        model,
        kind=meta_kind,
        num_classes=num_classes,
    )
    return prepared, num_classes


def _segmentation_core_splits(
    data: SplitData,
    kind: SegmentationKind,
) -> tuple[Dataset[Any], Dataset[Any]]:
    return (
        adapt_dataset_for_segmentation(data.train, kind=kind),
        adapt_dataset_for_segmentation(data.val, kind=kind),
    )


def segmentation_train_val_loaders(
    data: SplitData,
    config: TrainingConfig,
    segmentator: ImageSegmentator,
    *,
    data_kind: SegmentationKind,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_core, val_core = _segmentation_core_splits(data, data_kind)
    pin = torch.cuda.is_available()

    if data_kind == "instance":
        train_ds = _InstanceSegmentationDataset(train_core, segmentator.train_transform)
        val_ds = _InstanceSegmentationDataset(val_core, segmentator.val_transform)
        return (
            DataLoader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=pin,
                collate_fn=segmentation_collate_fn,
                drop_last=True,
            ),
            DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=pin,
                collate_fn=segmentation_collate_fn,
            ),
        )

    train_ds = _SemanticSegmentationDataset(
        train_core,
        image_size=segmentator.image_size,
        input_channels=segmentator.input_channels,
        image_transform=segmentator.train_transform,
    )
    val_ds = _SemanticSegmentationDataset(
        val_core,
        image_size=segmentator.image_size,
        input_channels=segmentator.input_channels,
        image_transform=segmentator.val_transform,
    )
    return (
        DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin,
            drop_last=True,
        ),
        DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin,
        ),
    )


def _semantic_loss(output: Any, target: Tensor, criterion: Module) -> Tensor:
    if isinstance(output, dict):
        logits = output["out"]
        aux = output.get("aux")
    else:
        logits = output
        aux = None

    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    loss = criterion(logits, target)
    if aux is not None:
        if aux.shape[-2:] != target.shape[-2:]:
            aux = F.interpolate(
                aux,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss = loss + 0.4 * criterion(aux, target)
    return loss


class SegmentationTrainingTask(TrainingTask):
    def __init__(self, segmentator: ImageSegmentator) -> None:
        self._segmentator = segmentator
        self._prepared_num_classes: int | None = None
        self._effective_kind: SegmentationKind | None = None

    def inference_meta_from_prepared(self, prepared: Module) -> dict[str, Any]:
        kind = self._effective_kind or self._segmentator.kind
        meta: dict[str, Any] = {
            "task": "segmentation",
            "kind": kind,
            "input_channels": self._segmentator.input_channels,
        }
        if self._prepared_num_classes is not None:
            meta["num_classes"] = self._prepared_num_classes
        if self._segmentator.image_size is not None:
            meta["image_size"] = list(self._segmentator.image_size)
        return meta

    def split_lengths(self, data: SplitData) -> tuple[int, int]:
        for kind in (self._segmentator.kind, self._fallback_kind_for_split()):
            try:
                train_core, val_core = _segmentation_core_splits(data, kind)
                return len(train_core), len(val_core)
            except SegmentationDataUnsupportedError:
                continue
        raise SegmentationDataUnsupportedError(
            "Не удалось определить размер сплита для сегментации; проверьте датасет "
            "и ImageSegmentator(kind='instance' | 'semantic')."
        )

    def _fallback_kind_for_split(self) -> SegmentationKind:
        return "semantic" if self._segmentator.kind == "instance" else "instance"

    def prepare(
        self,
        model: Module,
        data: SplitData,
        config: TrainingConfig,
    ) -> tuple[Module, DataLoader[Any], DataLoader[Any]]:
        backend = infer_segmentation_backend(model)
        effective_kind = resolve_segmentation_data_kind(
            user_kind=self._segmentator.kind,
            backend=backend,
        )
        self._effective_kind = effective_kind
        train_seg = adapt_dataset_for_segmentation(data.train, kind=effective_kind)
        prepared, num_classes = prepare_segmentation_model(
            model,
            train_seg,
            self._segmentator,
            backend=backend,
            effective_kind=effective_kind,
        )
        self._prepared_num_classes = num_classes
        train_ld, val_ld = segmentation_train_val_loaders(
            data,
            config,
            self._segmentator,
            data_kind=effective_kind,
        )
        return prepared, train_ld, val_ld

    def create_criterion(self, device: torch.device, config: TrainingConfig) -> Module:
        del config
        kind = self._effective_kind or self._segmentator.kind
        if kind == "instance":
            return torch.nn.Identity().to(device)
        return torch.nn.CrossEntropyLoss().to(device)

    def train_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> tuple[float, int]:
        kind = self._effective_kind or self._segmentator.kind
        if kind == "instance":
            return self._train_instance_batch(model, batch, optimizer, device)
        return self._train_semantic_batch(model, batch, criterion, optimizer, device)

    def eval_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        device: torch.device,
    ) -> tuple[float, int]:
        kind = self._effective_kind or self._segmentator.kind
        if kind == "instance":
            return self._eval_instance_batch(model, batch, device)
        return self._eval_semantic_batch(model, batch, criterion, device)

    def _train_instance_batch(
        self,
        model: Module,
        batch: Any,
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

    def _eval_instance_batch(
        self,
        model: Module,
        batch: Any,
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

    def _train_semantic_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> tuple[float, int]:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = _semantic_loss(output, masks, criterion)
        loss.backward()
        optimizer.step()
        n = images.size(0)
        return loss.detach().item() * n, n

    def _eval_semantic_batch(
        self,
        model: Module,
        batch: Any,
        criterion: Module,
        device: torch.device,
    ) -> tuple[float, int]:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        output = model(images)
        loss = _semantic_loss(output, masks, criterion)
        n = images.size(0)
        return loss.detach().item() * n, n
