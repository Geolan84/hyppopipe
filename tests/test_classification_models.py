from __future__ import annotations

import torch
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    resnet18,
    vit_b_16,
)
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from hyppopipe.pipeline import Pipeline  # noqa: F401
from hyppopipe.train.tasks.classification_model import (
    adapt_classifier_backbone,
    classifier_output_features,
    prepare_classification_model_from_meta,
)
from hyppopipe.train.tasks.classification_transforms import (
    classification_transform_from_spec,
    transform_spec_from_weights,
)
from hyppopipe.train.tasks.detection import adapt_detection_model


def test_adapt_classifier_backbone_torchvision_heads() -> None:
    cases = [
        (resnet18(weights=None), ResNet18_Weights.DEFAULT),
        (densenet121(weights=None), DenseNet121_Weights.DEFAULT),
        (efficientnet_b0(weights=None), EfficientNet_B0_Weights.DEFAULT),
        (convnext_tiny(weights=None), ConvNeXt_Tiny_Weights.DEFAULT),
        (vit_b_16(weights=None), ViT_B_16_Weights.DEFAULT),
    ]
    for model, weights in cases:
        adapt_classifier_backbone(model, 7)
        assert classifier_output_features(model) == 7
        spec = transform_spec_from_weights(weights)
        tf = classification_transform_from_spec(spec, canonical_channels=3, train=False)
        x = torch.randint(0, 255, (3, 320, 400), dtype=torch.uint8)
        y = tf(x)
        assert y.shape == (3, spec["crop_size"], spec["crop_size"])


def test_prepare_classification_model_from_meta_roundtrip() -> None:
    base = resnet18(weights=ResNet18_Weights.DEFAULT)
    prepared = prepare_classification_model_from_meta(
        base, num_classes=4, canonical_in_channels=3
    )
    assert classifier_output_features(prepared) == 4


def test_fasterrcnn_head_adaptation() -> None:
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=91)
    adapted = adapt_detection_model(model, num_classes=5)
    assert int(adapted.roi_heads.box_predictor.cls_score.out_features) == 5
