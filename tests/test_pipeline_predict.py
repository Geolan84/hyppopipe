from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from matplotlib import pyplot as plt

from hyppopipe.data.image import Image
from hyppopipe.inference.types import LocalizationPrediction
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.pipeline.image.localization import ImageLocalizer
from hyppopipe.pipeline.image.transform import ImageTransformer
from hyppopipe.pipeline.pipeline import _topological_step_order
from hyppopipe.train.bundle import PredictBundle, export_train_result
from hyppopipe.train.result import ModelRunResult, StepTrainResult, TrainResult


def test_topological_step_order_linear_chain() -> None:
    steps = {
        "localize": Step(ImageLocalizer(), inputs={"__input__"}),
        "classify": Step(ImageClassifier(), inputs={"localize"}),
    }
    order = _topological_step_order(steps)
    assert order == ["localize", "classify"]


def test_topological_step_order_cycle_raises() -> None:
    steps = {
        "a": Step(ImageClassifier(), inputs={"b"}),
        "b": Step(ImageClassifier(), inputs={"a"}),
    }
    with pytest.raises(ValueError, match="cycle"):
        _topological_step_order(steps)


def test_topological_unknown_dependency_raises() -> None:
    steps = {
        "x": Step(ImageClassifier(), inputs={"missing"}),
    }
    from hyppopipe.pipeline.errors import MissingInputsError

    with pytest.raises(MissingInputsError):
        _topological_step_order(steps)


def test_predict_bundle_manifest_roundtrip(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    wpath = weights_dir / "step_a.pth"
    torch.save({"dummy": torch.zeros(1)}, wpath)
    manifest = {
        "version": 1,
        "steps": {
            "step_a": {
                "task": "classification",
                "weights": "weights/step_a.pth",
                "model_spec": {
                    "kind": "torchvision_factory",
                    "factory": "torchvision.models.resnet.resnet18",
                    "weights_enum": (
                        "torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1"
                    ),
                },
                "inference_meta": {
                    "task": "classification",
                    "num_classes": 10,
                    "canonical_in_channels": 3,
                },
                "class_names": None,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = PredictBundle.load(tmp_path)
    assert bundle.root == tmp_path.resolve()
    assert "step_a" in bundle.steps
    assert bundle.steps["step_a"].weights_path.exists()


def test_export_train_result_copies_weights(tmp_path: Path) -> None:
    ckpt = tmp_path / "orig.pth"
    torch.save({"layer.weight": torch.ones(2, 2)}, ckpt)
    run = ModelRunResult(
        model_label="resnet18_IMAGENET1K_V1",
        best_val_loss=0.01,
        epochs_ran=1,
        stopped_early=False,
        checkpoint_path=str(ckpt),
        model_spec={
            "kind": "torchvision_factory",
            "factory": "torchvision.models.resnet.resnet18",
            "weights_enum": (
                "torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1"
            ),
        },
        inference_meta={
            "task": "classification",
            "num_classes": 4,
            "canonical_in_channels": 3,
        },
    )
    tr = TrainResult(
        steps={
            "only": StepTrainResult(step_name="only", runs=[run]),
        }
    )
    pipe_steps = {
        "only": Step(ImageClassifier(num_classes=4), inputs={"__input__"}),
    }
    export_train_result(tmp_path / "out", tr, pipe_steps)
    out_dir = tmp_path / "out"
    assert (out_dir / "manifest.json").is_file()
    copied = out_dir / "weights" / "only.pth"
    assert copied.is_file()
    loaded = torch.load(copied, weights_only=True)
    assert "layer.weight" in loaded


def test_localization_prediction_show_draws_filtered_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image(torch.zeros(3, 10, 10), sample_id="sample")
    prediction = LocalizationPrediction(
        detections={
            "boxes": torch.tensor([[1.0, 2.0, 4.0, 6.0], [0.0, 0.0, 2.0, 2.0]]),
            "scores": torch.tensor([0.9, 0.1]),
            "labels": torch.tensor([1, 2]),
        },
        crop=image,
        used_full_image_fallback=False,
        source_image=image,
        class_names=["tumor", "other"],
        score_thresh=0.5,
    )
    monkeypatch.setattr(plt, "show", lambda **_: None)

    plt.close("all")
    prediction.show()

    ax = plt.gcf().axes[0]
    assert len(ax.patches) == 1
    assert ax.texts[0].get_text() == "tumor 0.90"
    plt.close("all")


def test_instantiate_deeplab_from_spec_restores_aux_classifier_via_weights_enum() -> (
    None
):
    """``weights=None`` даёт DeepLab без aux; при ``weights_enum`` как при обучении — та же топология."""
    from hyppopipe.train.model_spec import instantiate_base_from_spec

    spec = {
        "kind": "torchvision_factory",
        "factory": "torchvision.models.segmentation.deeplabv3.deeplabv3_resnet50",
        "weights_enum": (
            "torchvision.models.segmentation.deeplabv3.DeepLabV3_ResNet50_Weights.DEFAULT"
        ),
    }
    model = instantiate_base_from_spec(spec)

    assert model.aux_classifier is not None
    assert any(k.startswith("aux_classifier.") for k in model.state_dict())


def test_ssdlite_factory_matches_coco_trained_backbone_width() -> None:
    """``weights=None`` defaults to ImageNet backbone (wider); COCO-trained checkpoints need reduced tail."""
    from hyppopipe.train.model_spec import instantiate_base_from_spec

    spec = {
        "kind": "torchvision_factory",
        "factory": "torchvision.models.detection.ssdlite.ssdlite320_mobilenet_v3_large",
        "weights_enum": (
            "torchvision.models.detection.ssdlite.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1"
        ),
    }
    m = instantiate_base_from_spec(spec)
    first_conv_out = m.state_dict()["backbone.features.1.0.3.0.weight"].shape[0]
    assert first_conv_out == 80


def test_pipeline_ctor_keeps_dict_order_for_topo() -> None:
    pipe = Pipeline(
        {
            "first": Step(ImageLocalizer(), inputs={"__input__"}),
            "second": Step(ImageClassifier(), inputs={"first"}),
        }
    )
    assert _topological_step_order(pipe.steps) == ["first", "second"]


def test_shift_result_chains_previous_output() -> None:
    pipe = Pipeline(
        {
            "a": Step(ImageLocalizer(), inputs=None),
            "b": Step(ImageClassifier(), inputs=None),
        },
        shift_result=True,
    )
    ordered = _topological_step_order(pipe.steps)
    assert ordered == ["a", "b"]
    pipe.registry = {"__input__": "IMG", "a": "OUT_A"}
    assert pipe._predict_step_inputs(ordered, 0, "a", pipe.steps["a"]) == ("IMG",)
    assert pipe._predict_step_inputs(ordered, 1, "b", pipe.steps["b"]) == ("OUT_A",)


def test_shift_result_matches_explicit_linear_deps() -> None:
    """Для линейного DAG цепочка совпадает с явными edges."""
    pipe_shift = Pipeline(
        {
            "first": Step(ImageLocalizer(), inputs=("__input__",)),
            "second": Step(ImageClassifier(), inputs=("first",)),
        },
        shift_result=True,
    )
    pipe_explicit = Pipeline(
        {
            "first": Step(ImageLocalizer(), inputs=("__input__",)),
            "second": Step(ImageClassifier(), inputs=("first",)),
        },
        shift_result=False,
    )
    ordered = _topological_step_order(pipe_shift.steps)
    pipe_shift.registry = {"__input__": "IMG", "first": "LOC"}
    pipe_explicit.registry = {"__input__": "IMG", "first": "LOC"}
    for idx, name in enumerate(ordered):
        step = pipe_shift.steps[name]
        assert pipe_shift._predict_step_inputs(
            ordered, idx, name, step
        ) == pipe_explicit._get_inputs(name, step)


def test_predict_transform_only_without_train_result() -> None:
    image = Image(torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8))
    pipe = Pipeline(
        {
            "transform": Step(
                ImageTransformer().resize(16),
            ),
        }
    )
    result = pipe.predict(image)
    out = result.outputs["transform"]
    assert isinstance(out, Image)
    assert out.body.shape[-2:] == (16, 16)


def test_predict_transform_only_rejects_train_artifacts() -> None:
    pipe = Pipeline({"transform": Step(ImageTransformer().resize(8))})
    image = Image(torch.zeros(3, 4, 4))
    with pytest.raises(ValueError, match="no steps that require trained weights"):
        pipe.predict(image, train_result=TrainResult(steps={}))


def test_shift_result_false_uses_step_inputs_only() -> None:
    pipe = Pipeline(
        {"only": Step(ImageClassifier(num_classes=2), inputs=("__input__",))},
        shift_result=False,
    )
    pipe.registry = {"__input__": "IMG"}
    ordered = ["only"]
    assert pipe._predict_step_inputs(ordered, 0, "only", pipe.steps["only"]) == ("IMG",)
