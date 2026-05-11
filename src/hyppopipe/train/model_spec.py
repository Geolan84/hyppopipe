from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import importlib

from torch.nn import Module


def model_spec_from_module(module: Module) -> dict[str, Any]:
    cls = module.__class__
    return {
        "kind": "instantiated",
        "class_module": cls.__module__,
        "class_qualname": cls.__qualname__,
    }


def instantiate_base_from_spec(spec: Mapping[str, Any]) -> Module:
    """Build a model shell compatible with the saved ``state_dict`` (then ``load_state_dict`` fills tensors).

    Torchvision ``ssdlite320_mobilenet_v3_large`` changes backbone width depending on
    ``weights`` / ``weights_backbone`` (see ``reduced_tail`` in torchvision). Training with
    ``weights=COCO_V1`` matches ``weights=None, weights_backbone=None``, not bare
    ``weights=None`` (which defaults to ImageNet backbone and wider channels).
    """
    kind = spec.get("kind")
    if kind != "torchvision_factory":
        msg = (
            f"Cannot instantiate base model from spec kind {kind!r}; "
            "use ModelCandidate(torchvision_fn, weights=[...]) for exportable pipelines, "
            "or pass ``step_base_models`` into Pipeline.predict for raw Module training."
        )
        raise ValueError(msg)
    factory_fqn = spec.get("factory")
    if not isinstance(factory_fqn, str):
        msg = "model_spec missing factory"
        raise ValueError(msg)
    mod_name, _, func_name = factory_fqn.rpartition(".")
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, func_name)

    if func_name == "ssdlite320_mobilenet_v3_large":
        return factory(weights=None, weights_backbone=None)

    try:
        return factory(weights=None)
    except TypeError:
        return factory()
