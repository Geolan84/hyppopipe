"""Serialize and restore torchvision model factories for training and inference.

Specs describe how to instantiate a compatible model shell before loading a
``state_dict``, including optional ``WeightsEnum`` FQNs for correct architecture.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import importlib

from torch.nn import Module


def _resolve_torchvision_weights_enum(fqn: str) -> Any:
    """Resolve a ``torchvision`` ``WeightsEnum`` member from a fully qualified name.

    Args:
        fqn: Dotted path such as
            ``torchvision.models.ResNet50_Weights.DEFAULT``.

    Returns:
        The corresponding weights enum member.

    Raises:
        ValueError: If ``fqn`` does not have at least module, enum class, and member.
    """
    parts = fqn.split(".")
    if len(parts) < 3:
        msg = f"Invalid weights_enum FQN: {fqn!r}"
        raise ValueError(msg)
    member = parts[-1]
    enum_cls_name = parts[-2]
    mod_path = ".".join(parts[:-2])
    mod = importlib.import_module(mod_path)
    enum_cls = getattr(mod, enum_cls_name)
    return getattr(enum_cls, member)


def model_spec_from_module(module: Module) -> dict[str, Any]:
    """Build a minimal spec for an already-instantiated ``Module``.

    Args:
        module: Trained or untrained model instance.

    Returns:
        Spec with ``kind='instantiated'`` and import path for the class.
    """
    cls = module.__class__
    return {
        "kind": "instantiated",
        "class_module": cls.__module__,
        "class_qualname": cls.__qualname__,
    }


def instantiate_base_from_spec(spec: Mapping[str, Any]) -> Module:
    """Instantiate a model shell compatible with a saved ``state_dict``.

    Torchvision ``ssdlite320_mobilenet_v3_large`` changes backbone width depending on
    ``weights`` / ``weights_backbone`` (see ``reduced_tail`` in torchvision). Training with
    ``weights=COCO_V1`` matches ``weights=None, weights_backbone=None``, not bare
    ``weights=None`` (which defaults to ImageNet backbone and wider channels).

    When ``spec`` contains ``weights_enum`` (as from ``ModelCandidate``), the factory is
    called with ``weights=...`` so the architecture matches training (e.g. DeepLab with
    ``aux_classifier`` when using ``DeepLabV3_ResNet50_Weights.DEFAULT``).

    Args:
        spec: Serialized factory descriptor (``kind='torchvision_factory'``).

    Returns:
        Fresh model instance; weights are not loaded.

    Raises:
        ValueError: If ``spec`` is not a supported torchvision factory descriptor.
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

    weights_fqn = spec.get("weights_enum")
    if isinstance(weights_fqn, str) and weights_fqn.strip():
        weights = _resolve_torchvision_weights_enum(weights_fqn)
        try:
            return factory(weights=weights)
        except TypeError:
            pass

    try:
        return factory(weights=None)
    except TypeError:
        return factory()
