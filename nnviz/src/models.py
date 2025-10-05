"""Model loading utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torchvision import models


_SUPPORTED_MODELS = {
    "resnet18": models.resnet18,
    "mobilenet_v3_small": models.mobilenet_v3_small,
}


def load_model(name: str = "resnet18", pretrained: bool = True, num_classes: int | None = None) -> nn.Module:
    """Load a torchvision model configured for CPU inference."""
    name = name.lower()
    if name not in _SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {name}")
    constructor = _SUPPORTED_MODELS[name]
    weights = None
    if pretrained:
        try:
            weights = constructor(weights="IMAGENET1K_V1").weights
        except Exception:
            weights = "DEFAULT"
    model = constructor(weights=weights if pretrained else None)
    if num_classes is not None and hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    return model


def enumerate_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return an ordered list of named layers suitable for hooks."""
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        layers.append((name, module))
    return layers


__all__ = ["load_model", "enumerate_layers"]
