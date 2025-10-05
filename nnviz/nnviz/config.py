"""Implement functions exactly as declared in this spec. Prefer readable code and explicit shapes. Add device/precision guards. Every plotting function must return a figure object and also save a `.png` to the current run folder."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .devices import detect_device


SUPPORTED_VISION_MODELS = ["resnet18", "mobilenet_v3_small", "vit_tiny_patch16_224"]
SUPPORTED_TEXT_MODELS = ["distilbert-base-uncased", "gpt2"]
DEFAULT_MAX_TOKENS = 256
DEFAULT_MAX_IMAGE_SIDE = 384
DEFAULT_ACTIVATION_MULTIPLIER = 0.05
SUPPORTED_PRECISIONS = ("fp16", "bf16", "fp32")


@dataclass
class NNVizConfig:
    """Configuration for nnviz runs."""

    model_name: str = "distilbert-base-uncased"
    modality: str = "text"
    precision: str = "fp32"
    device: str = dataclasses.field(default_factory=lambda: detect_device()[0])
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE
    activation_threshold_multiplier: float = DEFAULT_ACTIVATION_MULTIPLIER
    topk_units: int = 20
    topk_heads: int = 4
    lite_mode: bool = False
    run_name: Optional[str] = None
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def ensure_supported(self) -> None:
        if self.modality == "vision" and self.model_name not in SUPPORTED_VISION_MODELS:
            raise ValueError(
                f"Unsupported vision model '{self.model_name}'. Supported: {SUPPORTED_VISION_MODELS}"
            )
        if self.modality == "text" and self.model_name not in SUPPORTED_TEXT_MODELS:
            raise ValueError(
                f"Unsupported text model '{self.model_name}'. Supported: {SUPPORTED_TEXT_MODELS}"
            )
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision '{self.precision}'. Supported: {SUPPORTED_PRECISIONS}"
            )


def get_default_config(modality: str = "text", model_name: Optional[str] = None) -> NNVizConfig:
    """Return a default configuration for the given modality."""

    if modality not in ("text", "vision"):
        raise ValueError("modality must be either 'text' or 'vision'")
    if model_name is None:
        model_name = (
            SUPPORTED_TEXT_MODELS[0] if modality == "text" else SUPPORTED_VISION_MODELS[0]
        )
    config = NNVizConfig(modality=modality, model_name=model_name)
    config.ensure_supported()
    if config.run_name is None:
        config.run_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return config


__all__ = [
    "NNVizConfig",
    "SUPPORTED_VISION_MODELS",
    "SUPPORTED_TEXT_MODELS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_IMAGE_SIDE",
    "SUPPORTED_PRECISIONS",
    "get_default_config",
]
