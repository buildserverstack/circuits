"""Implement functions exactly as declared in this spec. Prefer readable code and explicit shapes. Add device/precision guards. Every plotting function must return a figure object and also save a `.png` to the current run folder."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from .config import NNVizConfig
from .hooks import RunCache


def compute_activation_stats(
    tensors: Iterable[torch.Tensor],
    threshold: float,
) -> Dict[str, np.ndarray]:
    """Compute summary statistics for a collection of tensors."""

    stats: Dict[str, List[np.ndarray]] = {
        "mean": [],
        "std": [],
        "max": [],
        "activation_rate": [],
    }
    for tensor in tensors:
        flat = tensor.detach().cpu().view(tensor.size(0), -1)
        mean = flat.mean(dim=1).numpy()
        std = flat.std(dim=1, unbiased=False).numpy()
        max_val = flat.max(dim=1).values.numpy()
        activation_rate = (flat.abs() > threshold).float().mean(dim=1).numpy()
        stats["mean"].append(mean)
        stats["std"].append(std)
        stats["max"].append(max_val)
        stats["activation_rate"].append(activation_rate)
    return {k: np.concatenate(v, axis=0) if v else np.array([]) for k, v in stats.items()}


def activation_threshold(tensor: torch.Tensor, multiplier: float) -> float:
    """Compute the activation threshold based on tensor std."""

    std = tensor.detach().cpu().std().item()
    return multiplier * std if std > 0 else multiplier


def collect_layer_activations(cache: RunCache, layer: str) -> torch.Tensor:
    """Return the cached activations for the specified layer."""

    bucket = cache.get(layer)
    if "post" not in bucket:
        raise KeyError(f"No post-activation cached for layer '{layer}'")
    return bucket["post"]


def classify_activations(
    tensor: torch.Tensor,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices for activated and unactivated units."""

    flat = tensor.detach().cpu().view(tensor.size(0), -1)
    active_mask = flat.abs() > threshold
    active_indices = active_mask.nonzero(as_tuple=False).numpy()
    inactive_indices = (~active_mask).nonzero(as_tuple=False).numpy()
    return active_indices, inactive_indices


def summarize_cache(cache: RunCache, config: NNVizConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute activation statistics for every cached layer."""

    summary: Dict[str, Dict[str, np.ndarray]] = {}
    for layer, tensors in cache.storage.items():
        if "post" not in tensors:
            continue
        tensor = tensors["post"]
        thresh = activation_threshold(tensor, config.activation_threshold_multiplier)
        summary[layer] = compute_activation_stats([tensor], thresh)
    return summary


__all__ = [
    "compute_activation_stats",
    "activation_threshold",
    "collect_layer_activations",
    "classify_activations",
    "summarize_cache",
]
