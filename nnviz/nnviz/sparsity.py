"""Compute activation rate using per-layer threshold based on std; return both indices and masks for active/inactive units."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .activations import activation_threshold
from .hooks import RunCache


def layer_sparsity(tensor: torch.Tensor, multiplier: float) -> Dict[str, np.ndarray]:
    """Return sparsity metrics for a tensor."""

    thresh = activation_threshold(tensor, multiplier)
    flat = tensor.detach().cpu().view(tensor.size(0), -1)
    active_mask = flat.abs() > thresh
    inactive_mask = ~active_mask
    sparsity = inactive_mask.float().mean(dim=1).numpy()
    return {
        "threshold": np.array([thresh]),
        "sparsity": sparsity,
        "active_mask": active_mask.numpy(),
        "inactive_mask": inactive_mask.numpy(),
    }


def cache_sparsity(cache: RunCache, multiplier: float) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute sparsity across cache."""

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for layer, tensors in cache.storage.items():
        if "post" not in tensors:
            continue
        results[layer] = layer_sparsity(tensors["post"], multiplier)
    return results


__all__ = ["layer_sparsity", "cache_sparsity"]
