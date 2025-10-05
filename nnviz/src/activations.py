"""Activation analysis utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from .hooks import ActivationStore


def compute_sparsity(tensor: torch.Tensor) -> float:
    """Return fraction of zero activations."""
    flat = tensor.detach().cpu()
    zeros = torch.count_nonzero(flat == 0).item()
    total = flat.numel()
    return float(zeros) / float(total) if total > 0 else 0.0


def summarize_store(store: ActivationStore, layers: Iterable[str], kind: str = "post") -> Dict[str, Dict[str, float]]:
    """Compute summary statistics per layer."""
    summary: Dict[str, Dict[str, float]] = {}
    for layer in layers:
        activations = store.get(layer, kind=kind)
        if not activations:
            continue
        merged = torch.cat([a.flatten() for a in activations if isinstance(a, torch.Tensor)])
        summary[layer] = {
            "mean": float(merged.mean().item()),
            "std": float(merged.std(unbiased=False).item()),
            "sparsity": float(compute_sparsity(merged)),
        }
    return summary


def collect_activation_batches(store: ActivationStore, layer: str, kind: str = "post") -> torch.Tensor | None:
    batches = store.get(layer, kind=kind)
    if not batches:
        return None
    try:
        return torch.cat(batches, dim=0)
    except Exception:
        return batches[0]


__all__ = ["compute_sparsity", "summarize_store", "collect_activation_batches"]
