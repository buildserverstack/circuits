"""Gradient analysis helpers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch

from .hooks import GradientStore


def summarize_gradients(store: GradientStore, layers: Iterable[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for layer in layers:
        grads = store.get(layer, kind="out_grad")
        if not grads:
            continue
        flat_values: List[torch.Tensor] = []
        for entry in grads:
            if isinstance(entry, torch.Tensor):
                flat_values.append(entry.flatten())
            elif isinstance(entry, (tuple, list)):
                for g in entry:
                    if isinstance(g, torch.Tensor):
                        flat_values.append(g.flatten())
        if not flat_values:
            continue
        merged = torch.cat(flat_values)
        summary[layer] = {
            "mean_abs_grad": float(merged.abs().mean().item()),
            "max_abs_grad": float(merged.abs().max().item()),
        }
    return summary


def last_param_grads(store: GradientStore, layer: str) -> List[Tuple[str, torch.Tensor]]:
    entries = store.get(layer, kind="param_grad")
    if not entries:
        return []
    return entries[-1:]


__all__ = ["summarize_gradients", "last_param_grads"]
