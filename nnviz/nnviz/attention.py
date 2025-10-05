"""Standardize attention to [heads, Q, K]. Provide safe head selection with bounds checks; raise ValueError with informative message."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .hooks import RunCache
from .io import RunContext


def extract_attention_map(attn_tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the tensor is shaped [heads, Q, K]."""

    if attn_tensor.dim() == 4:
        # [batch, heads, Q, K] -> select first batch for visualization
        return attn_tensor.detach().cpu()[0]
    if attn_tensor.dim() == 3:
        return attn_tensor.detach().cpu()
    raise ValueError("Attention tensor must have 3 or 4 dimensions")


def get_attention(cache: RunCache, layer: str) -> torch.Tensor:
    bucket = cache.get(layer)
    if "attention" not in bucket:
        raise KeyError(f"No attention data cached for layer '{layer}'")
    return extract_attention_map(bucket["attention"])


def select_head(attn: torch.Tensor, head_index: int) -> torch.Tensor:
    """Return attention map for the specified head with validation."""

    if head_index < 0 or head_index >= attn.size(0):
        raise ValueError(
            f"Head index {head_index} out of range for attention with {attn.size(0)} heads"
        )
    return attn[head_index]


def plot_attention(
    attn: torch.Tensor,
    head_index: int,
    tokens_q: Optional[np.ndarray] = None,
    tokens_k: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    run: Optional[RunContext] = None,
    filename: str = "attention.png",
    topk: Optional[int] = None,
) -> plt.Figure:
    """Render an attention heatmap and save it."""

    matrix = select_head(attn, head_index).numpy()
    if topk is not None and topk > 0:
        mask = np.zeros_like(matrix)
        flat_indices = np.argsort(matrix.flatten())[::-1][:topk]
        mask.flat[flat_indices] = matrix.flat[flat_indices]
        matrix = mask
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key Positions")
    ax.set_ylabel("Query Positions")
    if title:
        ax.set_title(title)
    if tokens_q is not None:
        ax.set_yticks(np.arange(len(tokens_q)))
        ax.set_yticklabels(tokens_q, fontsize=8)
    if tokens_k is not None:
        ax.set_xticks(np.arange(len(tokens_k)))
        ax.set_xticklabels(tokens_k, rotation=90, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if run is not None:
        path = Path(run.run_dir) / filename
        fig.savefig(path)
        run.register_artifact(path)
    return fig


__all__ = ["extract_attention_map", "get_attention", "select_head", "plot_attention"]
