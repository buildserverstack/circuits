"""Visualize fully connected layer weights as a grid."""
from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import torch

from ..utils import ensure_out_dir


def plot_neuron_grid(weights: torch.Tensor, bias: torch.Tensor | None = None, save_path: str = "outputs/neuron_grid.png") -> str:
    """Plot each neuron's weight vector as a mini heatmap."""
    ensure_out_dir(save_path)
    if weights.dim() != 2:
        raise ValueError("weights must be 2D (out_features, in_features)")
    out_features, in_features = weights.shape
    side = int(in_features ** 0.5)
    fig, axes = plt.subplots(ceil(out_features / 8), 8, figsize=(16, ceil(out_features / 8) * 2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= out_features:
            continue
        vec = weights[idx].detach().cpu()
        if side * side == in_features:
            image = vec.view(side, side)
        else:
            image = vec.view(1, -1)
        im = ax.imshow(image, cmap="coolwarm")
        title = f"n{idx}"
        if bias is not None:
            title += f"\nbias={float(bias[idx]):.2f}"
        ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


__all__ = ["plot_neuron_grid"]
