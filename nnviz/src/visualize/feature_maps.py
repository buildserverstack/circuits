"""Feature map plotting."""
from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch

from ..utils import ensure_out_dir


def plot_feature_maps(tensor: torch.Tensor, ncols: int = 8, save_path: str = "outputs/feature_maps.png") -> str:
    """Plot convolutional feature maps."""
    ensure_out_dir(save_path)
    if tensor.dim() == 4:
        tensor = tensor[0]
    channels = tensor.size(0)
    nrows = ceil(channels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = axes.flatten() if isinstance(axes, (list, tuple)) else axes.reshape(-1)
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= channels:
            continue
        fmap = tensor[idx].detach().cpu()
        ax.imshow(fmap, cmap="viridis")
        ax.set_title(f"C{idx}", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


__all__ = ["plot_feature_maps"]
