"""Activation distribution plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..activations import compute_sparsity
from ..utils import ensure_out_dir


def plot_activation_hist(layer_name: str, activations: torch.Tensor, save_path: str) -> str:
    """Plot histogram and sparsity for activations."""
    ensure_out_dir(save_path)
    tensor = activations.detach().cpu()
    flat = tensor.flatten().numpy()
    sparsity = compute_sparsity(tensor)
    active = np.count_nonzero(flat > 0)
    inactive = np.count_nonzero(flat <= 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(flat, bins=100, color="#3366CC", alpha=0.8)
    axes[0].set_title(f"{layer_name}\nSparsity={sparsity:.2f}")

    axes[1].bar(["Active", "Inactive"], [active, inactive], color=["#33AA33", "#AA3333"])
    axes[1].set_title("Activation Counts")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


__all__ = ["plot_activation_hist"]
