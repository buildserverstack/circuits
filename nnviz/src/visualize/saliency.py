"""Saliency visualization."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..utils import ensure_out_dir, tensor_to_pil


def saliency_map(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int | None = None,
    save_path: str | None = None,
) -> np.ndarray:
    """Compute a gradient-based saliency map and optionally save overlay."""
    model.eval()
    working = image.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        output = model(working)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        score = output[:, target_class]
        model.zero_grad()
        score.backward(torch.ones_like(score))
        saliency = working.grad.abs().max(dim=1)[0]
        saliency -= saliency.min()
        saliency /= saliency.max().clamp(min=1e-6)
        saliency_np = saliency.cpu().numpy()

    if save_path is not None:
        ensure_out_dir(save_path)
        base = tensor_to_pil(image)
        overlay = saliency_np[0]
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.imshow(base)
        plt.imshow(overlay, cmap="hot", alpha=0.5)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return saliency_np


__all__ = ["saliency_map"]
