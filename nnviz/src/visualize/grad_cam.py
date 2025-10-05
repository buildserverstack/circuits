"""Grad-CAM visualization."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..utils import ensure_out_dir, tensor_to_pil


def grad_cam(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    layer_name: str,
    save_path: str | None = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a specific convolutional layer."""
    model.eval()
    activations = {}
    gradients = {}

    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer {layer_name} not found")
    layer = modules[layer_name]

    def forward_hook(mod, inp, out):
        activations["value"] = out.detach()

    def backward_hook(mod, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    working = image.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        output = model(working)
        score = output[:, target_class]
        model.zero_grad()
        score.backward(torch.ones_like(score), retain_graph=False)

    handle_f.remove()
    handle_b.remove()

    if "value" not in activations or "value" not in gradients:
        raise RuntimeError("Failed to capture activations/gradients for Grad-CAM")

    act = activations["value"].cpu()
    grad = gradients["value"].cpu()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().numpy()
    cam -= cam.min()
    cam /= cam.max().clip(min=1e-6)

    if save_path is not None:
        ensure_out_dir(save_path)
        base = tensor_to_pil(image)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.imshow(base)
        plt.imshow(cam, cmap="inferno", alpha=0.5)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return cam


__all__ = ["grad_cam"]
