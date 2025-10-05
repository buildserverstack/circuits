"""Provide integrated_gradients(model, inputs, target_idx, steps=32); ensure baselines are zeros of same shape; return attributions normalized to [0,1]."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn


def vanilla_gradients(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    target_idx: int,
    loss_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute vanilla gradients of the target logit w.r.t. inputs."""

    model.zero_grad(set_to_none=True)
    prepared_inputs = inputs
    if isinstance(inputs, dict):
        prepared_inputs = {}
        for key, value in inputs.items():
            tensor = value
            if tensor.requires_grad is False:
                tensor = tensor.clone().detach().requires_grad_(True)
            prepared_inputs[key] = tensor
        outputs = model(**prepared_inputs)
    else:
        tensor = inputs
        if tensor.requires_grad is False:
            tensor = tensor.clone().detach().requires_grad_(True)
        prepared_inputs = tensor
        outputs = model(prepared_inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if loss_fn is None:
        target = logits[..., target_idx]
        loss = target.mean()
    else:
        loss = loss_fn(logits, target_idx)
    loss.backward()
    grads: Dict[str, torch.Tensor] = {}
    if isinstance(prepared_inputs, dict):
        for key, value in prepared_inputs.items():
            if value.requires_grad:
                grads[key] = value.grad.detach().clone()
    else:
        grads["inputs"] = prepared_inputs.grad.detach().clone()  # type: ignore[attr-defined]
    return grads


def _normalize(attrib: torch.Tensor) -> torch.Tensor:
    attrib = attrib.detach()
    attrib = attrib - attrib.min()
    denom = attrib.max().clamp(min=1e-6)
    return attrib / denom


def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    target_idx: int,
    steps: int = 32,
) -> torch.Tensor:
    """Integrated gradients for a tensor input."""

    model.eval()
    baseline = torch.zeros_like(inputs)
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(1, steps + 1)]
    grads: Optional[torch.Tensor] = None
    for scaled in scaled_inputs:
        scaled.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        output = model(scaled)
        logits = output.logits if hasattr(output, "logits") else output
        target = logits[..., target_idx]
        target.backward(torch.ones_like(target))
        grad = scaled.grad.detach()
        grads = grad if grads is None else grads + grad
    integrated = (inputs - baseline) * grads / steps
    return _normalize(integrated)


def grad_cam(
    activations: torch.Tensor,
    gradients: torch.Tensor,
) -> torch.Tensor:
    """Compute Grad-CAM heatmap."""

    weights = gradients.mean(dim=(-1, -2), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    return _normalize(cam)


def gradient_x_input(gradients: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Return Gradient x Input attribution."""

    return _normalize(gradients * inputs)


__all__ = [
    "vanilla_gradients",
    "integrated_gradients",
    "grad_cam",
    "gradient_x_input",
]
