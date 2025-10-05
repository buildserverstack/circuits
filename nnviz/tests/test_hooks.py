"""Tests for hook utilities."""
from __future__ import annotations

import torch
from torch import nn

from src.hooks import ActivationStore, GradientStore


def _build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def test_activation_store_captures_forward():
    model = _build_model()
    store = ActivationStore(store_pre=True, store_post=True, max_batches=1)
    layer_names = [name for name, _ in model.named_modules() if name]
    store.attach(model, layer_names)
    x = torch.randn(1, 4)
    model(x)
    for name in layer_names:
        assert store.get(name, "post"), f"Missing activations for {name}"
    store.clear()
    model(x)
    assert len(store.get(layer_names[0], "post")) == 1
    store.detach()


def test_gradient_store_captures_backward():
    model = _build_model()
    store = GradientStore()
    layer_names = [name for name, _ in model.named_modules() if name]
    store.attach(model, layer_names)
    x = torch.randn(1, 4, requires_grad=True)
    out = model(x)
    out.sum().backward()
    for name in layer_names:
        grads = store.get(name, "out_grad")
        assert grads, f"Missing gradients for {name}"
    store.detach()
