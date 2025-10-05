"""Utility helpers for NNViz."""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from .config import DEFAULT_OUT_DIR, IMAGENET_MEAN, IMAGENET_STD

LOGGER = logging.getLogger("nnviz")


def ensure_out_dir(path: str | Path) -> Path:
    """Ensure the parent directory for a path exists and return the path."""
    p = Path(path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor in (C,H,W) normalized format to a PIL image."""
    tensor = tensor.detach().cpu().clone()
    if tensor.ndim == 4:
        tensor = tensor[0]
    for c, mean, std in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        c.mul_(std).add_(mean)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def download_image(url: str) -> Image.Image:
    """Download an image from a URL into a PIL image."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def save_json(data: dict, path: str | Path) -> Path:
    """Save a dictionary to JSON."""
    p = ensure_out_dir(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p


def iter_batches(iterable: Iterable[torch.Tensor], max_batches: int | None = None) -> Iterator[torch.Tensor]:
    """Yield up to ``max_batches`` batches from an iterable."""
    for idx, batch in enumerate(iterable):
        if max_batches is not None and idx >= max_batches:
            break
        yield batch


def close_figure(fig: plt.Figure) -> None:
    """Close a matplotlib figure to free memory."""
    plt.close(fig)


__all__ = [
    "DEFAULT_OUT_DIR",
    "ensure_out_dir",
    "tensor_to_pil",
    "download_image",
    "save_json",
    "iter_batches",
    "close_figure",
]
