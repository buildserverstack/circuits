"""Data loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, SAMPLE_IMAGE_URL
from .utils import download_image


def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_sample_image(path_or_url: str | None = None) -> torch.Tensor:
    """Load an image from disk or URL and return normalized tensor (C,H,W)."""
    transform = _default_transform()
    if path_or_url is None:
        image = download_image(SAMPLE_IMAGE_URL)
    else:
        p = Path(path_or_url)
        if p.exists():
            image = Image.open(p).convert("RGB")
        elif path_or_url.startswith("http"):
            image = download_image(path_or_url)
        else:
            raise FileNotFoundError(f"Cannot locate image: {path_or_url}")
    tensor = transform(image)
    return tensor


def get_cifar10(batch_size: int = 8, train: bool = False) -> DataLoader:
    """Return a CIFAR-10 DataLoader with ImageNet normalization."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = datasets.CIFAR10(root="data", train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
    return loader


__all__ = ["get_sample_image", "get_cifar10"]
