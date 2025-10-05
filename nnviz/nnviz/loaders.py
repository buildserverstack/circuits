"""Implement functions exactly as declared in this spec. Prefer readable code and explicit shapes. Add device/precision guards. Every plotting function must return a figure object and also save a `.png` to the current run folder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import DEFAULT_MAX_IMAGE_SIDE, DEFAULT_MAX_TOKENS


TextBatch = Dict[str, torch.Tensor]


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a Hugging Face tokenizer for the provided model name."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token or tokenizer.unk_token
    return tokenizer


def prepare_text_batch(
    texts: Iterable[str],
    model_name: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    device: Optional[torch.device] = None,
) -> TextBatch:
    """Tokenize text inputs and return tensors truncated/padded to max_tokens."""

    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)
    encodings = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    if device is not None:
        encodings = {k: v.to(device) for k, v in encodings.items()}
    return encodings


def load_image(path: Path) -> Image.Image:
    """Load an image from disk."""

    with Image.open(path) as img:
        return img.convert("RGB")


def preprocess_image(
    image: Image.Image,
    max_side: int = DEFAULT_MAX_IMAGE_SIDE,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess an image and return (tensor, normalized_tensor)."""

    short, long = sorted(image.size)
    scale = min(max_side / short, 1.0)
    resize_size = (int(round(image.size[1] * scale)), int(round(image.size[0] * scale)))
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(min(resize_size)),
            transforms.ToTensor(),
        ]
    )
    tensor = transform_pipeline(image)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalized = normalize(tensor.clone())
    if device is not None:
        tensor = tensor.to(device)
        normalized = normalized.to(device)
    return tensor.unsqueeze(0), normalized.unsqueeze(0)


def batch_images_from_paths(
    paths: Iterable[Path],
    max_side: int = DEFAULT_MAX_IMAGE_SIDE,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load multiple images and stack them into a batch."""

    tensors: List[torch.Tensor] = []
    normalized: List[torch.Tensor] = []
    for path in paths:
        tensor, norm = preprocess_image(load_image(Path(path)), max_side=max_side, device=device)
        tensors.append(tensor)
        normalized.append(norm)
    return torch.cat(tensors, dim=0), torch.cat(normalized, dim=0)


__all__ = [
    "prepare_text_batch",
    "load_tokenizer",
    "load_image",
    "preprocess_image",
    "batch_images_from_paths",
]
