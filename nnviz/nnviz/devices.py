"""Implement functions exactly as declared in this spec. Prefer readable code and explicit shapes. Add device/precision guards. Every plotting function must return a figure object and also save a `.png` to the current run folder."""

from __future__ import annotations

from typing import Tuple

import torch


DEVICE_PRIORITIES = ("mps", "cuda", "cpu")


def detect_device() -> Tuple[str, torch.dtype]:
    """Detect the best available device and an appropriate default dtype."""

    for device_name in DEVICE_PRIORITIES:
        if device_name == "cuda" and torch.cuda.is_available():
            return device_name, torch.float32
        if device_name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return device_name, torch.float32
        if device_name == "cpu":
            return device_name, torch.float32
    return "cpu", torch.float32


def get_device_and_dtype(precision: str = "fp32") -> Tuple[torch.device, torch.dtype]:
    """Return the torch.device and dtype respecting precision and hardware support."""

    device_name, default_dtype = detect_device()
    device = torch.device(device_name)
    precision = precision.lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("precision must be one of {'fp16', 'bf16', 'fp32'}")
    if precision == "fp16":
        if device.type == "cpu":
            raise ValueError("fp16 precision is not supported on CPU")
        return device, torch.float16
    if precision == "bf16":
        return device, torch.bfloat16
    return device, default_dtype


__all__ = ["detect_device", "get_device_and_dtype", "DEVICE_PRIORITIES"]
