"""Configuration constants for NNViz."""
from __future__ import annotations

from pathlib import Path

DEFAULT_OUT_DIR = Path("outputs")
DEFAULT_OUT_DIR.mkdir(exist_ok=True)

DEFAULT_DEVICE = "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/5/54/Golden_Retriever_medium-to-light-coat.jpg"
