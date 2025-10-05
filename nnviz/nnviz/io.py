"""Create a timestamped run directory; save config, inputs, and all artifacts. Provide a load_run(path) helper to reopen previous runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from .config import NNVizConfig


class RunContext:
    """Handle run directories and artifact tracking."""

    def __init__(self, run_dir: Path, config: NNVizConfig) -> None:
        self.run_dir = str(run_dir)
        self.config = config
        self.artifacts: Dict[str, str] = {}

    def register_artifact(self, path: Path) -> None:
        self.artifacts[path.name] = str(path)

    def save_config(self) -> None:
        config_path = Path(self.run_dir) / "config.json"
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self.config), fh, indent=2)
        self.register_artifact(config_path)

    def save_tensor(self, name: str, tensor: torch.Tensor) -> None:
        path = Path(self.run_dir) / f"{name}.pt"
        torch.save(tensor.detach().cpu(), path)
        self.register_artifact(path)

    def to_dict(self) -> Dict[str, str]:
        return dict(self.artifacts)


def create_run_context(base_dir: Path, config: NNVizConfig) -> RunContext:
    """Create a run directory with timestamp."""

    run_name = config.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    context = RunContext(run_dir, config)
    context.save_config()
    return context


def load_run(path: Path) -> Dict[str, str]:
    """Load saved artifacts from a run directory."""

    manifest = {}
    for item in path.iterdir():
        manifest[item.name] = str(item)
    return manifest


__all__ = ["RunContext", "create_run_context", "load_run"]
