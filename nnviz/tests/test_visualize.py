"""Tests for visualization utilities."""
from __future__ import annotations

from pathlib import Path

import torch

from src.visualize.distributions import plot_activation_hist
from src.visualize.feature_maps import plot_feature_maps
from src.visualize.neuron_grid import plot_neuron_grid


def test_plot_feature_maps(tmp_path: Path) -> None:
    tensor = torch.randn(1, 4, 8, 8)
    path = tmp_path / "feature_maps.png"
    result = plot_feature_maps(tensor, ncols=2, save_path=str(path))
    assert path.exists()
    assert result == str(path)


def test_plot_activation_hist(tmp_path: Path) -> None:
    tensor = torch.randn(1, 4, 4, 4)
    path = tmp_path / "hist.png"
    result = plot_activation_hist("layer", tensor, str(path))
    assert path.exists()
    assert result == str(path)


def test_plot_neuron_grid(tmp_path: Path) -> None:
    weights = torch.randn(4, 4)
    bias = torch.randn(4)
    path = tmp_path / "grid.png"
    result = plot_neuron_grid(weights, bias, str(path))
    assert path.exists()
    assert result == str(path)
