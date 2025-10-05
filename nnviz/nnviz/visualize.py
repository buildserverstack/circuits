"""No seaborn; use matplotlib or plotly. Include axis labels, titles, and a colorbar for heatmaps. Provide savefig(path) per plot."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .io import RunContext


def bar_plot(
    values: np.ndarray,
    title: str,
    ylabel: str,
    labels: Optional[Iterable[str]] = None,
    run: Optional[RunContext] = None,
    filename: str = "bar_plot.png",
) -> plt.Figure:
    """Create a bar plot with optional labels."""

    indices = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(indices, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Index")
    if labels is not None:
        ax.set_xticks(indices)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    if run is not None:
        path = Path(run.run_dir) / filename
        fig.savefig(path)
        run.register_artifact(path)
    return fig


def heatmap(
    matrix: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    run: Optional[RunContext] = None,
    filename: str = "heatmap.png",
) -> plt.Figure:
    """Render a heatmap with colorbar."""

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if run is not None:
        path = Path(run.run_dir) / filename
        fig.savefig(path)
        run.register_artifact(path)
    return fig


def layer_timeline(
    values: np.ndarray,
    layer_names: Iterable[str],
    run: Optional[RunContext] = None,
    filename: str = "layer_timeline.png",
) -> plt.Figure:
    """Render a timeline strip plot for layers."""

    fig, ax = plt.subplots(figsize=(10, 2))
    matrix = values[None, :]
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(list(layer_names), rotation=45, ha="right")
    ax.set_title("Layer Timeline")
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.2)
    fig.tight_layout()
    if run is not None:
        path = Path(run.run_dir) / filename
        fig.savefig(path)
        run.register_artifact(path)
    return fig


__all__ = ["bar_plot", "heatmap", "layer_timeline"]
