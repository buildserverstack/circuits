"""Visualize model architecture as a graph."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import nn

from ..models import enumerate_layers
from ..utils import ensure_out_dir


def draw_network_graph(model: nn.Module, save_path: str = "outputs/network_graph.png") -> str:
    """Draw a layer graph weighted by parameter counts."""
    ensure_out_dir(save_path)
    graph = nx.DiGraph()
    layers = enumerate_layers(model)
    prev = "input"
    graph.add_node(prev, label="Input")

    for name, module in layers:
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        label = f"{name}\n{module.__class__.__name__}\nparams={param_count}"
        graph.add_node(name, label=label, size=max(param_count, 1))
        graph.add_edge(prev, name, weight=max(param_count, 1))
        prev = name
    graph.add_node("output", label="Output")
    graph.add_edge(prev, "output", weight=1)

    edge_widths = [max(1.0, (data["weight"] ** 0.3)) for _, _, data in graph.edges(data=True)]
    pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=True,
        labels={node: data.get("label", node) for node, data in graph.nodes(data=True)},
        node_size=[300 + data.get("size", 1) * 0.1 for _, data in graph.nodes(data=True)],
        width=edge_widths,
        arrows=True,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(save_path)


__all__ = ["draw_network_graph"]
