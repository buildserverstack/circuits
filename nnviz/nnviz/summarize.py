"""Implement functions exactly as declared in this spec. Prefer readable code and explicit shapes. Add device/precision guards. Every plotting function must return a figure object and also save a `.png` to the current run folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .io import RunContext


def build_circuit_graph(
    nodes: Iterable[Tuple[str, float]],
    edges: Iterable[Tuple[str, str, float]],
) -> nx.DiGraph:
    """Construct a directed graph from nodes and edges."""

    graph = nx.DiGraph()
    for node_id, score in nodes:
        graph.add_node(node_id, score=float(score))
    for src, dst, weight in edges:
        graph.add_edge(src, dst, weight=float(weight))
    return graph


def plot_circuit_graph(
    graph: nx.DiGraph,
    run: RunContext,
    filename: str = "circuit_graph.png",
) -> plt.Figure:
    """Visualize the circuit graph and save artifacts."""

    pos = nx.spring_layout(graph, seed=0)
    scores = np.array([graph.nodes[n].get("score", 0.0) for n in graph.nodes])
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_nodes(graph, pos, node_size=600, node_color=scores, cmap="viridis", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=10, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    ax.set_axis_off()
    fig.tight_layout()
    path = Path(run.run_dir) / filename
    fig.savefig(path)
    run.register_artifact(path)
    json_path = Path(run.run_dir) / "circuit_graph.json"
    graph_json = nx.readwrite.json_graph.node_link_data(graph)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(graph_json, fh, indent=2)
    run.register_artifact(json_path)
    return fig


__all__ = ["build_circuit_graph", "plot_circuit_graph"]
