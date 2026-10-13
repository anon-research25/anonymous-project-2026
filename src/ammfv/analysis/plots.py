from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..utils.io import read_jsonl


def draw_graph(trace_path, output):
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    graph = trace["graph"]
    if "pruned_adjacency" not in graph:
        raise ValueError("Trace does not contain a learned graph")
    ids = graph["node_ids"]
    adjacency = np.asarray(graph["pruned_adjacency"])
    mask = np.asarray(graph["pruned_edge_mask"])
    modalities = {item["evidence_id"]: item["modality"] for item in trace["retrieval"]["evidence"]}
    network = nx.DiGraph()
    for index, identifier in enumerate(ids):
        network.add_node(index, label=identifier)
    for source, target in zip(*np.nonzero(mask)):
        network.add_edge(int(source), int(target), weight=float(adjacency[source, target]))
    colors = ["#f4a261" if index == 0 else "#2a9d8f" if modalities[identifier] == "text" else "#457b9d"
              for index, identifier in enumerate(ids)]
    position = nx.spring_layout(network, seed=42)
    figure, axis = plt.subplots(figsize=(12, 9))
    nx.draw_networkx(network, position, ax=axis, labels=dict(enumerate(ids)), node_color=colors,
                     node_size=1600, font_size=8, arrows=True)
    nx.draw_networkx_edge_labels(network, position, ax=axis,
                                 edge_labels={(i, j): f"{data['weight']:.2f}" for i, j, data in network.edges(data=True)},
                                 font_size=7)
    axis.set_title("AMMFV retained directed relations (all original nodes preserved)")
    axis.axis("off")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=200)
    plt.close(figure)


def plot_training(metrics_path, output):
    records = list(read_jsonl(metrics_path))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in ("train/loss", "validation/loss"):
        rows = [r for r in records if key in r]
        axes[0].plot([r["step"] for r in rows], [r[key] for r in rows], label=key)
    for key in ("validation/accuracy", "validation/macro_f1"):
        rows = [r for r in records if key in r]
        axes[1].plot([r["step"] for r in rows], [r[key] for r in rows], label=key)
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_sensitivity(rows, parameter, metric, output):
    figure, axis = plt.subplots(figsize=(7, 5))
    groups = sorted({row.get("evidence_mode", "retrieved") for row in rows})
    for group in groups:
        selected = sorted([row for row in rows if row.get("evidence_mode", "retrieved") == group],
                          key=lambda row: row[parameter])
        axis.plot([row[parameter] for row in selected], [row[metric] * 100 for row in selected],
                   marker="o", label=group)
    axis.set_xlabel(parameter)
    axis.set_ylabel(f"{metric} (%)")
    axis.legend()
    axis.grid(alpha=0.2)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=200)
    plt.close(figure)
