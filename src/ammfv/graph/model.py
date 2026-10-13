from __future__ import annotations

import torch
from torch import nn

from .construction import SemanticGraphBuilder
from .layers import ClaimEvidenceReadout, TypedMessagePassing
from .pruning import GraphPruner


class HeterogeneousGraphVerifier(nn.Module):
    """Supervised graph probabilities are separate from the final LLM verdict."""

    def __init__(self, config):
        super().__init__()
        dimension, hidden = config["input_dim"], config["hidden_dim"]
        self.builder = SemanticGraphBuilder(dimension, config["epsilon"], config["metric"], config["heads"])
        self.pruner = GraphPruner(dimension, hidden, config["meta_paths"],
                                  config["neighbors"], config["temperature"])
        self.projection = nn.Linear(dimension, hidden)
        self.type_embedding = nn.Embedding(3, hidden)
        self.layers = nn.ModuleList(TypedMessagePassing(hidden, config["dropout"])
                                    for _ in range(config["layers"]))
        self.readout = ClaimEvidenceReadout(hidden, dropout=config["dropout"])

    def forward(self, features, node_types, node_mask):
        graph = self.builder(features.float(), node_types, node_mask)
        pruned = self.pruner(features.float(), node_types, graph)
        hidden = self.projection(features) + self.type_embedding(node_types.clamp_min(0))
        hidden = hidden * node_mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, pruned["adjacency"], node_types, node_mask)
        logits = self.readout(hidden, pruned["adjacency"], node_types, node_mask)
        # Keep the scorer in the backward graph even when thresholding leaves no edges.
        logits = logits + pruned["relevance_logits"].mean((1, 2)).unsqueeze(-1) * 0
        return {"logits": logits, "probabilities": logits.softmax(-1),
                "hidden": hidden, "graph": graph, "pruned": pruned}

    def set_sampling_temperature(self, temperature):
        if temperature <= 0:
            raise ValueError("Sampling temperature must be positive")
        self.pruner.sampler.temperature = temperature
