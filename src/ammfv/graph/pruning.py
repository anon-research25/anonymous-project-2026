"""Gamma relevance and repeated straight-through Gumbel edge selection."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NeighborRelevance(nn.Module):
    """Gamma(h_i,h_j,E_f[i,j]) with node types and meta-path features."""

    def __init__(self, input_dim, hidden_dim=256, type_dim=16, path_count=1):
        super().__init__()
        self.type_embedding = nn.Embedding(3, type_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2 + 1 + type_dim * 2 + path_count, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, features, node_types, adjacency, path_features):
        batch, count, dimension = features.shape
        source = features[:, :, None, :].expand(batch, count, count, dimension)
        target = features[:, None, :, :].expand_as(source)
        types = self.type_embedding(node_types.clamp_min(0))
        source_type = types[:, :, None, :].expand(batch, count, count, types.size(-1))
        target_type = types[:, None, :, :].expand_as(source_type)
        pair = torch.cat((source, target, adjacency.unsqueeze(-1),
                          source_type, target_type, path_features), dim=-1)
        return self.network(pair).squeeze(-1)


class GumbelNeighborhoodSampler(nn.Module):
    """Sample at most t distinct outgoing edges per node, retaining node identities.

    Training iterates categorical Gumbel-softmax t times without replacement.
    Forward values are hard; backward gradients follow each soft draw. Inference
    is deterministic top-t. Directed selection
    is intentional: symmetrizing would break the per-node outgoing-edge bound.
    """

    def __init__(self, neighbors=4, temperature=1.0):
        super().__init__()
        self.neighbors, self.temperature = neighbors, temperature

    def forward(self, logits, edge_mask):
        available = edge_mask.clone()
        selected = torch.zeros_like(logits)
        probabilities = torch.zeros_like(logits)
        has_edges = available.any(-1)
        if has_edges.any():
            probabilities[has_edges] = logits[has_edges].masked_fill(
                ~available[has_edges], -torch.inf).softmax(-1)
        for _ in range(self.neighbors):
            active = available.any(-1)
            if not active.any():
                break
            candidate_logits = logits[active].masked_fill(~available[active], -torch.inf)
            if self.training:
                draw = F.gumbel_softmax(candidate_logits, tau=self.temperature, hard=True, dim=-1)
            else:
                draw = F.one_hot(candidate_logits.argmax(-1), num_classes=logits.size(-1)).to(logits.dtype)
            addition = torch.zeros_like(logits)
            addition[active] = draw
            selected = selected + addition
            available = available & ~(addition.detach() > 0.5)
        return selected, probabilities


class GraphPruner(nn.Module):
    def __init__(self, input_dim, hidden_dim, paths, neighbors=4, temperature=1.0):
        super().__init__()
        from .metapaths import MetaPathNeighborhoods
        self.neighborhoods = MetaPathNeighborhoods(paths)
        self.relevance = NeighborRelevance(input_dim, hidden_dim, path_count=max(len(paths), 1))
        self.sampler = GumbelNeighborhoodSampler(neighbors, temperature)

    def forward(self, features, node_types, graph):
        topology = self.neighborhoods.features(graph["edge_mask"], node_types)
        logits = self.relevance(features, node_types, graph["adjacency"], topology)
        selection, probabilities = self.sampler(logits, graph["edge_mask"])
        return {"adjacency": graph["adjacency"] * selection,
                "selection": selection, "edge_mask": selection.detach() > 0.5,
                "neighbor_probabilities": probabilities, "relevance_logits": logits}
