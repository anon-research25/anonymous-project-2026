from __future__ import annotations

import torch
from torch import nn

from .similarity import MultiHeadWeightedCosine, cosine_adjacency


class SemanticGraphBuilder(nn.Module):
    """Create E_s and E_c in CLIP's shared space, retaining all original nodes."""

    def __init__(self, dimension, epsilon=0.2, metric="cosine", heads=4):
        super().__init__()
        self.epsilon, self.metric = epsilon, metric
        self.weighted_cosine = MultiHeadWeightedCosine(dimension, heads) if metric == "weighted_cosine" else None

    def forward(self, features, node_types, node_mask):
        similarity = (self.weighted_cosine(features) if self.weighted_cosine is not None
                      else cosine_adjacency(features))
        count = features.size(1)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        pair_mask &= ~torch.eye(count, dtype=torch.bool, device=features.device)[None]
        claim_pair = (node_types[:, :, None] == 0) | (node_types[:, None, :] == 0)
        evidence_pair = (node_types[:, :, None] != 0) & (node_types[:, None, :] != 0)
        active = pair_mask & (similarity >= self.epsilon)
        semantic = torch.where(active & claim_pair, similarity, 0.0)
        cross_evidence = torch.where(active & evidence_pair, similarity, 0.0)
        # Keep an explicit Boolean mask: epsilon=0 can retain zero-weight edges.
        return {"similarity": similarity, "claim_evidence": semantic,
                "evidence_evidence": cross_evidence,
                "adjacency": semantic + cross_evidence, "edge_mask": active}
