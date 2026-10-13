from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TypedMessagePassing(nn.Module):
    """Nine typed directed relations aggregate learned-pruned evidence features."""

    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.transforms = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(9))
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, adjacency, node_types, node_mask):
        message = torch.zeros_like(hidden)
        denominator = adjacency.abs().sum(-1, keepdim=True).clamp_min(1e-8)
        for source in range(3):
            for target in range(3):
                typed = adjacency * ((node_types[:, :, None] == source)
                                     & (node_types[:, None, :] == target)).to(adjacency.dtype)
                message = message + torch.bmm(typed / denominator,
                                              self.transforms[source * 3 + target](hidden))
        updated = self.norm(hidden + self.dropout(F.gelu(self.self_projection(hidden) + message)))
        return updated * node_mask.unsqueeze(-1)


class ClaimEvidenceReadout(nn.Module):
    def __init__(self, hidden_dim, classes=3, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, classes))

    def forward(self, hidden, adjacency, node_types, node_mask):
        claim = hidden[:, 0]
        evidence_mask = node_mask & (node_types != 0)
        pooled = (hidden * evidence_mask.unsqueeze(-1)).sum(1)
        pooled = pooled / evidence_mask.sum(1, keepdim=True).clamp_min(1)
        weights = adjacency[:, 0] * evidence_mask
        neighbor = torch.bmm(weights[:, None], hidden).squeeze(1)
        neighbor = neighbor / weights.abs().sum(1, keepdim=True).clamp_min(1e-8)
        return self.classifier(torch.cat((claim, pooled, neighbor, claim * neighbor), dim=-1))
