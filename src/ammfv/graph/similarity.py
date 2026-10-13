from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadWeightedCosine(nn.Module):
    """M_s(q,k) = (1/z) sum_i cos(w_i * q, w_i * k), paper metric equation."""

    def __init__(self, dimension: int, heads: int = 4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(heads, dimension))
        nn.init.normal_(self.weights, mean=1.0, std=0.05)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weighted = features[:, None, :, :] * self.weights[None, :, None, :]
        weighted = F.normalize(weighted, dim=-1, eps=1e-8)
        return torch.einsum("bhid,bhjd->bhij", weighted, weighted).mean(1)


def cosine_adjacency(features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features, dim=-1, eps=1e-8)
    return normalized @ normalized.transpose(-1, -2)
