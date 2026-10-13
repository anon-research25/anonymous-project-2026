"""Typed relation composition, with no creation of new evidence records."""
from __future__ import annotations

import torch

from ..schema import NODE_TYPES


def relation_mask(edge_mask, node_types, source: str, target: str):
    return (edge_mask & (node_types[:, :, None] == NODE_TYPES[source])
            & (node_types[:, None, :] == NODE_TYPES[target]))


def meta_path_reachability(edge_mask, node_types, path: list[str]) -> torch.Tensor:
    """Boolean reachability for A1 -> A2 -> ... -> An; composition is batched."""
    if len(path) < 2:
        raise ValueError("A meta-path requires at least two node types")
    result = relation_mask(edge_mask, node_types, path[0], path[1])
    for source, target in zip(path[1:-1], path[2:]):
        following = relation_mask(edge_mask, node_types, source, target)
        result = torch.bmm(result.float(), following.float()) > 0
    return result


class MetaPathNeighborhoods:
    """Configured paths yield topology features for Gamma's neighbor scoring.

    Default paths include the one-hop claim/evidence and evidence/evidence
    relations. Multi-hop reachability is a scoring feature; sampling still
    selects original edges, so per-node degree is bounded by t in E_f.
    """

    def __init__(self, paths: list[list[str]]):
        self.paths = paths
        for path in paths:
            if any(name not in NODE_TYPES for name in path):
                raise ValueError(f"Invalid meta-path {path}")

    def features(self, edge_mask, node_types):
        if not self.paths:
            return edge_mask.float().unsqueeze(-1)
        return torch.stack([meta_path_reachability(edge_mask, node_types, path).float()
                            for path in self.paths], dim=-1)
