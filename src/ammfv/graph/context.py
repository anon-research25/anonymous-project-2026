"""Allocate graph-selected references to independent evidence agents."""
from __future__ import annotations

from collections import deque

import numpy as np


def build_agent_context(evidence, graph_output, reference_size=7, max_agents=None, max_depth=3):
    """Select reachable original items by path strength, then bounded fallback.

    Pruning retains all nodes. Context selection is separately recorded. If no
    evidence is reachable (e.g. all edges below epsilon), fallback considers the
    original claim/evidence similarities; no synthetic factual record is added.
    """
    adjacency = graph_output["pruned"]["adjacency"][0].detach().float().cpu().numpy()
    edge_mask = graph_output["pruned"]["edge_mask"][0].detach().cpu().numpy()
    similarity = graph_output["graph"]["similarity"][0].detach().float().cpu().numpy()
    queue = deque([(0, 0, 1.0)])
    best = {0: 1.0}
    reachable = set()
    while queue:
        node, depth, strength = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in np.flatnonzero(edge_mask[node]):
            neighbor = int(neighbor)
            score = strength * max(abs(float(adjacency[node, neighbor])), 1e-8)
            if neighbor != 0:
                reachable.add(neighbor)
            if score > best.get(neighbor, -1):
                best[neighbor] = score
                queue.append((neighbor, depth + 1, score))
    ranked = sorted(reachable, key=lambda node: (-best[node], evidence[node - 1].evidence_id))
    fallback = not ranked and bool(evidence)
    if fallback:
        ranked = sorted(range(1, len(evidence) + 1),
                        key=lambda node: (-float(similarity[0, node]), evidence[node - 1].evidence_id))
    references = ranked[:reference_size]
    agent_nodes = references if max_agents is None else references[:max_agents]
    selected = [evidence[node - 1] for node in agent_nodes]
    trace = {"node_ids": ["CLAIM"] + [item.evidence_id for item in evidence],
             "initial_adjacency": graph_output["graph"]["adjacency"][0].detach().cpu().tolist(),
             "initial_edge_mask": graph_output["graph"]["edge_mask"][0].detach().cpu().tolist(),
             "pruned_adjacency": adjacency.tolist(), "pruned_edge_mask": edge_mask.tolist(),
             "neighbor_probabilities": graph_output["pruned"]["neighbor_probabilities"][0].detach().cpu().tolist(),
             "graph_probabilities": graph_output["probabilities"][0].detach().cpu().tolist(),
             "reachable_evidence_ids": [evidence[node - 1].evidence_id for node in ranked],
             "reference_ids": [evidence[node - 1].evidence_id for node in references],
             "agent_evidence_ids": [item.evidence_id for item in selected],
             "omitted_ids": [item.evidence_id for item in evidence if item not in selected],
             "fallback_used": fallback, "selection_rule": "path_strength_then_id"}
    return selected, [evidence[node - 1] for node in references], trace
