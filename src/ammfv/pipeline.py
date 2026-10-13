from __future__ import annotations

from pathlib import Path

import torch

from .agents.debate import DebateController
from .data.features import encode_claim_graph
from .graph.context import build_agent_context
from .schema import NODE_TYPES, VerificationResult
from .utils.io import safe_id, write_json
from .utils.reproducibility import claim_seed


class AMMFVPipeline:
    def __init__(self, config, corpus, clip, graph, controller, runner):
        self.config, self.corpus, self.clip, self.graph = config, corpus, clip, graph
        self.controller, self.runner = controller, runner
        self.debate = DebateController(runner, config["debate"])

    @torch.inference_mode()
    def verify(self, claim, forced_strategy=None):
        self.runner.reset_trace()
        seed = claim_seed(self.config["seed"], claim.claim_id)
        self.last_trace = {"claim": claim.public_context(), "seed": seed}
        acquired = self.controller.acquire(claim, forced_strategy)
        trace = {"claim": claim.public_context(), "seed": seed,
                 "model_assets": getattr(self, "assets_manifest", {}),
                 "routing": self.controller.last_routing_record,
                 "retrieval": {"strategy": acquired.strategy, "scores": acquired.scores,
                               "hops": acquired.hops,
                               "evidence": [item.to_dict() for item in acquired.evidence]}}
        self.last_trace = trace
        if acquired.strategy == "S":
            direct = acquired.direct_prediction
            result = VerificationResult(claim.claim_id, direct["verdict"], direct["explanation"], "S", [],
                                        epistemic_mode="parametric_non_retrieval",
                                        termination_reason="direct_generation",
                                        limitations=["No external evidence was acquired or inspected."])
        else:
            evidence = acquired.evidence[:self.config["retrieval"]["max_evidence"]]
            if evidence and self.graph is not None:
                features = torch.from_numpy(encode_claim_graph(claim, evidence, self.clip))[None].to(
                    self.config["graph"]["device"])
                types = torch.tensor([[0] + [NODE_TYPES[item.modality] for item in evidence]], device=features.device)
                output = self.graph(features, types, torch.ones_like(types, dtype=torch.bool))
                evidence, reference_evidence, trace["graph"] = build_agent_context(
                    evidence, output, self.config["context"]["reference_size"],
                    self.config["context"]["max_agents"], self.config["context"]["max_graph_depth"])
            else:
                before = [item.evidence_id for item in evidence]
                reference_evidence = evidence[:self.config["context"]["reference_size"]]
                count = self.config["context"]["reference_size"]
                if self.config["context"]["max_agents"] is not None:
                    count = min(count, self.config["context"]["max_agents"])
                evidence = evidence[:count]
                trace["graph"] = {"enabled": False, "candidate_ids": before,
                                  "agent_evidence_ids": [e.evidence_id for e in evidence],
                                  "selection_rule": "retrieval_order"}
            result, trace["deliberation"] = self.debate.verify(
                claim, evidence, acquired.strategy, seed, reference_evidence=reference_evidence)
        trace["calls"] = self.runner.calls
        trace["result"] = result.to_dict()
        if self.config["output"]["save_traces"]:
            path = Path(self.config["output"]["dir"]) / "traces" / f"{safe_id(claim.claim_id)}_{acquired.strategy}.json"
            result.trace_path = str(path)
            trace["result"] = result.to_dict()
            write_json(path, trace)
        return result, trace
