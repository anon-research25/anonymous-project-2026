from __future__ import annotations

from ..schema import Claim, RetrievalResult


class ExpertController:
    """Hard expert dispatch; this MoE routes algorithms, not token-level experts."""

    def __init__(self, config, corpus, experts, router=None):
        self.config, self.corpus, self.experts, self.router = config, corpus, experts, router
        self.last_routing_record = {}

    def acquire(self, claim: Claim, forced_strategy: str | None = None) -> RetrievalResult:
        gold = self.config["retrieval"]["evidence_mode"] == "gold"
        bypass = gold and self.config["routing"]["gold_policy"] == "bypass"
        if forced_strategy:
            strategy, probabilities = forced_strategy, None
        elif bypass:
            strategy, probabilities = "GOLD", None
        elif not self.config["routing"]["adaptive"]:
            strategy, probabilities = self.config["routing"]["fixed_strategy"], None
        else:
            strategy, probabilities = self.router.predict(claim.text)
        self.last_routing_record = {"strategy": strategy, "probabilities": probabilities,
                                    "gold_policy": self.config["routing"]["gold_policy"] if gold else None,
                                    "forced": forced_strategy is not None}
        if gold and strategy != "S":
            evidence = self.corpus.select(claim.gold_evidence_ids)
            if not self.config["retrieval"]["use_images"]:
                evidence = [item for item in evidence if item.modality == "text"]
            return RetrievalResult(strategy, evidence, hops=[{"gold_ids": [e.evidence_id for e in evidence]}])
        if strategy not in self.experts:
            raise ValueError(f"Unknown retrieval expert {strategy}")
        return self.experts[strategy].retrieve(claim)
