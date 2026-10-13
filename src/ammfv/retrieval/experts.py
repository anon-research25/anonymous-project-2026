"""External retrieval is bounded to the configured evidence collection."""
from __future__ import annotations

from ..schema import Claim, RetrievalResult


class DirectExpert:
    """S route: a separately marked parametric prediction, without observed evidence."""

    def __init__(self, role_runner):
        self.runner = role_runner

    def retrieve(self, claim: Claim) -> RetrievalResult:
        prediction = self.runner.call("direct", {"claim": claim.public_context()}, evidence=[])
        return RetrievalResult("S", [], direct_prediction=prediction)


class SingleStepExpert:
    def __init__(self, corpus, text_index, image_index, bge, clip, text_k=8, image_k=4,
                 use_images=True):
        self.corpus, self.text_index, self.image_index = corpus, text_index, image_index
        self.bge, self.clip = bge, clip
        self.text_k, self.image_k, self.use_images = text_k, image_k, use_images
        if text_index.metadata["encoder"] != bge.identity:
            raise ValueError("Text index encoder differs from query encoder")
        if image_index.metadata["encoder"] != clip.identity:
            raise ValueError("Image index encoder differs from query encoder")

    def search(self, query: str, exclude: set[str] | None = None):
        # Scores from BGE and CLIP are not treated as calibrated across modalities.
        text_hits = self.text_index.search(self.bge.encode([query], query=True)[0], self.text_k, exclude)
        image_hits = []
        if self.use_images:
            image_hits = self.image_index.search(self.clip.encode_text([query])[0], self.image_k, exclude)
        return text_hits + image_hits

    def retrieve(self, claim: Claim) -> RetrievalResult:
        hits = self.search(claim.text)
        return RetrievalResult("M", self.corpus.select(identifier for identifier, _ in hits),
                               dict(hits), [{"hop": 0, "query": claim.text,
                                             "retrieved_ids": [identifier for identifier, _ in hits]}])


class MultiStepExpert:
    """Claim -> evidence -> grounded query planning -> additional evidence.

    The reasoner never accesses annotated labels or explanations. Generated
    summaries guide queries only; they are never appended as observed evidence.
    """

    def __init__(self, single_step: SingleStepExpert, role_runner, max_hops=3,
                 max_queries_per_hop=2, max_evidence=32, planner_context_size=7):
        self.single, self.runner = single_step, role_runner
        self.max_hops, self.max_queries = max_hops, max_queries_per_hop
        self.max_evidence = max_evidence
        self.planner_context_size = planner_context_size

    def retrieve(self, claim: Claim) -> RetrievalResult:
        collected, scores, hops = {}, {}, []
        queries = [claim.text]
        for hop in range(self.max_hops):
            before = len(collected)
            for query in queries[:self.max_queries]:
                hits = self.single.search(query, set(collected))
                new_ids = []
                for identifier, score in hits:
                    if len(collected) >= self.max_evidence:
                        break
                    collected[identifier] = self.single.corpus[identifier]
                    scores[identifier] = score
                    new_ids.append(identifier)
                hops.append({"hop": hop, "query": query, "retrieved_ids": new_ids})
            if len(collected) >= self.max_evidence or len(collected) == before or hop + 1 == self.max_hops:
                break
            planner_items = list(collected.values())[-self.planner_context_size:]
            plan = self.runner.call("retrieval_planner", {
                "claim": claim.public_context(), "hop": hop,
                "queries_permitted": self.max_queries,
                "previous_queries": [record["query"] for record in hops],
                "acquired_evidence_ids": list(collected),
            }, planner_items)
            hops[-1]["planner"] = plan
            hops[-1]["planner_evidence_ids"] = [item.evidence_id for item in planner_items]
            if plan["sufficient"] or not plan["queries"]:
                break
            queries = list(dict.fromkeys(plan["queries"]))[:self.max_queries]
        return RetrievalResult("C", list(collected.values()), scores, hops)
