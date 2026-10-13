"""Bounded evidence reassessment plus exchanged standard/counterfactual critiques."""
from __future__ import annotations

import random

from ..schema import VerificationResult
from .aggregator import Aggregator
from .confidence import AgreementConfidence
from .convergence import assessment_signature, stabilized
from .counterfactual import CounterfactualAgent
from .evidence import EvidenceAgent
from .judge import JudgeAgent
from .rewriter import Rewriter
from .standard import StandardAgent


class DebateController:
    def __init__(self, runner, config):
        self.runner, self.config = runner, config
        self.evidence_agent, self.aggregator = EvidenceAgent(runner), Aggregator(runner)
        self.counterfactual, self.standard = CounterfactualAgent(runner), StandardAgent(runner)
        self.judge, self.rewriter = JudgeAgent(runner), Rewriter(runner)

    def verify(self, claim, evidence, strategy, seed, reference_evidence=None):
        reference_evidence = evidence if reference_evidence is None else reference_evidence
        if not evidence:
            result = VerificationResult(claim.claim_id, "NOT_ENOUGH_INFO",
                                        "No permitted observed evidence is available.", strategy, [],
                                        termination_reason="no_evidence",
                                        limitations=["No evidence-specific assessment was possible."])
            return result, {"rounds": [], "no_evidence": True}
        rng = random.Random(seed)
        rounds, signature, previous_summary, previous_assessments = [], None, None, {}
        initial, pre_verdict, final_judgment, final_counterfactual = None, None, None, None
        last_standard = None
        for round_index in range(self.config["max_rounds"]):
            assessments = [self.evidence_agent.assess(claim, item, round_index, previous_summary,
                                                      previous_assessments.get(item.evidence_id))
                           for item in evidence]
            aggregate, order = self.aggregator.synthesize(claim, reference_evidence, assessments, rng)
            if initial is None:
                initial, pre_verdict = assessments, aggregate["provisional_verdict"]
            if not self.config["enabled"]:
                polished = self.runner.call("one_shot", {
                    "claim": claim.public_context(), "assessments": assessments,
                    "aggregate": aggregate}, reference_evidence)
                result = VerificationResult(claim.claim_id, polished["verdict"], polished["explanation"],
                                            strategy, [e.evidence_id for e in reference_evidence], pre_verdict, initial,
                                            termination_reason="debate_disabled",
                                            limitations=polished["limitations"])
                return result, {"rounds": [{"assessments": assessments, "aggregate": aggregate,
                                           "aggregate_order": order}], "one_shot": polished}
            state = {"verdict": aggregate["provisional_verdict"], "explanation": aggregate["explanation"]}
            exchanges = []
            for exchange in range(self.config["critique_exchanges"]):
                # Both drafts see the same E_{i-1}; refinement sees the opponent's
                # draft, rather than letting generation order determine information.
                draft_c = self.counterfactual.critique(claim, reference_evidence, state, phase="initial")
                draft_s = self.standard.critique(claim, reference_evidence, state, phase="initial")
                refined_c = self.counterfactual.critique(claim, reference_evidence, state, draft_s, "refinement")
                refined_s = self.standard.critique(claim, reference_evidence, state, draft_c, "refinement")
                exchanges.append({"exchange": exchange, "state": state, "draft_C": draft_c,
                                  "draft_S": draft_s, "refined_C": refined_c, "refined_S": refined_s})
                state = {**state, "counterfactual_critique": refined_c, "standard_critique": refined_s}
            current = assessment_signature(assessments, refined_c, refined_s)
            converged = stabilized(signature, current)
            termination = ("convergence" if converged else "round_limit" if round_index + 1 == self.config["max_rounds"]
                           else "continuation")
            judgment = self.judge.arbitrate(claim, reference_evidence, assessments, aggregate, refined_c, refined_s,
                                            converged, termination, rounds[-1] if rounds else None)
            record = {"round": round_index, "assessments": assessments, "aggregate": aggregate,
                      "aggregate_order": order, "exchanges": exchanges, "judge": judgment,
                      "controller_converged": converged, "termination_reason": termination}
            rounds.append(record)
            final_judgment, final_counterfactual, last_standard = judgment, refined_c, refined_s
            if termination != "continuation":
                break
            signature = current
            previous_assessments = {a["evidence_id"]: a for a in assessments}
            previous_summary = {"provisional_verdict": aggregate["provisional_verdict"],
                                "explanation": aggregate["explanation"],
                                "adjudicated_update": {"verdict": judgment["verdict"],
                                                       "rationale": judgment["arbitration_rationale"],
                                                       "accepted_corrections": judgment["accepted_corrections"]}}
        rewritten = self.rewriter.rewrite(claim, reference_evidence, final_judgment, final_counterfactual)
        confidence = (AgreementConfidence().score(final_judgment["verdict"], rounds[-1]["assessments"],
                                                  final_counterfactual, last_standard)
                      if self.config["confidence"] == "agreement" else None)
        result = VerificationResult(claim.claim_id, final_judgment["verdict"], rewritten["explanation"],
                                    strategy, [e.evidence_id for e in reference_evidence], pre_verdict, initial,
                                    True, len(rounds), final_judgment["termination_reason"],
                                    final_judgment["consensus_reached"], confidence=confidence,
                                    limitations=rewritten["limitations"])
        return result, {"rounds": rounds, "rewriter": rewritten,
                        "reference_ids": [e.evidence_id for e in reference_evidence],
                        "assessed_ids": [e.evidence_id for e in evidence],
                        "confidence_rule": self.config["confidence"]}
