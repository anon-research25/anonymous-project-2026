from __future__ import annotations

from ..schema import VerificationResult


class LlavaVerifier:
    """Direct, verification-procedure, and one-example ICL comparators."""

    def __init__(self, runner, variant="plain", demonstration=None):
        self.runner, self.variant, self.demonstration = runner, variant, demonstration
        if variant not in {"plain", "pe", "icl1"}:
            raise ValueError(f"Unknown LLaVA comparator {variant}")
        if variant == "icl1" and (not demonstration or demonstration.get("split") != "train"):
            raise ValueError("ICL-1 requires one explicitly selected training demonstration")

    def verify(self, claim, evidence):
        self.runner.reset_trace()
        context = {"claim": claim.public_context()}
        if self.variant == "pe":
            context["verification_procedure"] = [
                "Identify material assertions and contextual alignment.",
                "Distinguish observed support, contradiction, and insufficient coverage.",
                "Cite decisive evidence and preserve unresolved conflicts."]
        elif self.variant == "icl1":
            if self.demonstration["claim_id"] == claim.claim_id:
                raise ValueError("Cannot use the evaluated claim as its own demonstration")
            context["training_demonstration_not_current_evidence"] = self.demonstration
        output = self.runner.call("one_shot", context, evidence)
        return VerificationResult(claim.claim_id, output["verdict"], output["explanation"],
                                  f"llava_{self.variant}", [item.evidence_id for item in evidence],
                                  termination_reason="baseline_single_pass", limitations=output["limitations"])
