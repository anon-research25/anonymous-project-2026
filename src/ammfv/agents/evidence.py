class EvidenceAgent:
    """One original evidence item per independent initial assessment."""

    def __init__(self, runner):
        self.runner = runner

    def assess(self, claim, evidence, round_index, previous_summary=None, previous_assessment=None):
        context = {"claim": claim.public_context(), "round": round_index,
                   "assigned_evidence_id": evidence.evidence_id}
        if previous_summary is not None:
            context["previous_aggregate"] = previous_summary
            context["previous_assessment"] = previous_assessment["assessment"]
        return self.runner.call("evidence", context, [evidence])
