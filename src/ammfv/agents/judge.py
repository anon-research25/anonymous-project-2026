class JudgeAgent:
    def __init__(self, runner):
        self.runner = runner

    def arbitrate(self, claim, evidence, assessments, aggregate, counterfactual, standard,
                  converged, termination_reason, previous_record=None):
        previous_summary = None
        if previous_record is not None:
            previous_exchange = previous_record["exchanges"][-1]
            previous_summary = {
                "round": previous_record["round"], "judgment": previous_record["judge"],
                "evidence_assessments": [{"evidence_id": item["evidence_id"],
                                           "assessment": item["assessment"]}
                                          for item in previous_record["assessments"]],
                "C_recommendation": previous_exchange["refined_C"]["observed_evidence_recommendation"],
                "S_recommendation": previous_exchange["refined_S"]["verdict_recommendation"],
            }
        return self.runner.call("judge", {
            "claim": claim.public_context(), "assessments": assessments, "aggregate": aggregate,
            "counterfactual_critique": counterfactual, "standard_critique": standard,
            "previous_round": previous_summary, "controller_converged": converged,
            "controller_termination_reason": termination_reason,
            "convergence_rule": "All evidence-agent assessments and C/S observed-evidence recommendations "
                                "equal the previous complete round, compared by persistent evidence ID. "
                                "Stability is separate from veracity and does not establish correctness."
        }, evidence)
