class CounterfactualAgent:
    """Dependency critique: hypothetical records never enter the evidence corpus."""

    def __init__(self, runner):
        self.runner = runner

    def critique(self, claim, evidence, explanation_state, opponent=None, phase="initial"):
        return self.runner.call("counterfactual", {"claim": claim.public_context(),
                                "explanation_state": explanation_state,
                                "standard_critique": opponent, "phase": phase}, evidence)
