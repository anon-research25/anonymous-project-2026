class StandardAgent:
    def __init__(self, runner):
        self.runner = runner

    def critique(self, claim, evidence, explanation_state, opponent=None, phase="initial"):
        return self.runner.call("standard", {"claim": claim.public_context(),
                                "explanation_state": explanation_state,
                                "counterfactual_critique": opponent, "phase": phase}, evidence)
