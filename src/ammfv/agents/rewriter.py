class Rewriter:
    def __init__(self, runner):
        self.runner = runner

    def rewrite(self, claim, evidence, judgment, counterfactual=None):
        return self.runner.call("rewriter", {"claim": claim.public_context(), "judgment": judgment,
                                "retained_dependency_analysis": counterfactual}, evidence)
