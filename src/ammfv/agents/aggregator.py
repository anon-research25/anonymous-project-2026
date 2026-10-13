class Aggregator:
    def __init__(self, runner):
        self.runner = runner

    def synthesize(self, claim, evidence, assessments, rng):
        shuffled = list(assessments)
        rng.shuffle(shuffled)
        result = self.runner.call("aggregator", {"claim": claim.public_context(),
                                  "assessments": shuffled}, evidence)
        return result, [record["evidence_id"] for record in shuffled]
