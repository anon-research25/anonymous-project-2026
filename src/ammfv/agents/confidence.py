class AgreementConfidence:
    """Optional descriptive agreement score; deliberately not called calibrated.

    The paper lists possible confidence ingredients without a formula. This
    opt-in score uses the fraction of latest evidence/C/S recommendations equal
    to the judge verdict. It is never used to choose or override the verdict.
    """

    def score(self, verdict, assessments, counterfactual, standard):
        labels = [item["assessment"] for item in assessments]
        if counterfactual:
            labels.append(counterfactual["observed_evidence_recommendation"])
        if standard:
            labels.append(standard["verdict_recommendation"])
        return sum(label == verdict for label in labels) / len(labels) if labels else None
