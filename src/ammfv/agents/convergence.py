def assessment_signature(assessments, counterfactual, standard):
    """Canonical recommendations, not free-form explanation string equality."""
    evidence_states = tuple(sorted((item["evidence_id"], item["assessment"]) for item in assessments))
    return (evidence_states, counterfactual["observed_evidence_recommendation"], standard["verdict_recommendation"])


def stabilized(previous, current):
    return previous is not None and previous == current
