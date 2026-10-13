def conflict_group(initial_assessments):
    labels = {item["assessment"] for item in initial_assessments}
    return "conflicting" if {"SUPPORTED", "REFUTED"} <= labels else "non_conflicting"


def paired_debate_metrics(claims, records, task_mapping):
    """Table-style same-run pre/post comparisons, including unchanged verdicts."""
    claims = {claim.claim_id: claim for claim in claims}
    groups = {name: [] for name in ("conflicting", "non_conflicting")}
    excluded = {"no_debate": 0, "failed": 0, "unlabeled": 0}
    for record in records:
        claim = claims.get(record["claim_id"])
        if claim is None or claim.label is None:
            excluded["unlabeled"] += 1
            continue
        if record.get("status", "ok") != "ok":
            excluded["failed"] += 1
            continue
        if not record.get("debate_invoked") or record.get("pre_debate_verdict") is None:
            excluded["no_debate"] += 1
            continue
        gold = task_mapping.gold(claim)
        before = task_mapping.prediction(record["pre_debate_verdict"]) == gold
        after = task_mapping.prediction(record["verdict"]) == gold
        groups[conflict_group(record["initial_assessments"])].append((before, after))
    output = {}
    for name, pairs in groups.items():
        n = len(pairs)
        corrected = sum(not before and after for before, after in pairs)
        harmed = sum(before and not after for before, after in pairs)
        output[name] = {"n": n, "pre_accuracy": sum(before for before, _ in pairs) / n if n else None,
                        "post_accuracy": sum(after for _, after in pairs) / n if n else None,
                        "corrected": corrected, "harmed": harmed, "net_corrected": corrected - harmed,
                        "delta_accuracy_pp": 100 * (corrected - harmed) / n if n else None,
                        "unchanged_correct": sum(before and after for before, after in pairs),
                        "unchanged_wrong": sum(not before and not after for before, after in pairs)}
    return {"groups": output, "excluded": excluded,
            "protocol": "Initial assessment disagreement; same-run initial aggregator vs final verdict. "
                        "No second retrieval or graph evaluation for the paired comparison."}
