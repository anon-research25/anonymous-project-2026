def retrieval_metrics(claims, records, k_values=(1, 3, 5, 10)):
    """Optional diagnostics, independent of paper classification/explanation tables."""
    by_id = {record["claim_id"]: record for record in records}
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics["mrr"] = []
    for claim in claims:
        gold = set(claim.gold_evidence_ids)
        if not gold:
            continue
        predicted = by_id.get(claim.claim_id, {}).get("evidence_ids", [])
        for k in k_values:
            metrics[f"recall@{k}"].append(len(set(predicted[:k]) & gold) / len(gold))
        matches = [i + 1 for i, identifier in enumerate(predicted) if identifier in gold]
        metrics["mrr"].append(1 / min(matches) if matches else 0)
    return {"n_with_gold": len(metrics["mrr"]),
            **{key: sum(values) / len(values) if values else None for key, values in metrics.items()},
            "ranking_note": "Uses recorded acquisition order; BGE/CLIP scores are not cross-calibrated."}
