from collections import Counter, defaultdict


def analyze_router_labels(records):
    records = list(records)
    strata = defaultdict(list)
    for record in records:
        strata[record["strategy_label"]].append(record)
    return {"n": len(records), "strategy_labels": dict(Counter(r["strategy_label"] for r in records)),
            "all_strategy_failure_count": sum(r["all_strategies_failed"] for r in records),
            "strategy_accuracy": {strategy: sum(r["predictions"][strategy] == r["gold_label"] for r in records) / len(records)
                                  if records else None for strategy in ("S", "M", "C")},
            "fallback_C_success_rate": sum(r["predictions"]["C"] == r["gold_label"] for r in strata["C"]) / len(strata["C"])
                                      if strata["C"] else None,
            "interpretation": "Operational strategy-relative labels, not intrinsic complexity annotations."}
