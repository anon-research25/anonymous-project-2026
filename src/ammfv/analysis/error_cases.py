from __future__ import annotations

from ..evaluation.debate import conflict_group
from ..utils.io import write_jsonl


def export_error_cases(claims, predictions, task_mapping, output):
    by_id = {record["claim_id"]: record for record in predictions}
    rows = []
    for claim in claims:
        record = by_id.get(claim.claim_id, {})
        gold = task_mapping.gold(claim)
        final = task_mapping.prediction(record.get("verdict"))
        initial = (task_mapping.prediction(record["pre_debate_verdict"])
                   if record.get("pre_debate_verdict") else None)
        category = ("failed" if record.get("status", "missing") != "ok" else
                    "corrected" if initial is not None and initial != gold and final == gold else
                    "harmed" if initial == gold and final != gold else
                    "unchanged_wrong" if final != gold else "correct")
        rows.append({"claim_id": claim.claim_id, "claim": claim.text, "gold": gold,
                     "pre": initial, "post": final, "category": category,
                     "initial_conflict": conflict_group(record.get("initial_assessments", [])),
                     "evidence_ids": record.get("evidence_ids", []),
                     "explanation": record.get("explanation"), "trace_path": record.get("trace_path")})
    write_jsonl(output, rows)
