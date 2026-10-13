from __future__ import annotations

from pathlib import Path

from ..utils.io import file_sha256, read_jsonl, write_json
from .classification import classification_metrics
from .debate import paired_debate_metrics
from .explanations import explanation_metrics
from .labels import TaskLabelMapping


def evaluate_predictions(dataset, prediction_path, config, output):
    records = list(read_jsonl(prediction_path))
    by_id = {record["claim_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate prediction IDs")
    expected = {claim.claim_id for claim in dataset.claims}
    if set(by_id) - expected:
        raise ValueError("Predictions contain IDs outside the evaluation split")
    task = TaskLabelMapping(config)
    gold, predicted, hypotheses, references, paired = [], [], [], [], []
    missing, failed, explanation_failures = 0, 0, 0
    for claim in dataset.claims:
        if claim.label is None:
            raise ValueError("Classification evaluation requires annotated canonical labels")
        record = by_id.get(claim.claim_id)
        if record is None:
            missing += 1
            record = {"claim_id": claim.claim_id, "status": "missing", "verdict": None}
        elif record.get("status", "ok") != "ok":
            failed += 1
        gold.append(task.gold(claim))
        predicted.append(task.prediction(record.get("verdict") if record.get("status", "ok") == "ok" else None))
        paired.append(record)
        if claim.explanation is not None:
            text = record.get("explanation", "") if record.get("status", "ok") == "ok" else ""
            explanation_failures += not bool(text)
            hypotheses.append(text)
            references.append(claim.explanation)
    metrics = {"classification": classification_metrics(gold, predicted, task.classes, config["averaging"]),
               "coverage": {"expected": len(dataset), "present": len(records), "missing": missing,
                            "failed": failed, "empty_or_failed_explanations": explanation_failures},
               "debate": paired_debate_metrics(dataset.claims, paired, task),
               "provenance": {"claims_sha256": file_sha256(dataset.path),
                              "predictions_sha256": file_sha256(prediction_path), "evaluation_config": config},
               "status": "computed_from_predictions"}
    if references:
        metrics["explanations"] = explanation_metrics(hypotheses, references, config["rouge_stemmer"], config["bleu_tokenizer"])
    write_json(Path(output), metrics)
    return metrics
