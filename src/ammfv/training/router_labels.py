"""Labeled training claims only: evaluate S/M/C and select the simplest success."""
from __future__ import annotations

from pathlib import Path

from ..retrieval.router import success_priority_label
from ..utils.io import append_jsonl, fingerprint, read_jsonl, write_json


def generate_router_labels(pipeline, claims, output, resume=False):
    claims = list(claims)
    if any(claim.split != "train" or claim.label is None for claim in claims):
        raise ValueError("Router-label generation is restricted to labeled TRAIN claims")
    if pipeline.config["retrieval"]["evidence_mode"] != "retrieved":
        raise ValueError("Strategy labels must compare real retrieval experts, not gold evidence")
    output = Path(output)
    provenance = {"pipeline_config": pipeline.config,
                  "claims_sha256": fingerprint([c.to_dict() for c in claims]),
                  "selection_rule": "S if S matches gold, else M if M matches gold, else C",
                  "corpus_sha256": pipeline.corpus.sha256,
                  "model_assets": pipeline.assets_manifest,
                  "prompt_hashes": pipeline.runner.prompt_manifest()}
    manifest = output.with_suffix(".manifest.json")
    existing = {}
    if output.exists():
        if not resume:
            raise FileExistsError("Use --resume to append only missing router labels")
        import json
        if json.loads(manifest.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Cannot resume router labeling with changed provenance")
        existing = {r["claim_id"]: r for r in read_jsonl(output)}
    write_json(manifest, provenance)
    for claim in claims:
        if claim.claim_id in existing:
            continue
        predictions, traces = {}, {}
        # Generate all three strategy outputs, including C, even though the
        # fallback assignment can be computed solely from S and M correctness.
        for strategy in ("S", "M", "C"):
            result, trace = pipeline.verify(claim, forced_strategy=strategy)
            predictions[strategy] = result.verdict
            traces[strategy] = result.trace_path
        label = success_priority_label(claim.label, predictions)
        append_jsonl(output, {"claim_id": claim.claim_id, "text": claim.text, "split": "train",
                             "strategy_label": label, "gold_label": claim.label,
                             "predictions": predictions,
                             "all_strategies_failed": all(v != claim.label for v in predictions.values()),
                             "trace_paths": traces})
