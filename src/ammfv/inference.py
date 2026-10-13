"""Resumable batch inference with explicit failure accounting and independent shards."""
from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from .utils.io import append_jsonl, file_sha256, fingerprint, read_jsonl, safe_id, write_json
from .utils.logging import save_run_manifest, setup_logger


def run_inference(pipeline, dataset, resume=False, shard_index=0, shard_count=1, forced_strategy=None):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    output_dir = Path(pipeline.config["output"]["dir"])
    suffix = f".shard{shard_index:04d}" if shard_count > 1 else ""
    output = output_dir / f"predictions{suffix}.jsonl"
    manifest = output_dir / f"inference_manifest{suffix}.json"
    provenance = {"config_sha256": fingerprint(pipeline.config),
                  "claims_sha256": file_sha256(dataset.path), "corpus_sha256": pipeline.corpus.sha256,
                  "prompt_hashes": pipeline.runner.prompt_manifest(),
                  "model_assets": pipeline.assets_manifest,
                  "shard_index": shard_index, "shard_count": shard_count, "forced_strategy": forced_strategy}
    existing = {}
    if output.exists():
        if not resume:
            raise FileExistsError(f"Prediction artifact exists: {output}; choose another output directory or --resume")
        if json.loads(manifest.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Resume provenance mismatch")
        existing = {row["claim_id"]: row for row in read_jsonl(output)}
    write_json(manifest, provenance)
    save_run_manifest(output_dir / f"worker{suffix}", pipeline.config, "infer", provenance)
    logger = setup_logger(output_dir / f"worker{suffix}")
    claims = [claim for index, claim in enumerate(dataset.claims) if index % shard_count == shard_index]
    for claim in tqdm(claims, desc=f"AMMFV shard {shard_index}/{shard_count}"):
        if claim.claim_id in existing:
            continue
        started = time.monotonic()
        try:
            result, _ = pipeline.verify(claim, forced_strategy)
            row = {**result.to_dict(), "status": "ok"}
        except Exception as error:
            logger.exception("claim=%s verification failed", claim.claim_id)
            failure_path = output_dir / "failures" / f"{safe_id(claim.claim_id)}.json"
            write_json(failure_path, {"claim": claim.public_context(), "error_type": type(error).__name__,
                                      "error": str(error), "partial_pipeline_trace": pipeline.last_trace,
                                      "calls": pipeline.runner.calls})
            row = {"claim_id": claim.claim_id, "status": "failed", "verdict": None,
                   "error_type": type(error).__name__, "error": str(error), "trace_path": str(failure_path)}
            if pipeline.config["output"]["fail_fast"]:
                raise
        row["elapsed_seconds"] = time.monotonic() - started
        append_jsonl(output, row)
    return output


def merge_shards(paths, output, expected_claim_ids=None):
    records = {}
    for path in paths:
        for row in read_jsonl(path):
            if row["claim_id"] in records:
                raise ValueError(f"Duplicate prediction ID across shards: {row['claim_id']}")
            records[row["claim_id"]] = row
    if expected_claim_ids is not None and set(records) != set(expected_claim_ids):
        raise ValueError("Merged shard IDs do not match expected evaluation claims")
    from .utils.io import write_jsonl
    write_jsonl(output, (records[key] for key in sorted(records)))
