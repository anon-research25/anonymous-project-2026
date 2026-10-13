"""Field-map conversion for exported JSON, JSONL and CSV data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..schema import Claim, Evidence
from ..utils.io import file_sha256, read_jsonl, write_json, write_jsonl


def lookup(record: dict, path: str | None, default=None):
    if path is None:
        return default
    current = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def raw_records(path: str | Path, records_key: str | None = None):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            yield from csv.DictReader(handle)
    elif path.suffix.lower() == ".jsonl":
        yield from read_jsonl(path)
    elif path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if records_key is not None:
                raise
            yield from read_jsonl(path)
            return
        if records_key:
            value = lookup(value, records_key)
        if not isinstance(value, list):
            raise ValueError("JSON source must contain a list; supply records_key if nested")
        yield from value
    else:
        raise ValueError(f"Expected JSON, JSONL or CSV: {path}")


def array_value(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("Evidence-ID columns must contain JSON arrays")
    return value


def prepare_claims(raw_path, output, dataset, split, fields: dict, label_mapping: dict,
                   records_key=None):
    if not label_mapping:
        raise ValueError("Explicit benchmark-to-canonical label mapping is required")
    claims = []
    for raw in raw_records(raw_path, records_key):
        raw_label = lookup(raw, fields.get("label"))
        benchmark = str(raw_label) if raw_label is not None else None
        if benchmark is not None and benchmark not in label_mapping:
            raise ValueError(f"Unmapped benchmark label {benchmark!r}")
        claim = Claim(claim_id=str(lookup(raw, fields["claim_id"])),
                      text=lookup(raw, fields["text"], ""), split=split, dataset=dataset,
                      label=label_mapping.get(benchmark), benchmark_label=benchmark,
                      gold_evidence_ids=tuple(str(v) for v in array_value(
                          lookup(raw, fields.get("gold_evidence_ids")))),
                      explanation=lookup(raw, fields.get("explanation")),
                      metadata={"raw_source": Path(raw_path).name})
        if claim.claim_id == "None":
            raise ValueError("Source record has no claim ID")
        claims.append(claim)
    if len({c.claim_id for c in claims}) != len(claims):
        raise ValueError("Duplicate claim IDs in raw source")
    write_jsonl(output, (claim.to_dict() for claim in claims))
    write_json(str(output) + ".manifest.json", {
        "source_sha256": file_sha256(raw_path), "output_sha256": file_sha256(output),
        "count": len(claims), "dataset": dataset, "split": split,
        "field_mapping": fields, "label_mapping": label_mapping,
        "note": "Existing source split preserved; no random repartitioning."})


def prepare_corpus(raw_path, output, fields: dict, records_key=None):
    evidence = []
    for raw in raw_records(raw_path, records_key):
        identifier = lookup(raw, fields["evidence_id"])
        if identifier is None:
            raise ValueError("Every observed evidence item needs a persistent source ID")
        source = {key: lookup(raw, raw_key) for key, raw_key in fields.get("source", {}).items()
                  if lookup(raw, raw_key) is not None}
        item = Evidence(evidence_id=str(identifier), modality=lookup(raw, fields["modality"]),
                        text=lookup(raw, fields.get("text"), "") or "",
                        image_path=lookup(raw, fields.get("image_path")),
                        caption=lookup(raw, fields.get("caption"), "") or "", source=source)
        evidence.append(item)
    if len({e.evidence_id for e in evidence}) != len(evidence):
        raise ValueError("Duplicate evidence IDs in raw collection")
    write_jsonl(output, (item.to_dict() for item in evidence))
    write_json(str(output) + ".manifest.json", {
        "source_sha256": file_sha256(raw_path), "output_sha256": file_sha256(output),
        "count": len(evidence), "fields": fields,
        "note": "Only explicitly mapped evidence fields copied; no labels/explanations injected."})
