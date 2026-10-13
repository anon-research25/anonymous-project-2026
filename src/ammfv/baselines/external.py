"""Export fixed evidence and import external baseline outputs without imitation."""
from __future__ import annotations

from ..schema import VERDICTS
from ..utils.io import fingerprint, read_jsonl, write_json, write_jsonl


EXTERNAL_METHODS = ("Triple-Check", "Ino", "MOCHEG", "HGTMFC", "LVLM4FV", "CRMFC-LLaVA", "ProgramFC", "DEFAME")


def export_evidence_bundle(claims, corpus, acquisition_records, output, method):
    if method not in EXTERNAL_METHODS:
        raise ValueError(f"Supported external method names: {EXTERNAL_METHODS}")
    by_id = {record["claim_id"]: record for record in acquisition_records}
    rows = []
    for claim in claims:
        record = by_id[claim.claim_id]
        items = corpus.select(record["evidence_ids"])
        if method == "ProgramFC":
            items = [item for item in items if item.modality == "text"]
        rows.append({"claim": claim.public_context(), "evidence": [item.to_dict() for item in items],
                     "method": method, "additional_external_tools_permitted": False,
                     "input_fingerprint": fingerprint([claim.public_context(), [e.to_dict() for e in items]])})
    write_jsonl(output, rows)
    write_json(str(output) + ".manifest.json", {"method": method, "n": len(rows),
                                                "protocol": "fixed supplied evidence only",
                                                "bundle_sha256": fingerprint(rows)})


def import_external_predictions(input_path, output, method, canonical_mapping, expected_ids=None):
    rows = []
    seen = set()
    for record in read_jsonl(input_path):
        identifier = record["claim_id"]
        if identifier in seen:
            raise ValueError(f"Duplicate external prediction {identifier}")
        seen.add(identifier)
        label = str(record["label"])
        if label not in canonical_mapping or canonical_mapping[label] not in VERDICTS:
            raise ValueError(f"Explicit external-to-canonical mapping missing for {label}")
        rows.append({"claim_id": identifier, "verdict": canonical_mapping[label],
                     "explanation": record.get("explanation", ""), "status": "ok", "strategy": method,
                     "evidence_ids": record.get("evidence_ids", []), "debate_invoked": False,
                     "external_provenance": record.get("provenance", {})})
    if expected_ids is not None and set(expected_ids) != seen:
        raise ValueError("External predictions do not exactly cover the evaluation split")
    write_jsonl(output, rows)
