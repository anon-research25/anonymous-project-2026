"""Derive fixed evidence inputs from actual AMMFV traces for baseline comparisons."""
import argparse
import json
from pathlib import Path

from ammfv.utils.io import read_jsonl, write_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scope", choices=["acquired", "references", "assessed"], default="references")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for record in read_jsonl(args.predictions):
        if record.get("status", "ok") != "ok":
            raise ValueError("Fixed-evidence export requires successful traces for every supplied claim")
        trace = json.loads(Path(record["trace_path"]).read_text(encoding="utf-8"))
        if args.scope == "acquired":
            ids = [item["evidence_id"] for item in trace["retrieval"]["evidence"]]
        elif args.scope == "assessed":
            ids = [item["evidence_id"] for item in record["initial_assessments"]]
        else:
            ids = record["evidence_ids"]
        rows.append({"claim_id": record["claim_id"], "strategy": record["strategy"],
                     "evidence_ids": ids, "scope": args.scope, "trace_path": record["trace_path"]})
    write_jsonl(args.output, rows)
