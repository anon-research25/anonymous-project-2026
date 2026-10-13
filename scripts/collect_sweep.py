"""Read a launcher plan and collect actual sensitivity metrics only."""
import argparse
import json
from pathlib import Path

from ammfv.utils.io import write_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = []
    for job in plan["jobs"]:
        path = Path(job["output_dir"]) / "evaluation.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))["classification"]
        parameter = job.get("parameter", "context.reference_size").split(".")[-1]
        rows.append({parameter: job.get("value"), "evidence_mode": job["evidence_mode"],
                     "seed": job["seed"], **{key: metrics[key] for key in ("label_accuracy", "f1")}})
    write_jsonl(args.output, rows)
