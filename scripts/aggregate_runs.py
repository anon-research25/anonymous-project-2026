"""Aggregate measured runs by seed, retaining metric and task provenance."""
import argparse
import json
from pathlib import Path

import numpy as np

from ammfv.utils.io import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metrics]
    protocols = [row["provenance"]["evaluation_config"] for row in rows]
    if any(protocol != protocols[0] for protocol in protocols):
        raise ValueError("Cannot aggregate different evaluation protocols")
    data_ids = [row["provenance"]["claims_sha256"] for row in rows]
    if len(set(data_ids)) != 1:
        raise ValueError("All seed runs must evaluate the same claim artifact")
    metrics = {}
    for key in ("label_accuracy", "precision", "recall", "f1"):
        values = [row["classification"][key] for row in rows]
        metrics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "values": values, "n_runs": len(values)}
    write_json(args.output, {"metrics": metrics, "protocol": protocols[0], "claims_sha256": data_ids[0],
                             "metric_files": args.metrics, "status": "aggregated_measured_runs"})
