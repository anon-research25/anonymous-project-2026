"""Materialize and execute ordered dataset, training, inference and evaluation stages."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ammfv.config import load_config
from ammfv.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets/mocheg.yaml")
    parser.add_argument("--stages", nargs="+", default=["index", "features", "graph", "router_labels", "router", "infer", "evaluate"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root", help="Native benchmark release directory for the prepare stage")
    parser.add_argument("--plan-output", default="outputs/pipeline_plan.json")
    args = parser.parse_args()
    config = load_config(args.config)
    labels = str(Path("data") / config["data"]["dataset"] / "router_labels.jsonl")
    predictions = str(Path(config["output"]["dir"]) / "predictions.jsonl")
    prefix = [sys.executable, "-m", "ammfv"]
    options = ["--config", args.config]
    commands = {
        "prepare": prefix + ["prepare-dataset"] + options + ["--root", args.root or ""],
        "index": prefix + ["build-index"] + options,
        "features": prefix + ["cache-features"] + options,
        "graph": prefix + ["train-graph"] + options,
        "router_labels": prefix + ["router-labels"] + options + ["--output", labels],
        "router": prefix + ["train-router"] + options + ["--labels", labels],
        "infer": prefix + ["infer"] + options,
        "evaluate": prefix + ["evaluate"] + options + ["--predictions", predictions,
                    "--output", str(Path(config["output"]["dir"]) / "evaluation.json")],
    }
    if "prepare" in args.stages and not args.root:
        parser.error("The prepare stage requires --root")
    plan = [{"stage": name, "command": commands[name]} for name in args.stages]
    write_json(args.plan_output, plan)
    if args.execute:
        for stage in plan:
            print(stage["stage"], flush=True)
            subprocess.run(stage["command"], check=True)


if __name__ == "__main__":
    main()
