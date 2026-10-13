"""Materialize or execute experiment matrices through the same public CLI.

Default action writes the concrete command plan. --execute runs model inference
and evaluation for each configured job.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

from ammfv.config import load_config
from ammfv.utils.io import write_json, atomic_text


VARIANT_OVERRIDES = {
    "full": {}, "no_images": {"retrieval.use_images": False},
    "no_adaptive": {"routing.adaptive": False, "routing.fixed_strategy": "M"},
    "no_graph": {"graph.enabled": False}, "no_debate": {"debate.enabled": False},
}


def override_string(key, value):
    return f"{key}={json.dumps(value)}"


def make_command(command, config, overrides, extra=()):
    result = [sys.executable, "-m", "ammfv", command, "--config", str(config)]
    for key, value in overrides.items():
        result.extend(["--set", override_string(key, value)])
    result.extend(extra)
    return result


def matrix_plan(spec):
    jobs = []
    for dataset, evidence, variant, seed in itertools.product(
            spec["datasets"], spec["evidence_modes"], spec["variants"], spec["seeds"]):
        directory = Path(spec["output_root"]) / dataset / evidence / variant / f"seed{seed}"
        overrides = {"seed": seed, "retrieval.evidence_mode": evidence,
                     "output.dir": str(directory), **VARIANT_OVERRIDES[variant]}
        jobs.append({"dataset": dataset, "evidence_mode": evidence, "variant": variant, "seed": seed,
                     "config": f"configs/datasets/{dataset}.yaml", "overrides": overrides,
                     "output_dir": str(directory)})
    return jobs


def sweep_plan(spec):
    jobs = []
    for value, evidence, seed in itertools.product(spec["values"], spec["evidence_modes"], spec["seeds"]):
        directory = Path(spec["output_root"]) / evidence / f"value{value}" / f"seed{seed}"
        overrides = {"seed": seed, "retrieval.evidence_mode": evidence, "output.dir": str(directory),
                     **spec.get("overrides", {}), spec["parameter"]: value}
        jobs.append({"parameter": spec["parameter"], "value": value, "evidence_mode": evidence,
                     "seed": seed, "config": spec["base_config"], "overrides": overrides,
                     "output_dir": str(directory)})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--plan-output", default="outputs/launch_plan.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    jobs = sweep_plan(spec) if "parameter" in spec else matrix_plan(spec)
    for job in jobs:
        prediction = str(Path(job["output_dir"]) / "predictions.jsonl")
        metric = str(Path(job["output_dir"]) / "evaluation.json")
        extra = ["--resume"] if args.resume else []
        job["commands"] = [make_command("infer", job["config"], job["overrides"], extra),
                            make_command("evaluate", job["config"], job["overrides"],
                                         ["--predictions", prediction, "--output", metric])]
    write_json(args.plan_output, {"spec": args.spec, "jobs": jobs})
    if args.execute:
        for job in jobs:
            print(f"Launching {job['output_dir']}", flush=True)
            for command in job["commands"]:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
