import argparse

from ammfv.analysis.error_cases import export_error_cases
from ammfv.cli import claims_for
from ammfv.config import load_config
from ammfv.evaluation.labels import TaskLabelMapping
from ammfv.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/mocheg.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    export_error_cases(claims_for(config).claims, read_jsonl(args.predictions),
                       TaskLabelMapping(config["evaluation"]), args.output)


if __name__ == "__main__":
    main()
