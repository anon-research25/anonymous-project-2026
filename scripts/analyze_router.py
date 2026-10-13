import argparse

from ammfv.analysis.routing import analyze_router_labels
from ammfv.utils.io import read_jsonl, write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, analyze_router_labels(read_jsonl(args.labels)))
