#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/datasets/mocheg.yaml}"
python scripts/run_pipeline.py --config "$CONFIG" --stages infer evaluate --execute
