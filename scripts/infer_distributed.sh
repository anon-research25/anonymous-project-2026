#!/usr/bin/env bash
set -euo pipefail
GPUS="${1:-5}"
CONFIG="${2:-configs/datasets/mocheg.yaml}"
torchrun --standalone --nproc_per_node "$GPUS" -m ammfv infer --config "$CONFIG"
# Merge explicitly using ammfv merge-shards and --claims after all workers finish.
