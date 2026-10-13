param([string]$Config = "configs/datasets/mocheg.yaml")
$ErrorActionPreference = "Stop"
python scripts/run_pipeline.py --config $Config --stages index features graph router_labels router --execute
if ($LASTEXITCODE -ne 0) { throw "AMMFV training pipeline failed" }
