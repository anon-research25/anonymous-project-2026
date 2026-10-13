param([string]$Config = "configs/datasets/mocheg.yaml")
$ErrorActionPreference = "Stop"
python scripts/run_pipeline.py --config $Config --stages infer evaluate --execute
if ($LASTEXITCODE -ne 0) { throw "AMMFV inference/evaluation pipeline failed" }
