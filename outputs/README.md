# Runtime outputs

Executions create distinct experiment directories containing predictions, computed metrics, manifests, logs, traces, failure records, and optional figures. Choose separate output directories for evidence policies, checkpoint selections, model profiles, seeds, and task mappings.

`results/reference/` contains manuscript-reported reference tables. The evaluation commands calculate run metrics from saved prediction files and record data/prediction hashes and evaluation settings. See `docs/outputs.md` for output contracts and resume/shard behavior.
