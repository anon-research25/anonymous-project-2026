# Experiment reproduction

## Main matrix

Prepare both datasets and train their graph/router assets using the README sequence. Then execute:

```bash
python scripts/launch.py --spec configs/experiments/matrix.yaml --execute
```

The matrix combines MOCHEG/COSMOS, retrieved/gold evidence, full/no-images/no-adaptive/no-graph/no-debate variants, and the configured seeds. Every job has a distinct output directory, resolved CLI overrides, predictions, and evaluation file. Without `--execute`, the launcher saves the concrete command plan.

The gold protocol defaults to bypass. A routed gold experiment sets `routing.gold_policy=route` and uses the trained router. Under bypass, removing adaptive retrieval leaves gold acquisition unchanged. Retrieved runs use the actual S/M/C acquisition experts.

## Training and component ablations

Default ablations remove components from fixed trained assets. No images removes image nodes and image agent inputs. No adaptive uses M. No graph selects acquisition-order references. No debate retains independent assessment and aggregation before one-pass output.

For retrained variants, select independent feature directories, graph outputs/checkpoints, router-label artifacts, and router outputs/checkpoints. Use the same index→features→graph→router_labels→router sequence with the variant settings. Model profiles that change representation dimensions use newly encoded features and matching graph parameters.

List multiple seeds in the matrix to measure inference ordering variation with fixed assets. For independent training runs, train each seed into its own graph/router paths and evaluate those checkpoints. `scripts/aggregate_runs.py` checks the same evaluated claim artifact and evaluation configuration before reporting means/standard deviations.

## Sensitivity studies

```bash
python scripts/launch.py --spec configs/sweeps/reference_size.yaml --plan-output outputs/p_plan.json --execute
python scripts/collect_sweep.py --plan outputs/p_plan.json --output outputs/p_metrics.jsonl
ammfv plot --kind sensitivity --input outputs/p_metrics.jsonl --parameter reference_size --metric label_accuracy --output outputs/p.png
python scripts/launch.py --spec configs/sweeps/agent_count.yaml --plan-output outputs/m_plan.json --execute
```

p ranges from 1 to 10 and controls the common reference budget. m ranges from 1 to 5 and controls independently assessed items within the reference set. Fixed synthesis/critique/Judge/Rewriter roles are unchanged. Traces identify every reference and assessed item. Keep acquisition, checkpoint, and unrelated budgets fixed across a sweep.

## Paired debate analysis

Initial and final predictions come from the same execution with the same acquisition and graph. Conflict membership is fixed before debate: at least one SUPPORTED and one REFUTED initial evidence assessment. All remaining eligible debated cases are non-conflicting.

Corrected means initial wrong/final right; harmed means initial right/final wrong. Unchanged cases remain in the group denominator. Net improvements equal corrected minus harmed. Reports retain group size, pre/post accuracy, delta, and exclusions for direct S routes, disabled debate, missing annotations, and failed records. The optional bootstrap module estimates paired uncertainty over these records.

## Baseline evaluation

Local LLaVA plain/verification-procedure/ICL comparators operate on acquired evidence records. LEAD-3 and greedy ROUGE ORACLE produce explanation-only records. External methods use fixed-evidence export and prediction import with explicit label mappings. All complete prediction files use the same classification evaluator; explanation-only files use `evaluate-explanations`.

## Saved experiment identity

Retain dataset preparation manifests, model download manifests, trained checkpoints, complete configuration, feature metadata, router-label provenance, inference manifests, prompt hashes, traces, predictions, and evaluation outputs. These records identify the task mapping, corpus, model commits, acquisition budgets, p/m/T, decoding settings, seed, device placement, and metric definitions.

Evaluation artifacts carry `status=computed_from_predictions`; manuscript reference tables carry `manuscript_reported`. Reference tables are stored in `results/reference/`; generated metrics stay in each output directory.
