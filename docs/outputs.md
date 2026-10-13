# Artifacts and output contracts

## Training outputs

Graph training writes `best.pt`, `last.pt`, sidecar JSON, `best_metrics.json`,
`metrics.jsonl`, per-rank logs and `run_manifest.json`. Router training writes
partition IDs, HF trainer checkpoints, best model/tokenizer assets, validation
metrics and a run manifest. Router-label generation writes claim/strategy/gold
records, all-strategy failure indicators and its provenance manifest.

Feature directories contain `metadata.json`, `manifest.jsonl`, and claim-hashed
NPZ arrays. Arrays store frozen CLIP features and node types; manifests store
labels/splits/evidence IDs separately. Model inputs use feature arrays, not the
annotation text in manifests.

## Prediction records

Successful rows contain:

| Field | Meaning |
|---|---|
| claim_id | Persistent claim identifier |
| status | ok |
| verdict | One canonical three-state label |
| explanation | Generated explanation |
| strategy | S/M/C or GOLD; baseline names for comparators |
| epistemic_mode | evidence_grounded or parametric_non_retrieval |
| evidence_ids | Reference items actually permitted for final reasoning |
| pre_debate_verdict | Initial aggregator label, if applicable |
| initial_assessments | Independent pre-debate assessments |
| debate_invoked | Whether iterative C/S/J deliberation actually ran |
| rounds | Executed outer rounds |
| termination_reason | direct_generation/convergence/round_limit/no_evidence/etc. |
| consensus_reached | Cross-round stability Boolean |
| confidence | Null or documented descriptive agreement score |
| limitations | Retained uncertainty/constraints |
| trace_path | Full trace artifact |
| elapsed_seconds | Measured inference wall time when execution occurs |

Failed rows contain `status: failed`, null verdict, error type/message, and
failure trace path. Evaluation additionally synthesizes missing rows as errors;
it never deletes failures from the expected split denominator.

## Traces

Trace JSON keeps the public claim context, claim seed, route probabilities,
acquisition queries/hops, original evidence, graph masks/weights/sampling
probabilities, reference/assessed IDs, all debate rounds and role call attempts.
Each round retains evidence assessments, aggregate order/output, initial/refined
C/S critiques, Judge output and controller stopping fields. Rewriter output is
recorded separately.

Role calls record prompt hash, serialized user context, permitted IDs, image
paths, truncation accounting, cache key, cache status, output text, token counts,
image preprocessing dimensions, and contract errors for formatting attempts. Cache-hit token records describe
the cached generation; analysis counts only uncached attempts as this run's
generation cost.

## Evaluation records

`classification` values use scale 0..1; result tables/plots multiply by 100.
Explanation metrics use 0..100 and disclose variants/signatures. Confusion
matrices have benchmark ground-truth rows and prediction columns, including
failure/abstention columns. F1 is averaged from class-wise F1 values.

`coverage` exposes expected/present/missing/failed records and explanation gaps.
`debate` includes corrected/harmed and unchanged outcomes plus exclusions.
`provenance` includes prediction/claim file SHA-256 and the exact evaluation map.

Classification accuracy or ROUGE/BLEU does not establish evidence faithfulness,
counterfactual correctness or calibrated confidence. Trace inspection makes
those questions reviewable, but does not itself measure them.

## Resuming and shards

Resume verifies artifact identity and skips IDs already recorded, including
failed records. Independent workers must use different prediction shard files.
The merge command refuses duplicate IDs and optionally requires exact coverage
of a specified claim file. Shards are merged before evaluation, preserving
class-wise metric denominators.
