# Programmatic interfaces

The public CLI assembles these same concrete modules. Code snippets below are
usage examples, not execution records.

## Full pipeline

```python
from ammfv.config import load_config
from ammfv.factory import make_pipeline
from ammfv.schema import Claim

config = load_config("configs/datasets/mocheg.yaml", ["output.dir=outputs/programmatic"])
pipeline = make_pipeline(config)
claim = Claim(claim_id="my-claim", text="The supplied claim.",
              split="unlabeled", dataset="custom")
result, trace = pipeline.verify(claim)
print(result.verdict, result.explanation)
```

`make_pipeline()` loads the configured permitted collection, encoders, retrieval
indexes, router if required, graph checkpoint if enabled and shared generative
backend. It rejects absent trained assets rather than silently using a randomly
initialized verifier. `forced_strategy` optionally selects S/M/C without a router.

`make_pipeline(config, load_router=False, direct_only=True)` constructs the S
pipeline with no retrieval encoders, indexes, router, or graph. Its controller
selects S by default; the public claim context and result contract are unchanged.

`result` is a `VerificationResult`; `trace` includes original evidence and detailed
graph/deliberation/call records. Call `result.to_dict()` for JSON serialization.
The claim's annotations are never passed through `public_context()`.

## Standalone graph module

```python
from ammfv.config import load_config
from ammfv.graph.model import HeterogeneousGraphVerifier
from ammfv.training.losses import SupervisedVeracityLoss

config = load_config("configs/datasets/mocheg.yaml")
model = HeterogeneousGraphVerifier(config["graph"])
# Supply frozen features [B,N,768], node_types [B,N], and node_mask [B,N].
output = model(features, node_types, node_mask)
loss = SupervisedVeracityLoss()(output["logits"], annotated_class_ids)
loss.backward()
```

`output` has `logits`, `probabilities`, `hidden`, initial `graph`, and `pruned`.
During `.train()`, pruning uses straight-through Gumbel sampling. During `.eval()`,
it uses deterministic top-t. Graph probabilities are diagnostics/training outputs,
not final AMMFV labels. All valid node identities survive pruning.

## Retrieval and routing

`BGEEncoder.encode(texts, query=False)` returns normalized passage vectors; query
encoding adds the configured query prefix. `CLIPEncoder.encode_text` and
`encode_images` share CLIP's projection space. `DenseIndex.search(query,k,exclude)`
returns persistent-ID/score pairs, with optional dedup exclusions.

`SingleStepExpert.retrieve(claim)` returns text/image evidence and one-hop records.
`MultiStepExpert.retrieve` calls grounded planning between bounded acquisitions.
`DirectExpert.retrieve` returns no observed evidence, only a separately marked
parametric prediction.

`ClaimRouter.predict(claim_text)` returns strategy and S/M/C probabilities.
`success_priority_label(gold,predictions)` implements the exact S/M/C label rule,
including the C all-failure fallback.

## Agents

`RoleRunner.call(role,context,evidence)` formats versioned role instructions,
serializes permitted evidence, attaches original images, generates with the shared
backend and validates output structure. The call returns a JSON-compatible object;
`runner.calls` contains attempt records. `runner.reset_trace()` begins a new claim.

`DebateController.verify(claim,evidence,strategy,seed,reference_evidence=None)`
runs isolated evidence assessment, aggregation, critique exchange, stability
checking, arbitration and rewriting. `evidence` identifies independently assessed
items; optional `reference_evidence` records broader permitted synthesis context.

## Metrics and analysis

`classification_metrics(gold,predicted,classes,averaging)` explicitly includes
failure/abstention columns. `explanation_metrics(hypotheses,references)` records
ROUGE F1 and corpus BLEU variants. `paired_debate_metrics` compares the same run's
initial aggregate and final verdict by pre-debate assessment group.

Analysis utilities expose optional paired/bootstrap intervals, routing-label
failure strata, error-case exports, graph figures and standalone HTML traces.
These diagnostics operate on actual records; none generate assumed paper outcomes.
