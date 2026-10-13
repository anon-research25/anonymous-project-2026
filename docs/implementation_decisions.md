# Implementation configuration

The AMMFV implementation exposes acquisition, graph learning, reference allocation, and deliberation as separate modules. Defaults reside in `configs/base.yaml`; dataset, model, hardware, and experiment profiles override these settings.

## Expert routing and acquisition

| Component | Implementation |
|---|---|
| Router backbone | DeBERTa-v3-base with an S/M/C classification head |
| Router loss | Three-class cross entropy |
| Strategy labels | S if correct, otherwise M if correct, otherwise C |
| Router partition | Seeded 10% holdout within generated training labels |
| S expert | Claim-only frozen LLaVA direct prediction |
| M expert | BGE passage top-8 and CLIP image top-4 |
| C expert | Up to 3 acquisition hops, 2 queries per hop, 32 items |
| Planner context | Most recent 7 original evidence items, prior queries, acquired IDs |
| Index backend | Chunked exact NumPy search or FAISS IndexFlatIP |
| Deduplication | Persistent evidence IDs with previously acquired IDs excluded |

The planner produces grounded queries and a sufficient-evidence flag. It stops when sufficient, no new query remains, no new item is acquired, or a configured acquisition budget is reached. Query text and planning summaries remain acquisition records. BGE and CLIP modality scores are ranked within their own indexes.

## Graph representation and training

CLIP ViT-L/14 encodes normalized 768-dimensional claim/text/image vectors. Text uses a 77-token limit and images use 224×224 preprocessing. The graph includes one claim node followed by the acquired evidence nodes. Semantic thresholding with epsilon=0.2 creates symmetric initial edges without self links.

Gamma concatenates source/target representations, semantic edge weight, node-type embeddings, and Boolean meta-path indicators. Its feed-forward layers use GELU and produce a scalar relevance score. Valid-neighbor softmax normalizes the scores. Training uses up to t=4 straight-through Gumbel-softmax draws without replacement; inference selects deterministic top-t relevance. Selection is directed so each node retains its outgoing degree cap.

Two typed residual propagation layers transform the pruned graph. Nine source/target type relations use separate linear projections. The readout concatenates the claim representation, pooled evidence, weighted claim neighbors, and their interaction. A three-class softmax is trained with supervised cross entropy.

| Graph training setting | Default |
|---|---|
| Optimizer | AdamW |
| Learning rate / weight decay | 1e-4 / 0.01 |
| Epochs / batch size | 20 / 16 |
| Warmup | 10% of optimizer steps |
| Scheduler | Cosine decay |
| Gradient norm limit | 1.0 |
| Gumbel temperature | Initial 1.0, multiplier 0.95 per epoch, minimum 0.3 |
| Checkpoint selection | Validation macro-F1 |

Features are frozen; graph parameters remain trainable. AMP and gradient accumulation are configurable. Empty-edge graphs retain a connected backward path to relevance parameters. AMP overflow skips the optimizer/scheduler update together. Distributed validation reduces exact, unpadded split totals. Checkpoints retain per-rank RNG states alongside optimizer, scheduler, scaler, and feature provenance.

## Reference and agent allocation

Claim-reachable evidence is ranked by path strength up to depth 3, with persistent IDs resolving ties. If no evidence is reachable, original claim/evidence similarity selects the context. Pruning retains all nodes; reference allocation records both selected and omitted IDs.

`context.reference_size` is p, the maximum number of reference items used by synthesis, critique, judgment, and rewriting. `context.max_agents` is m, the cap on independently assessed evidence items. Null m assigns one assessor to every selected reference. The fixed role set remains Aggregator/C/S/Judge/Rewriter across these budgets.

## Deliberation and stopping

Initial evidence calls contain the claim and one assigned item. Later calls also contain the preceding aggregate and adjudicated correction summary. The seeded aggregator shuffles assessments. C/S drafts share the same current state; each refinement receives the opponent draft. Inner exchanges carry the refined critiques forward.

The controller compares evidence labels keyed by item ID and the observed-case C/S label recommendations against the prior round. An unchanged complete signature yields convergence. First-round stability is false. The Judge receives and preserves the controller stopping fields separately from the veracity label. The latest judgment is used at the outer round limit. Rewriter output must retain the Judge verdict.

The optional agreement score is the fraction of final evidence/C/S recommendations matching the Judge. It is descriptive, uncalibrated, and never changes the adjudication. Confidence is omitted by default.

## Protocol variants

- Gold bypass supplies annotated evidence directly to graph/debate; gold route selects S/M/C before supplying annotated evidence to M/C.
- No images removes visual acquisition and visual graph/agent inputs.
- No adaptive selects fixed M acquisition.
- No graph allocates references in acquisition order.
- No debate runs evidence assessment, aggregation, and a one-pass final output.
- Component ablations use fixed checkpoints by default; independent retraining selects separate caches and model paths.

## Data and metric contracts

MOCHEG uses canonical SUPPORTED/REFUTED/NOT_ENOUGH_INFO labels. COSMOS uses paired-caption contextual consistency: same-image and cross-image source pairs supervise training/validation, and original test annotations define binary evaluation. SUPPORTED maps to 0 and REFUTED maps to 1; NEI is an abstention error.

Classification includes all expected examples, including failed/missing predictions. Macro F1 averages class-wise F1 values. Explanation metrics compute mean per-example ROUGE F1 and corpus BLEU at orders 1..4. Paired debate statistics use initial/final verdicts from a single acquisition and graph execution.

## Assets and execution records

Backbone names, effective model commits, corpus hashes, cache metadata, checkpoint fingerprints, prompt hashes, and complete configurations are recorded in generated manifests. Prediction traces contain the evidence actually supplied and each agent call. Source annotations and reference explanations are excluded from claim inference context. Counterfactual objects remain labeled hypothetical scenarios.
