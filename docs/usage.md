# Detailed usage guide

[Back to the project README](../README.md)

This guide covers configuration options, alternative training protocols, distributed execution, baselines, and result analysis. Run commands from the repository root.

## Environment installation

Use Python 3.10 or newer. Install a PyTorch build appropriate for the machine before installing the package when selecting a specific CUDA runtime.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional dependencies:

```bash
python -m pip install -e ".[index]"          # FAISS exact inner-product indexing
python -m pip install -e ".[logging]"        # TensorBoard
python -m pip install -e ".[quantization]"   # NF4 LLaVA loading
```

`ammfv` and `python -m ammfv` expose the same CLI. Run commands from the repository root so dataset, checkpoint, and output paths resolve consistently. Full inference requires the configured pretrained backbones and trained graph/router artifacts. CPU and quantized configurations are provided for alternative placements.

## Data preparation

### MOCHEG

Obtain the release through the [MOCHEG dataset repository](https://github.com/PLUM-Lab/Mocheg). Keep the released training, validation, and test folders and their `Corpus2.csv` files. Place the released image directories and the supplementary sentence-level retrieval collection under the same dataset root.

```bash
python -m ammfv prepare-dataset --config configs/datasets/mocheg.yaml --root data/raw/mocheg
python -m ammfv inventory --config configs/datasets/mocheg.yaml --output data/mocheg/inventory.json
```

The converter groups repeated claim rows by ID, normalizes supported/refuted/NEI labels, associates proof images, imports retrieval passages and image assets, and preserves source splits. It writes:

```text
data/mocheg/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── corpus.jsonl
└── preparation_manifest.json
```

Native image paths are resolved to their source files. Keep that source directory available after conversion. Reference explanations are stored with claims for evaluation; the evidence collection contains passages and images.

### COSMOS

Obtain the annotations and images through the [COSMOS dataset repository](https://github.com/shivangi-aneja/COSMOS). Use a root containing `train.json`, `val.json`, `test.json`, and the corresponding image paths.

```bash
python -m ammfv prepare-dataset --config configs/datasets/cosmos.yaml --root data/raw/cosmos --pairs-per-image 1
python -m ammfv inventory --config configs/datasets/cosmos.yaml --output data/cosmos/inventory.json
```

The task is caption-pair contextual consistency with an associated image. The query contains both captions. Source test labels map `0` / not-ooc to SUPPORTED and `1` / ooc to REFUTED. Train/validation caption collections produce same-image positive pairs and cross-image negative pairs using the configured seed. Their records carry `label_source=self_supervised_source_pair`; test records carry `label_source=source_annotation`. Source partitions remain separate.

The COSMOS evaluation profile maps predictions back to the binary benchmark labels. NOT_ENOUGH_INFO is an abstention counted as an error. Test caption pairs and test labels are kept in claim records, separately from retrieval evidence. See [Data protocol](data_protocol.md) for exact fields and pair construction.

### Canonical data and custom collections

Each claim JSONL row has `claim_id`, `text`, `split`, and `dataset`. Training/evaluation rows additionally contain `label`; gold-evidence evaluation uses `gold_evidence_ids`. Optional `explanation` and `benchmark_label` support reference scoring and task mapping.

```json
{"claim_id":"example-001","text":"The pictured event happened in the stated location.","split":"test","dataset":"custom","label":"REFUTED","gold_evidence_ids":["passage-001","image-001"]}
```

Evidence rows use persistent IDs and a modality:

```json
{"evidence_id":"passage-001","modality":"text","text":"A source passage describing the event.","source":{"url":"https://example.org/source"}}
{"evidence_id":"image-001","modality":"image","image_path":"event.jpg","caption":"A supplied source caption.","source":{"collection":"custom"}}
```

Relative image paths resolve against `data.image_root`; absolute paths remain valid. All gold IDs must resolve in `corpus.jsonl`. Canonical inference input may omit labels and explanations. `Claim.public_context()` exposes only the claim ID and text to generative models.

Adapters support CSV, JSON, and JSONL exports:

```bash
python -m ammfv prepare --kind claims --adapter configs/adapters/mocheg.yaml --input data/export/train.csv --output data/mocheg/train.jsonl --split train
python -m ammfv prepare --kind corpus --adapter configs/adapters/mocheg.yaml --input data/export/evidence.jsonl --output data/mocheg/corpus.jsonl
```

`scripts/chunk_corpus.py` creates overlapping passage chunks with source IDs. Its `--claims` option remaps annotated document IDs into retained chunk IDs after deduplication. `scripts/extract_frames.py` extracts image assets and sampled-frame timestamps through ffmpeg. Schemas and examples are supplied in `schemas/` and `examples/`.

## Model and weight preparation

| Component | Default backbone | Role |
|---|---|---|
| CLIP | `openai/clip-vit-large-patch14` | Shared graph features and visual retrieval |
| BGE | `BAAI/bge-large-en-v1.5` | Normalized dense textual retrieval |
| LLaVA | `llava-hf/llava-v1.6-mistral-7b-hf` | Shared frozen generative backbone for all roles |
| Router | `microsoft/deberta-v3-base` | Trainable S/M/C classifier |

```bash
python scripts/download_models.py --config configs/datasets/mocheg.yaml --manifest outputs/pretrained_assets.json
```

The download manifest records resolved snapshots. Model configuration accepts Hugging Face identifiers or local directories. `revision` selects the snapshot; effective commits are retained in model identities. For offline use, set local model paths and use the Hugging Face offline environment variables.

Training produces `checkpoints/mocheg/graph/best.pt` and `checkpoints/mocheg/router/best/`. COSMOS uses the analogous `checkpoints/cosmos/` paths. Loading checks parameter shapes, graph architecture, encoder identity, and asset fingerprints. Model changes affecting CLIP dimensions require matching indexes, cached features, graph training, and router supervision.

## Training

### 1. Build retrieval indexes

```bash
python -m ammfv build-index --config configs/datasets/mocheg.yaml
```

BGE encodes passage nodes. CLIP encodes image nodes and text queries in its shared projection space. Index directories contain vectors, persistent IDs, and encoder/corpus metadata. The NumPy backend uses chunked exact search; `--set retrieval.index_backend=faiss` selects FAISS exact inner-product search. Text and image rankings retain their own similarity scores.

### 2. Cache frozen CLIP graph features

```bash
python -m ammfv cache-features --config configs/datasets/mocheg.yaml
```

The default caches training and validation claims with their gold evidence. Each graph stores the claim node followed by evidence nodes in NPZ format, plus a split/label/ID manifest. Text features use 77 CLIP tokens; image features use 224×224 preprocessing. Agent contexts use independent tokenizer budgets and original image attachments.

For retrieved graph-training evidence:

```bash
python -m ammfv retrieve --config configs/datasets/mocheg.yaml --split train --strategy M --output data/mocheg/retrieval_train.jsonl
python -m ammfv retrieve --config configs/datasets/mocheg.yaml --split validation --strategy M --output data/mocheg/retrieval_validation.jsonl
python -m ammfv cache-features --config configs/datasets/mocheg.yaml --retrieval-records data/mocheg/retrieval_train.jsonl data/mocheg/retrieval_validation.jsonl --output data/mocheg/features_retrieved
```

Use `--strategy C` for multi-hop acquisition. Multiple cache inputs are joined by claim ID; duplicate or missing records are rejected. Image removal is applied before caching when `retrieval.use_images=false`.

### 3. Train the heterogeneous graph

```bash
python -m ammfv train-graph --config configs/datasets/mocheg.yaml
# Retrieved cache alternative
python -m ammfv train-graph --config configs/datasets/mocheg.yaml --features data/mocheg/features_retrieved
```

The objective is supervised cross entropy:

```text
L_sup = - mean_examples sum_classes y[class] * log pi[class]
```

Frozen encoder features feed trainable Gamma relevance, optional weighted similarity, relation-specific message passing, and the class readout. Training uses AdamW, cosine warmup, gradient clipping, accumulation, optional AMP, and Gumbel-temperature annealing. Validation macro-F1 selects `best.pt`; `last.pt` stores the latest epoch.

Resume an epoch checkpoint:

```bash
python -m ammfv train-graph --config configs/datasets/mocheg.yaml --resume checkpoints/mocheg/graph/last.pt
```

Distributed training:

```bash
torchrun --standalone --nproc_per_node=5 -m ammfv train-graph --config configs/datasets/mocheg.yaml
```

Checkpoints include optimizer, scheduler, scaler, epoch, step, per-rank RNG states, configuration, and feature provenance. Distributed validation uses disjoint unpadded subsets and globally reduced totals.

### 4. Generate router supervision

```bash
python -m ammfv router-labels --config configs/datasets/mocheg.yaml --output data/mocheg/router_labels.jsonl
```

Each labeled training claim is evaluated through S, M, and C. Strategy labels follow:

```text
S if prediction_S == gold
otherwise M if prediction_M == gold
otherwise C
```

C remains the fallback if every strategy is incorrect. Records retain all three predictions, failure flags, and strategy trace paths. This stage requires the trained graph and retrieval indexes; it does not require a router checkpoint.

```bash
python -m ammfv router-labels --config configs/datasets/mocheg.yaml --output data/mocheg/router_labels.jsonl --resume
```

Resume checks corpus, inputs, assets, prompt hashes, and configuration before appending missing records.

### 5. Train the router

```bash
python -m ammfv train-router --config configs/datasets/mocheg.yaml --labels data/mocheg/router_labels.jsonl
```

The router trains on claim text and generated strategy labels. A seeded 10% holdout is drawn from those training claims. Validation/test benchmark labels never generate router supervision. Hugging Face Trainer handles checkpointing and best-model selection. Resume with `--resume` pointing to a Trainer checkpoint directory.

### End-to-end launchers

```bash
python scripts/run_pipeline.py --config configs/datasets/mocheg.yaml --root data/raw/mocheg --stages prepare index features graph router_labels router infer evaluate --execute
python scripts/run_pipeline.py --config configs/datasets/cosmos.yaml --root data/raw/cosmos --stages prepare index features graph router_labels router infer evaluate --execute
```

Omit `--execute` to save the concrete command plan. Stages are selectable. PowerShell and Bash convenience scripts provide the MOCHEG training/evaluation sequence; `scripts/train_graph_distributed.sh` launches graph DDP.

## Evaluation and inference

### Retrieved-evidence inference

```bash
python -m ammfv infer --config configs/datasets/mocheg.yaml
python -m ammfv evaluate --config configs/datasets/mocheg.yaml --predictions outputs/mocheg/retrieved/full/predictions.jsonl --output outputs/mocheg/retrieved/full/evaluation.json
```

```bash
python -m ammfv infer --config configs/datasets/cosmos.yaml
python -m ammfv evaluate --config configs/datasets/cosmos.yaml --predictions outputs/cosmos/retrieved/full/predictions.jsonl --output outputs/cosmos/retrieved/full/evaluation.json
```

Inference produces verdicts, explanations, strategy choices, initial assessments, round counts, stopping reasons, optional agreement scores, and trace paths. Full traces include acquisition hops, graph matrices, reference allocation, agent drafts/refinements, Judge outputs, and generation token counts.

Multimodal requests are checked after processor expansion. If native images exceed the context budget, aspect-preserving bounded images are reprocessed with original/retained dimensions logged. Evidence IDs and complete image token blocks are preserved. Judge history retains the preceding adjudication and assessment signature without duplicating its full exchange transcript.

### Gold-evidence inference

```bash
python -m ammfv infer --config configs/experiments/mocheg_gold.yaml
python -m ammfv evaluate --config configs/experiments/mocheg_gold.yaml --predictions outputs/mocheg/gold/full/predictions.jsonl --output outputs/mocheg/gold/full/evaluation.json
python -m ammfv infer --config configs/experiments/cosmos_gold.yaml
```

`routing.gold_policy=bypass` sends annotated evidence directly to graph/debate. `routing.gold_policy=route` applies claim routing: S remains direct; M/C consume gold evidence. Both policies are explicit configuration choices. Under bypass, disabling adaptive retrieval leaves gold acquisition unchanged.

### Custom claims and fixed strategies

```bash
python -m ammfv infer --config configs/datasets/mocheg.yaml --input data/custom/claims.jsonl --set output.dir=outputs/custom
python -m ammfv infer --config configs/datasets/mocheg.yaml --strategy C --set output.dir=outputs/mocheg/fixed_C
python -m ammfv infer --config configs/datasets/mocheg.yaml --resume
```

A forced strategy bypasses router loading. Forced S also skips retrieval encoders, indexes, and graph loading. Resume verifies artifact identity and skips recorded claim IDs. Formatting failures and generation errors retain partial traces; evaluation counts failed/missing predictions in the expected split denominator.

### Sharded inference

```bash
torchrun --standalone --nproc_per_node=2 -m ammfv infer --config configs/datasets/mocheg.yaml
python -m ammfv merge-shards --inputs outputs/mocheg/retrieved/full/predictions.shard0000.jsonl outputs/mocheg/retrieved/full/predictions.shard0001.jsonl --claims data/mocheg/test.jsonl --output outputs/mocheg/retrieved/full/predictions.jsonl
```

Each worker writes a distinct shard. Merge rejects duplicate IDs and checks expected coverage. Evaluate the merged artifact.

### Metrics

Classification includes label accuracy, precision, recall, class-wise F1, configured macro/weighted/micro averaging, and confusion matrices. Explanation evaluation includes mean per-example ROUGE-1/2/L F1 and corpus BLEU-1 through BLEU-4, with tokenizer/signature records. Classification values use 0..1; ROUGE/BLEU use 0..100.

Paired debate analysis compares the initial aggregate and final verdict from the same execution. A conflicting group contains initial SUPPORTED and REFUTED assessments. Reports include pre/post accuracy, corrected, harmed, unchanged cases, net improvements, and exclusions. Retrieval metrics measure acquired-ID coverage and ranking against annotated evidence IDs.

```bash
python -m ammfv retrieval-metrics --config configs/datasets/mocheg.yaml --records data/mocheg/retrieval_test.jsonl --output outputs/mocheg/retrieval_metrics.json
python scripts/evaluate_graph_head.py --config configs/datasets/mocheg.yaml --output outputs/mocheg/graph_head.jsonl
```

The graph-head command evaluates its classifier separately from the final AMMFV verdict.

## Configuration

YAML `extends` resolves parent files relative to the current configuration. Shared ancestors are merged once; later profiles override earlier profiles. Data/output paths resolve from the working directory. Repeated `--set dotted.key=value` options provide typed YAML overrides; environment variables in string values are expanded.

| Key | Default | Controls |
|---|---|---|
| `retrieval.evidence_mode` | `retrieved` | Retrieved or annotated evidence |
| `routing.adaptive` | `true` | Claim classifier or fixed strategy |
| `routing.fixed_strategy` | `M` | Expert when adaptive routing is disabled |
| `routing.gold_policy` | `bypass` | Gold evidence with or without route selection |
| `retrieval.text_k`, `image_k` | `8`, `4` | Modality-specific top-k |
| `retrieval.max_hops` | `3` | C acquisition rounds |
| `retrieval.queries_per_hop` | `2` | Queries per acquisition round |
| `retrieval.max_evidence` | `32` | Acquired item cap |
| `retrieval.planner_context_size` | `7` | Recent planner evidence window |
| `graph.metric` | `cosine` | Standard or multi-head weighted cosine |
| `graph.epsilon` | `0.2` | Initial semantic edge threshold |
| `graph.neighbors` | `4` | Outgoing sampled-neighbor cap |
| `graph.temperature` | `1.0` | Initial Gumbel temperature |
| `graph.hidden_dim`, `layers` | `256`, `2` | Typed graph encoder |
| `context.reference_size` | `7` | Reference-item budget p |
| `context.max_agents` | `null` | One assessor per reference; integer sets assessor cap m |
| `context.max_graph_depth` | `3` | Graph reachability depth |
| `debate.max_rounds` | `3` | Outer deliberation cap T |
| `debate.critique_exchanges` | `1` | Reciprocal C/S refinement exchanges per round |
| `debate.confidence` | `none` | Optional descriptive agreement score |
| `llm.do_sample`, `temperature` | `false`, `0` | Greedy generation |
| `agents.evidence_text_tokens` | `320` | Per-item passage context |
| `agents.format_retries` | `2` | Structured-output repair budget |
| `output.save_traces` | `true` | Detailed trace artifacts |

The optional agreement score counts final evidence/C/S recommendations matching the Judge; it does not change the verdict. Inference pruning uses deterministic top-t relevance selection. Graph-to-agent allocation ranks claim-reachable evidence by path strength; an empty reachable set falls back to original claim/evidence similarity.

Combine profiles in a configuration file, for example `configs/experiments/cosmos_v100.yaml`:

```yaml
extends:
  - ../datasets/cosmos.yaml
  - ../hardware/v100_5gpu.yaml
output:
  dir: outputs/cosmos/v100
```

Alternative CLIP dimensions and learned metric architectures use separately trained graph assets. Runtime reference sizes, round counts, device placement, and checkpoint paths can be selected independently.

## Main experiments and ablations

The main matrix covers MOCHEG/COSMOS, retrieved/gold evidence, and four component removals:

```bash
python scripts/launch.py --spec configs/experiments/matrix.yaml --execute
```

The default matrix performs fixed-checkpoint component ablations. Variant controls are:

| Variant | Behavior |
|---|---|
| `full` | Adaptive acquisition, graph, and debate |
| `no_images` | Text acquisition and text-only graph/agent inputs |
| `no_adaptive` | Fixed M acquisition |
| `no_graph` | Acquisition-order references with unchanged p/m budgets |
| `no_debate` | Evidence assessment, aggregation, and single-pass final output |

Standalone YAMLs reside in `configs/ablations/`. Additional profiles select fixed S/M/C acquisition, weighted similarity, and a large neighbor cap. For retrained variants, select distinct feature, graph, router-label, and router paths through the same training commands. Seed-specific checkpoints measure training variation; fixed checkpoints with changed inference seeds measure inference ordering variation.

Sensitivity studies:

```bash
python scripts/launch.py --spec configs/sweeps/reference_size.yaml --plan-output outputs/p_sweep_plan.json --execute
python scripts/collect_sweep.py --plan outputs/p_sweep_plan.json --output outputs/p_sweep.jsonl
python -m ammfv plot --kind sensitivity --input outputs/p_sweep.jsonl --parameter reference_size --metric label_accuracy --output outputs/p_sweep.png
python scripts/launch.py --spec configs/sweeps/agent_count.yaml --plan-output outputs/m_sweep_plan.json --execute
```

The p sweep covers 1..10 references; the m sweep covers 1..5 independently assessed items. Aggregator/C/S/Judge/Rewriter receive the selected reference set; only independently assessed items contribute evidence-agent votes and cross-round assessment signatures. Full protocol details are in [Reproduction](reproduction.md).

## Baselines

Local comparators include plain LLaVA, a verification-procedure prompt, one-training-example ICL, LEAD-3, and greedy ROUGE extractive ORACLE. Generative comparators produce complete verdict/explanation records; extractive comparators are explanation-only.

```bash
python -m ammfv retrieve --config configs/datasets/mocheg.yaml --split test --strategy M --output data/mocheg/retrieval_test.jsonl
python -m ammfv baseline --config configs/datasets/mocheg.yaml --method plain --evidence-records data/mocheg/retrieval_test.jsonl --output outputs/baselines/plain.jsonl
python -m ammfv baseline --config configs/datasets/mocheg.yaml --method lead3 --evidence-records data/mocheg/retrieval_test.jsonl --output outputs/baselines/lead3.jsonl
python -m ammfv evaluate-explanations --config configs/datasets/mocheg.yaml --predictions outputs/baselines/lead3.jsonl --output outputs/baselines/lead3_metrics.json
```

ICL uses a JSON demonstration with `split=train`; ORACLE uses reference explanations for greedy sentence selection. External baseline implementations consume exported evidence bundles and return imported predictions through `export-external` / `import-external`, then use the same evaluator.

## Logs, visualization, and result analysis

```bash
python -m ammfv analyze --predictions outputs/mocheg/retrieved/full/predictions.jsonl --output outputs/mocheg/retrieved/full/analysis.json
python -m ammfv plot --kind training --input checkpoints/mocheg/graph/metrics.jsonl --output outputs/mocheg/graph_training.png
python scripts/analyze_errors.py --config configs/datasets/mocheg.yaml --predictions outputs/mocheg/retrieved/full/predictions.jsonl --output outputs/mocheg/errors.jsonl
python scripts/analyze_router.py --labels data/mocheg/router_labels.jsonl --output outputs/mocheg/router_analysis.json
```

`trace-html` renders a selected trace to standalone HTML. `plot --kind graph` renders its initial/pruned graph. `tables` combines evaluation files into CSV and optional LaTeX tables. `scripts/aggregate_runs.py` reports run means and standard deviations after checking matching task/data protocols. `analysis/bootstrap.py` provides paired bootstrap intervals.

[Outputs](outputs.md) documents predictions, partial failure traces, model checkpoints, metric logs, resume identities, and shard contracts. [Assets](assets.md) describes generated provenance records. `results/reference/` contains manuscript-reported tables; computed evaluations are saved under each run's output directory.
