# Manuscript-to-code map

The following map links the AMMFV manuscript mechanisms to their source modules.
Line numbers refer to the local manuscript snapshot.

| Source file | SHA-256 |
|---|---|
| `bare_jrnl.tex` | `74ac9a14ccc23af3d2db18c7b68b200266150602f1e3e29bae1b9a212c68b526` |
| `supplementary_material.tex` | `b49425738975a16773c84bdfdee9b73432c647385dbd74328291ef1fc854ac1b` |

The source manuscripts remain outside this repository at their original location.
No manuscript text, figures, authoring files, or compiled PDFs were modified.

## Problem and retrieval

| Manuscript specification | Source | Concrete implementation |
|---|---|---|
| Claim c; textual E_t; visual E_v; SUPP/REF/NEI | Problem Definition, lines 676–694 | `schema.py`, `data/corpus.py`, `evaluation/labels.py` |
| S parametric prediction is distinct from evidence-grounded judgment | Method retrieval, lines 739–758 | `retrieval/experts.py:DirectExpert`, `pipeline.py` |
| BGE textual retrieval | Lines 745–746 | `encoders/bge.py`, `retrieval/index.py` |
| CLIP claim/image features and retrieval | Lines 735–736 | `encoders/clip.py`, `SingleStepExpert.search` |
| Iterative reasoner/retriever interaction | Lines 748–749 | `MultiStepExpert.retrieve`, `prompts/v1/retrieval_planner.txt` |
| Finetuned LM claim-only routing | Lines 751–758 | `retrieval/router.py`, `training/router_trainer.py` |
| Train-only S/M/C success-priority assignment and all-failure C fallback | Lines 754–758 | `success_priority_label`, `training/router_labels.py` |

## Graph equations

| Manuscript specification | Concrete implementation | Qualification |
|---|---|---|
| CLIP projection h_c/h_e | `data/features.py`, `encoders/clip.py` | Frozen normalized shared-space features |
| E_s[c,e] = M_s(h_c,h_e) if similarity >= epsilon else 0 | `graph/construction.py` | Explicit edge mask preserves threshold semantics |
| z-head weighted cosine with Hadamard feature weights | `graph/similarity.py:MultiHeadWeightedCosine` | Optional formulated metric |
| Practical simplification to standard cosine | `cosine_adjacency`, `graph.metric=cosine` | Default matches practical manuscript statement |
| E_c evidence–evidence links and E_f union | `SemanticGraphBuilder` | All text/image pair types retained as eligible relations |
| Typed meta-path neighbor sets | `graph/metapaths.py` | Boolean relation composition |
| Gamma(h_c,h_e,E_f[c,e]) scalar MLP | `graph/pruning.py:NeighborRelevance` | Extended explicitly with type/path features |
| s = softmax(m) over valid neighbors | `GumbelNeighborhoodSampler` | Invalid/padded edges masked |
| Gumbel-softmax z with temperature tau | Same sampler | Straight-through hard forward, soft backward |
| Repeat sampling t times, original nodes retained | Same sampler, `graph/context.py` | Without replacement; directed outgoing-degree cap |
| Supervised cross entropy on graph class probabilities | `training/losses.py`, `graph/model.py` | Typed propagation and claim/evidence readout |

The graph section appears at manuscript lines 760–810. The loss appears at
lines 884–898. `graph/layers.py` implements typed propagation and the probability
readout; `graph/context.py` implements the graph-to-agent allocation interface.

## Deliberation

| Manuscript specification | Source | Concrete implementation |
|---|---|---|
| One isolated initial assessment per evidence | Lines 827–831 | `agents/evidence.py`, `debate.py` |
| Reassessment conditioned on preceding aggregate | Lines 830–831, 858–860 | `EvidenceAgent.assess`, previous-summary state |
| Shuffled evidence assessments to aggregator | Lines 832–834 | `agents/aggregator.py` |
| Agent-C perturbs premises analytically | Lines 837–849 | `agents/counterfactual.py`, counterfactual prompt and schema |
| Agent-S direct observed-evidence critique | Same subsection | `agents/standard.py` |
| Draft critique, exchange, refinement and state update | Lines 850–857 | `DebateController` exchange loop |
| Judge arbiters and Boolean stopping differs from label | Lines 862–867 | `agents/judge.py`, `agents/convergence.py` |
| Rewriter synthesizes final report | Lines 867–873 | `agents/rewriter.py`, verdict equality contract |
| All agents preserve previous assessments => stop | Lines 869–872 | Canonical evidence/C/S assessment signature |
| Latest assessments used at round limit | Same subsection | Explicit `round_limit`, no false convergence |
| Optional confidence in [0,1] | Lines 874–878 | Default omitted; opt-in agreement score documented |

The supplementary working notes define independent evidence, aggregator,
counterfactual, standard, judge, and rewriter contracts. Those specifications are
implemented in `agents/prompts/v1/` and `agents/contracts.py`.

## Training and experiments

| Manuscript specification | Source | Concrete implementation |
|---|---|---|
| CLIP ViT-L/14, text 77 tokens, images 224×224 | Lines 985–987 | `configs/base.yaml`, `CLIPEncoder` |
| LLaVA-v1.6-Mistral-7B common backbone | Lines 988–992 | `LlavaBackend`, role-based prompts |
| Temperature 0, stochastic output disabled | Same subsection | `do_sample=False`, greedy generation |
| MOCHEG train/dev/test and explanations | Lines 973–974 | Split-preserving adapters and canonical claim records |
| COSMOS contextual consistency benchmark | Lines 971–972 | `data/benchmarks.py`, paired-caption conversion and binary output map |
| LA, precision, recall, per-class then averaged F1 | Lines 979–983 | `evaluation/classification.py` |
| ROUGE-1/2/L and BLEU-1..4 | Lines 981–984 and Table 2 | `evaluation/explanations.py` |
| Visual/routing/graph/debate ablations | Lines 1075–1125 | `configs/ablations`, experiment matrix |
| Conflicting if initial SUPPORTED and REFUTED coexist | Lines 1129 onward | `evaluation/debate.py:conflict_group` |
| Paired initial aggregate/final predictions, fixed acquired evidence/graph | Same analysis | Same-run trace/result records; no rerun for paired scoring |
| p reference-size and m agents sensitivity | Lines 1155–1330 | Reference and assessment budgets, sweep configurations |

## End-to-end workflow

`cli.py` exposes native data preparation, index construction, frozen-feature
caching, graph optimization, training-only router supervision, router optimization,
inference, evaluation, and analysis. `scripts/run_pipeline.py` sequences these
stages; `scripts/launch.py` launches main matrices and sensitivity sweeps. The
dataset preparation and binary/canonical maps are implemented in
`data/benchmarks.py` and `evaluation/labels.py`. Complete operational settings
are listed in `configs/base.yaml` and `docs/implementation_decisions.md`.
