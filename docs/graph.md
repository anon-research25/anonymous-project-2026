# Heterogeneous graph structure learning

## Nodes and shared-space representations

Node 0 is the claim, followed by the ordered original evidence items. Types are
claim=0, text=1, image=2. Padding uses -1 and a separate Boolean mask. CLIP features
are frozen and L2-normalized. A batch has features `[B,N,D]`, types `[B,N]`, and
node mask `[B,N]`. ViT-L/14's default projection dimension is 768.

Text encoding uses CLIP's 77-token maximum. Images are converted to RGB, resized
to the manuscript's 224×224 encoder size and processed by CLIP. Source captions
are not embedded in place of image content. These preprocessing choices do not
truncate the original corpus passage used for agent context serialization.

## Semantic adjacency

For normalized shared-space vectors, the practical metric is:

\[
M_s(h_i,h_j)=\cos(h_i,h_j).
\]

The optional formulated metric is:

\[
M_s(h_i,h_j)=\frac{1}{z}\sum_{a=1}^{z}
\cos(w_a\odot h_i,w_a\odot h_j).
\]

All eligible distinct valid-node pairs satisfying `M_s >= epsilon` are retained.
`claim_evidence` contains pairs with a claim endpoint; `evidence_evidence` contains
text/image pairs. Their sum is E_f. Initial adjacency is symmetric, but the
sampled outgoing neighborhoods can be directed. No self edges are counted toward
t. Residual self information is handled in message passing.

Similarity is a semantic weight, not an annotated support/contradiction label.
The graph does not claim to identify contradiction merely from cosine scores.

## Meta-paths and Gamma

A meta-path is a sequence of node types; Boolean matrix multiplication composes
typed relations to obtain endpoint reachability. Default direct paths cover
claim/text/image endpoint types. Two-hop `claim -> text -> image` and
`claim -> image -> text` reachability supply additional topology features.

The concrete Gamma input concatenates source/target frozen vectors, E_f weight,
source/target type embeddings and path-reachability indicators. A two-hidden-layer
MLP emits one scalar relevance logit per ordered pair. Types and path indicators
condition relevance on the heterogeneous relation structure.
Multi-hop reachability affects scoring but does not add factual nodes or turn
meta-path endpoints into new base neighbors outside E_f.

## Sampling and gradients

For each node, logits are normalized over eligible outgoing edges only. The
sampler performs up to t draws. Each training draw adds independent Gumbel noise
through `torch.nn.functional.gumbel_softmax`, producing a hard one-hot forward
selection with soft backward derivatives. A selected edge is removed from later
draws in that row, implementing sampling without replacement. Degree is bounded
by `min(t, eligible_neighbors)`.

Rows with no eligible edges skip softmax/sampling instead of applying softmax to
an all-negative-infinity row. Padding never participates. All original nodes
remain in the result.

Pruned edge weights multiply E_f by the straight-through selection tensor.
Typed message passing and readout use that tensor so supervised cross entropy
has a computational path to Gamma. Learned discrete choices are not backpropagated
through a Python-only top-k node list. If t keeps every candidate, structural
selection has little or no useful training signal; this is expected in the
unpruned control.

At inference, the same scorer uses deterministic top-t selection.
Temperature annealing affects training, not deterministic evaluation rankings.

## Supervised readout

Each directed pair of node types has a learned feature transform. Messages are
weighted by retained adjacency and normalized by outgoing absolute-weight sum.
Residual self projection, GELU, dropout and LayerNorm provide two default layers.
The readout concatenates:

1. The claim representation.
2. Mean original-evidence representation.
3. Weighted claim-neighbor representation.
4. Elementwise claim/neighbor interaction.

An MLP returns three logits and softmax probabilities pi. The only core graph
training objective is:

\[
L_{sup}=-\sum_{k=1}^{3} y_k\log\pi_k.
\]

This probability head trains the graph parameters. The final debate verdict is
computed separately; graph probabilities remain graph diagnostics.

## Graph-to-agent contexts

The graph retains every input node. Context selection is a separate allocation
interface with its own trace:

- Traverse directed retained edges from the claim up to `max_graph_depth`.
- Rank reachable original evidence by strongest traversed product of edge weights,
  with persistent ID tie breaking.
- Retain at most p reference items.
- If `max_agents=m` is specified, independently assess at most m of these items;
  reasoning/synthesis roles may still inspect the p permitted references.
- If no evidence is reachable, explicitly fall back to original claim/evidence
  similarity ranking; record `fallback_used=true`.

No fallback creates new facts. Traces include initial and pruned edge masks,
weights, sampling probabilities, all node IDs, reachable/reference/assessed/omitted
IDs and graph class probabilities. The m budget controls the number of
independently assessed evidence items within the p reference set.
