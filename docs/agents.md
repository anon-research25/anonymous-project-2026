# Agent execution and evidence contracts

All generative roles share a frozen LLaVA backbone. Independent initial evidence
assessments are separate calls with isolated context, not independently trained
models. No role has a web/search tool. Multi-hop retrieval is controlled by the
retriever and planner outside the debate.

## Role inputs and outputs

| Role | Permitted information | Output |
|---|---|---|
| Direct S | Claim only; explicitly parametric | Canonical verdict and explanation |
| Retrieval planner | Claim + recent observed items, prior queries, acquired IDs | Sufficient flag, grounded queries, unresolved assertions |
| Evidence agent | Claim + one assigned item; later previous summary | Assessment, observed facts, item references, rationale |
| Aggregator | Shuffled assessments + supplied reference items | Provisional verdict, assertion/evidence map, conflicts, explanation |
| Agent-C | Claim, reference items, current state, available S critique | Hypothetical scenarios and observed-case recommendation |
| Agent-S | Claim, reference items, current state, available C critique | Evidence audit and observed-case recommendation |
| Judge | Current/previous records, reference items, C/S critiques, stopping rule | Verdict, arbitration, unresolved issues, stopping fields |
| Rewriter | Judge record, permitted items, retained dependency analysis | Same verdict, faithful explanation, references, limitations |

## Round execution

Round zero calls each evidence agent without other agents' outputs. Assessments
are shuffled using a claim-specific seeded RNG and aggregated. C/S drafts are
formed against the same current explanation state. Each role then refines its
draft after reading the opponent's draft. An optional exchange count repeats
this inner critique process, carrying the updated critiques in the explanation
state.

The controller compares the current complete evidence/C/S recommendation
signature with the preceding complete round. It supplies the result and explicit
termination reason to the Judge, which must preserve both. At continuation,
the next evidence round sees the prior aggregate plus adjudicated correction
summary. At convergence or T exhaustion, the latest Judge record is rewritten
once into the final explanation.

Judge history retains the preceding judgment, evidence-ID/assessment pairs, and
C/S recommendations. Full exchanges remain in the trace rather than being
duplicated in the next prompt.

The signature compares canonical evidence labels keyed by persistent evidence ID,
Agent-C's observed-evidence recommendation, and Agent-S's recommendation. It does
not compare raw prose strings, require a majority vote, or interpret agreement as
truth. The first round cannot satisfy a cross-round stability criterion.

## Prompts and preprocessing

Prompt files under `src/ammfv/agents/prompts/v1/` contain a shared contract and
nine role definitions. The supplied supplementary working-note contracts provide
the evidence/debate role specifications. Direct generation and query planning
are additional explicit prompts required by the retrieval implementation.

For Mistral compatibility, controller role instructions are included in the user
message before delimited data; the processor's native chat template formats that
message. Original images are attached in the order declared by
`image_attachment_index`. The configured per-item text/caption tokenizer budgets
are distinct from CLIP encoder preprocessing. Truncation token counts are recorded
for each item. The fully expanded processor prompt is checked against the context
budget. Oversized visual requests are reprocessed with aspect-preserving bounded
image dimensions; original/retained sizes are logged. Complete image token blocks
and all permitted evidence IDs remain intact.

## Structured-output constraints

The runner requires complete JSON output fields with the declared types. It
checks canonical verdicts, assigned claim/evidence IDs, allowed structured evidence
references, later-round previous-assessment fields, CF-prefixed hypothetical IDs,
Judge stopping fields, and Rewriter verdict preservation. Counterfactual scenarios
are analytical JSON objects, never Evidence instances or corpus insertions.

Formatting retries use the same original evidence with a controller repair
instruction. They do not retrieve new material. After the retry budget, a contract
failure becomes a failed prediction and an inspectable failure trace.

These checks enforce structural consistency and retain the complete generation
records for evidence and explanation analysis.

## Optional confidence

Default confidence is null. The optional agreement score is the fraction of the
latest evidence/C/S recommendations matching the Judge verdict. It does not use
Judge self-certainty or a separate calibration model. It is not a calibrated
probability and never overrides the verdict. Enable only with explicit disclosure:

```bash
ammfv infer --config configs/datasets/mocheg.yaml --set debate.confidence=agreement \
  --set output.dir=outputs/mocheg/with_agreement_score
```
