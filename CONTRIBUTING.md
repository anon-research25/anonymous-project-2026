# Contributing

Keep method components modular and maintain the manuscript-to-code map when changing behavior. Preserve canonical labels, persistent evidence IDs, source partitions, and annotation-free claim inference contexts.

- Keep counterfactual scenarios separate from observed corpus evidence.
- Keep supervised graph probabilities separate from final generative verdicts.
- Version role prompts and retain content hashes in generated records.
- Save complete configuration and asset provenance for experiment changes.
- Keep manuscript reference tables separate from computed evaluation artifacts.
- Preserve explicit benchmark mappings and failure/abstention accounting.
- Use external evidence/prediction interfaces for separately maintained baseline implementations.

Document changed parameters, affected workflows, and output contracts with each contribution. The source distribution does not include a test suite or CI workflow.
