# Local dataset artifacts

`prepare-dataset` imports native MOCHEG/COSMOS releases into canonical `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `corpus.jsonl` files. It also writes preparation manifests with source hashes, split counts, and task mappings.

Feature caches and retrieval indexes reside beneath each dataset directory. Image records reference source files; retain those files at the prepared paths. Large datasets, image assets, feature caches, and indexes are excluded from version control. Files in `examples/` illustrate canonical record structure with explicitly fictional content.

Native conversion, exported-data adapters, inventory, and custom-dataset configuration are documented in the root README and `docs/data_protocol.md`.
