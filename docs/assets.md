# Model and dataset provenance artifacts

Asset identity is recorded automatically as files are prepared, downloaded, trained, and used for inference.

| Artifact | Producer | Contents |
|---|---|---|
| `data/{dataset}/preparation_manifest.json` | `prepare-dataset` | Native source hashes, counts, split policy, task mapping, corpus hash |
| `outputs/pretrained_assets.json` | `scripts/download_models.py` | Requested model revisions and resolved snapshots |
| `indexes/{modality}/metadata.json` | `build-index` | Encoder identity, corpus hash, vector count/dimension |
| `features/metadata.json` | `cache-features` | Encoder, corpus, evidence source, dimension, record hash |
| `graph/{best,last}.pt` | `train-graph` | Graph parameters, optimizer/scheduler/scaler, RNG, configuration, provenance |
| `graph/{best,last}.json` | `train-graph` | Epoch, step, best metric and configuration/provenance identity |
| `router_labels.manifest.json` | `router-labels` | Input claims, strategy rule, corpus, assets, prompt hashes |
| `router/partition.json` | `train-router` | Generated-training-label partition IDs |
| `router/best/` | `train-router` | HF model, tokenizer, S/M/C class mapping |
| `inference_manifest*.json` | `infer` | Claim/corpus hashes, model fingerprints, prompt hashes, shard identity |
| `run_manifest.json` | Training/inference | Effective configuration and software environment |
| `evaluation.json` | `evaluate` | Prediction/claim hashes, mapping, metric configuration |

The factory computes model asset fingerprints from the checkpoint files actually loaded. CLIP/BGE/LLaVA identities include configured model names/revisions and resolved commits. Graph loading checks architecture and CLIP provenance; indexes check their corpus and encoder. Resume compares saved artifact identity before appending predictions or labels.

Backbone configurations accept remote identifiers and local paths. Set fixed revision commits or immutable local snapshots when retaining experiment assets. Keep image files referenced by canonical evidence available for inference. Dataset and pretrained model licenses apply to their respective artifacts.
