# Data preparation and task protocol

## Canonical records

Claim JSONL fields are `claim_id`, `text`, `split`, `dataset`, optional canonical `label`, `gold_evidence_ids`, `explanation`, `benchmark_label`, and `metadata`. Splits are train/validation/test/unlabeled. Evidence JSONL fields are `evidence_id`, `modality`, `text`, `image_path`, `caption`, and `source`.

Canonical labels are SUPPORTED, REFUTED, and NOT_ENOUGH_INFO. Text evidence requires a nonempty passage; image evidence requires a path. Relative paths resolve against `data.image_root`. Source metadata is preserved; supplied captions are distinguished from direct image observations. Claim inference exposes only ID and text.

## Native MOCHEG import

```bash
ammfv prepare-dataset --config configs/datasets/mocheg.yaml --root data/raw/mocheg
```

The converter reads split-specific `Corpus2.csv` files. It accepts train/training and validation/val/dev/valid directory aliases. Native claim fields include claim_id, Claim, cleaned_truthfulness, Evidence, and ruling_outline. Repeated claim rows are grouped by ID and checked for consistent text/labels. Evidence passages receive persistent content-derived IDs. Reference explanations remain attached to claims.

Images under the root images directory and split image directories enter the corpus. Names following the claim-ID proof pattern link to annotated claim evidence. Supplementary `Corpus3_sentence_level.csv` supplies retrieval passages; `Corpus3.csv` supplies document text when sentence-level files are absent. Labels and explanation targets remain outside evidence records.

The converter preserves original partitions, rejects duplicate claim IDs across splits, verifies evidence references, and writes train/validation/test JSONL, corpus JSONL, and a preparation manifest with source hashes and counts. The source release is available through [MOCHEG](https://github.com/PLUM-Lab/Mocheg).

## Native COSMOS import

```bash
ammfv prepare-dataset --config configs/datasets/cosmos.yaml --root data/raw/cosmos --pairs-per-image 1
```

The source files are train.json, val.json, and test.json with local image paths. Source image/caption assets follow the [COSMOS release](https://github.com/shivangi-aneja/COSMOS). Annotation files may contain JSON arrays or JSONL records.

The query represents two captions associated with one image:

```text
The following two captions describe the same image in a compatible context.
Caption A: ...
Caption B: ...
```

For training and validation, the converter uses captions grouped by source image. Each configured pair index yields a same-image positive and a randomly selected cross-image negative. A single-caption positive uses the caption twice. Negatives select another source image uniformly; duplicate caption text is replaced by a distinct source caption. If no distinct negative text exists, that pair is skipped. The configured seed controls negative sampling. Pair IDs contain source image, source ordinal, pair index, and positive/negative kind. These records carry `label_source=self_supervised_source_pair` and preserve the original train/validation partitions.

Test records use caption1/caption2 and the original labels. Label 0/not-ooc becomes SUPPORTED; label 1/ooc becomes REFUTED. Test metadata carries `label_source=source_annotation`. The associated source image is annotated evidence. Test captions and test labels remain in the claim task; they are not inserted as separate retrieval passages.

COSMOS stores binary `benchmark_label` alongside canonical labels. Evaluation maps SUPPORTED→0 and REFUTED→1. NOT_ENOUGH_INFO becomes `__ABSTAIN__` and counts as an incorrect prediction. Source training captions provide retrieval text; source images provide visual evidence. The preparation manifest records pair construction, seed, label mapping, output classes, and split policy.

## Export adapters and custom datasets

`prepare --kind claims` converts exported CSV/JSON/JSONL according to `configs/adapters/`. Dot-separated field mappings access nested records. A configured records key selects arrays nested inside JSON objects. `prepare --kind corpus` creates canonical evidence rows. Native dataset conversion uses `prepare-dataset` and requires no manual field-map editing.

To add a dataset, write canonical records and a YAML profile inheriting `configs/base.yaml`. Set claims/corpus/image roots, output paths, checkpoint paths, and evaluation class mapping. For binary tasks, configure the explicit NEI policy. Custom inference rows may omit supervision and use split=unlabeled.

## Inventory and feature artifacts

`inventory` reports split counts, modality counts, corpus identity, split-ID overlap, and unresolved gold IDs. EvidenceCorpus rejects duplicate evidence IDs. Retrieval indexes record corpus and encoder identities. Frozen feature caches accept gold evidence or one/more retrieved-evidence files keyed by claim ID, reject duplicate/missing acquisition records, and record dimension, selected IDs, split labels, and encoder provenance.

No annotation/explanation field enters the public model claim context. Reference explanations are read by the explanation evaluator and the reference-aware ORACLE comparator. Gold evidence is accessed only under its selected protocol or when generating supervised feature caches.
