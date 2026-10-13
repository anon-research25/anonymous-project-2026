"""Native MOCHEG and COSMOS conversion into canonical claims and evidence."""
from __future__ import annotations

import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from ..schema import Claim, Evidence
from ..utils.io import file_sha256, fingerprint, safe_id, write_json, write_jsonl
from .adapters import raw_records


MOCHEG_LABELS = {
    "supported": "SUPPORTED", "supports": "SUPPORTED", "true": "SUPPORTED",
    "refuted": "REFUTED", "refutes": "REFUTED", "false": "REFUTED",
    "nei": "NOT_ENOUGH_INFO", "not enough info": "NOT_ENOUGH_INFO",
    "not enough information": "NOT_ENOUGH_INFO", "not_enough_info": "NOT_ENOUGH_INFO",
}
COSMOS_LABELS = {
    "0": "0", "0.0": "0", "not-ooc": "0", "not_ooc": "0", "not ooc": "0",
    "consistent": "0", "in-context": "0", "not-out-of-context": "0",
    "1": "1", "1.0": "1", "ooc": "1", "out-of-context": "1", "inconsistent": "1",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"<br\s*/?>|</p>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def first_field(record, names, default=None):
    for name in names:
        if name in record and record[name] is not None and str(record[name]).strip():
            return record[name]
    return default


def native_id(value) -> str:
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def split_file(root, split, names):
    directories = {"train": ["train", "training"], "validation": ["validation", "val", "dev", "valid"],
                   "test": ["test", "testing"]}[split]
    for directory in directories:
        for name in names:
            candidate = root / directory / name
            if candidate.is_file():
                return candidate
    for name in names:
        for directory in directories:
            candidates = [root / f"{directory}_{name}", root / f"{directory}{Path(name).suffix}"]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(f"No {split} source found under {root}; searched {directories} / {names}")


def register_evidence(items: dict[str, Evidence], item: Evidence):
    previous = items.get(item.evidence_id)
    if previous is not None and previous.to_dict() != item.to_dict():
        raise ValueError(f"Conflicting content for evidence ID {item.evidence_id}")
    items[item.evidence_id] = item


def save_benchmark(output, dataset, claims_by_split, evidence, sources, protocol):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    evidence_ids = set(evidence)
    for split, claims in claims_by_split.items():
        for claim in claims:
            if claim.claim_id in seen:
                raise ValueError(f"Claim ID duplicated across splits: {claim.claim_id}")
            seen.add(claim.claim_id)
            unknown = set(claim.gold_evidence_ids) - evidence_ids
            if unknown:
                raise ValueError(f"Gold references absent from collection: {claim.claim_id}: {unknown}")
        write_jsonl(output / f"{split}.jsonl", (claim.to_dict() for claim in claims))
    write_jsonl(output / "corpus.jsonl", (evidence[key].to_dict() for key in sorted(evidence)))
    write_json(output / "preparation_manifest.json", {
        "dataset": dataset, "source_files": {str(path): file_sha256(path) for path in sorted(set(sources))},
        "splits": {name: len(claims) for name, claims in claims_by_split.items()},
        "corpus_items": len(evidence), "corpus_sha256": file_sha256(output / "corpus.jsonl"),
        "claim_files": {split: file_sha256(output / f"{split}.jsonl") for split in claims_by_split},
        "protocol": protocol,
    })


def prepare_mocheg(root, output):
    """Import release CSVs, proof images and the supplementary retrieval collection."""
    root = Path(root).resolve()
    evidence, grouped, sources = {}, {}, []
    for split in ("train", "validation", "test"):
        path = split_file(root, split, ["Corpus2.csv", "claims.csv", "claims.json", "claims.jsonl"])
        sources.append(path)
        claims = {}
        for row in raw_records(path):
            raw_id = first_field(row, ["claim_id", "id"])
            if raw_id is None:
                raise ValueError(f"Source claim has no ID in {path}")
            identifier = native_id(raw_id)
            label_name = clean_text(first_field(row, ["cleaned_truthfulness", "label", "truthfulness"])).lower()
            if label_name not in MOCHEG_LABELS:
                raise ValueError(f"Unknown MOCHEG truthfulness value {label_name!r}")
            text = clean_text(first_field(row, ["Claim", "claim", "text"]))
            label = MOCHEG_LABELS[label_name]
            record = claims.setdefault(identifier, {
                "text": text, "label": label, "ids": [],
                "explanation": clean_text(first_field(row, ["ruling_outline", "explanation"])),
            })
            if (record["text"], record["label"]) != (text, label):
                raise ValueError(f"Inconsistent rows for claim {identifier}")
            passage = clean_text(first_field(row, ["Evidence", "text_evidence", "evidence"]))
            if passage:
                item_id = f"mocheg:proof:{identifier}:{fingerprint(passage)[:16]}"
                item = Evidence(item_id, "text", text=passage,
                                source={"dataset": "mocheg", "record_kind": "evidence_passage"})
                register_evidence(evidence, item)
                record["ids"].append(item_id)
        grouped[split] = claims
    proof_images = defaultdict(list)
    image_directories = [root / "images"] + [path for path in root.glob("*/images") if path.is_dir()]
    for directory in image_directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            item_id = f"mocheg:image:{safe_id(relative)}"
            register_evidence(evidence, Evidence(item_id, "image", image_path=str(path),
                                                source={"dataset": "mocheg", "source_image": relative}))
            match = re.match(r"^(\d+)-proof(?:-|\.)", path.name)
            if match:
                proof_images[native_id(match.group(1))].append(item_id)
    collection_paths = sorted(set(root.glob("**/Corpus3_sentence_level.csv")))
    if not collection_paths:
        collection_paths = sorted(set(root.glob("**/Corpus3.csv")))
    for path in collection_paths:
        sources.append(path)
        for ordinal, row in enumerate(raw_records(path)):
            passage = clean_text(first_field(row, ["paragraph", "Origin Document", "text", "Evidence"]))
            if not passage:
                continue
            identity = [native_id(row.get(key, ordinal)) for key in ("claim_id", "relevant_document_id", "paragraph_id")]
            item_id = f"mocheg:collection:{':'.join(identity)}:{fingerprint(passage)[:16]}"
            register_evidence(evidence, Evidence(item_id, "text", text=passage,
                                                source={"dataset": "mocheg", "record_kind": "retrieval_passage"}))
    claims_by_split = {}
    for split, records in grouped.items():
        claims_by_split[split] = [Claim(
            claim_id=f"mocheg:claim:{identifier}", text=record["text"], split=split, dataset="mocheg",
            label=record["label"], benchmark_label=record["label"],
            gold_evidence_ids=tuple(dict.fromkeys(record["ids"] + proof_images[identifier])),
            explanation=record["explanation"] or None,
            metadata={"source_claim_id": identifier},
        ) for identifier, record in records.items()]
    save_benchmark(output, "mocheg", claims_by_split, evidence, sources,
                   {"split_policy": "source_splits", "label_space": "three_state_veracity",
                    "collection": "original_evidence_and_retrieval_passages_and_images",
                    "reference_explanations_in_collection": False})


def cosmos_claim(caption_a, caption_b) -> str:
    return ("The following two captions describe the same image in a compatible context.\n"
            f"Caption A: {caption_a}\nCaption B: {caption_b}")


def cosmos_image_path(root, value):
    path = Path(str(value))
    if path.is_absolute() and path.is_file():
        return path
    parts = path.parts
    candidates = [root / path, root / "images" / path.name]
    candidates.extend(root.joinpath(*parts[index:]) for index in range(len(parts)))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"COSMOS image not found: {value}")


def prepare_cosmos(root, output, seed=42, pairs_per_image=1):
    """Build paired-caption context claims with source-split self-supervised training."""
    if pairs_per_image < 1:
        raise ValueError("pairs_per_image must be positive")
    root = Path(root).resolve()
    rng = random.Random(seed)
    evidence, claims_by_split, sources = {}, {}, []
    for split in ("train", "validation", "test"):
        path = split_file(root, split, ["data.json", "data.jsonl", "test_data.json", "train_data.json", "val_data.json"])
        sources.append(path)
        records = list(raw_records(path))
        prepared = []
        for ordinal, row in enumerate(records):
            image_value = first_field(row, ["img_local_path", "image_path", "image"])
            if image_value is None:
                raise ValueError(f"COSMOS sample {ordinal} has no image path")
            image_path = cosmos_image_path(root, image_value)
            identity = fingerprint(str(image_value).replace("\\", "/"))[:20]
            image_id = f"cosmos:image:{identity}"
            register_evidence(evidence, Evidence(image_id, "image", image_path=str(image_path),
                                                source={"dataset": "cosmos", "record_kind": "source_image"}))
            captions = []
            for article in row.get("articles", []):
                caption = clean_text(first_field(article, ["caption", "caption_modified"]))
                if caption:
                    captions.append(caption)
                    text_id = f"cosmos:caption:{identity}:{fingerprint(caption)[:16]}"
                    item = Evidence(text_id, "text", text=caption, source={
                        "dataset": "cosmos", "record_kind": "source_caption", "image_reference": image_id})
                    register_evidence(evidence, item)
            prepared.append({"row": row, "image_id": image_id, "captions": list(dict.fromkeys(captions)),
                             "ordinal": ordinal, "identity": identity})
        claims = []
        if split == "test":
            for item in prepared:
                row = item["row"]
                caption_a = clean_text(first_field(row, ["caption1", "caption1_modified"]))
                caption_b = clean_text(first_field(row, ["caption2", "caption2_modified"]))
                raw_label = str(row.get("label", "")).strip().lower()
                if not caption_a or not caption_b or raw_label not in COSMOS_LABELS:
                    raise ValueError(f"Malformed labeled COSMOS pair at test record {item['ordinal']}")
                benchmark = COSMOS_LABELS[raw_label]
                claims.append(Claim(f"cosmos:test:{item['identity']}:{item['ordinal']}",
                                    cosmos_claim(caption_a, caption_b), split, "cosmos",
                                    label="SUPPORTED" if benchmark == "0" else "REFUTED",
                                    benchmark_label=benchmark, gold_evidence_ids=(item["image_id"],),
                                    metadata={"task": "paired_caption_context", "label_source": "source_annotation"}))
        else:
            eligible = [item for item in prepared if item["captions"]]
            if len({item["image_id"] for item in eligible}) < 2:
                raise ValueError(f"COSMOS {split} requires at least two captioned images")
            by_image = defaultdict(list)
            for item in eligible:
                by_image[item["image_id"]].extend(item["captions"])
            image_keys = sorted(by_image)
            image_positions = {key: index for index, key in enumerate(image_keys)}
            for item in eligible:
                captions = item["captions"]
                for pair_index in range(pairs_per_image):
                    anchor = captions[pair_index % len(captions)]
                    partner = captions[(pair_index + 1) % len(captions)]
                    sampled_index = rng.randrange(len(image_keys) - 1)
                    if sampled_index >= image_positions[item["image_id"]]:
                        sampled_index += 1
                    negative_image = image_keys[sampled_index]
                    negative = rng.choice(by_image[negative_image])
                    if negative == anchor:
                        # Equal captions convey no contrast; search a distinct source caption.
                        alternatives = [text for key in image_keys if key != item["image_id"]
                                        for text in by_image[key] if text != anchor]
                        if not alternatives:
                            continue
                        negative = rng.choice(alternatives)
                    for kind, second, benchmark in (("positive", partner, "0"), ("negative", negative, "1")):
                        claims.append(Claim(
                            f"cosmos:{split}:{item['identity']}:{item['ordinal']}:{pair_index}:{kind}",
                            cosmos_claim(anchor, second), split, "cosmos",
                            label="SUPPORTED" if benchmark == "0" else "REFUTED", benchmark_label=benchmark,
                            gold_evidence_ids=(item["image_id"],),
                            metadata={"task": "paired_caption_context", "label_source": "self_supervised_source_pair",
                                      "pair_kind": kind},
                        ))
        claims_by_split[split] = claims
    save_benchmark(output, "cosmos", claims_by_split, evidence, sources, {
        "task": "paired_caption_context", "split_policy": "source_splits",
        "training_pairs": "same_image_positive_cross_image_negative", "pairs_per_image": pairs_per_image,
        "seed": seed, "test_labels": "source_annotation", "output_classes": ["0", "1"],
        "canonical_mapping": {"0": "SUPPORTED", "1": "REFUTED"},
        "nei_policy": "abstain_as_error", "reference_explanations_in_collection": False,
    })
