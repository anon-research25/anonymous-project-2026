from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..schema import NODE_TYPES, VERDICTS
from ..utils.io import fingerprint, read_jsonl, safe_id, write_json, write_jsonl


def encode_claim_graph(claim, evidence, clip):
    return np.concatenate((clip.encode_text([claim.text]), clip.encode_evidence(evidence)), axis=0)


def cache_graph_features(claims, corpus, clip, output, retrieval_records=None, max_evidence=32,
                         use_images=True):
    """Cache frozen encoder features; trainable graph features are computed online."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if max_evidence < 1:
        raise ValueError("max_evidence must be positive")
    paths = ([retrieval_records] if isinstance(retrieval_records, (str, Path))
             else list(retrieval_records or []))
    retrieved = {} if paths else None
    for path in paths:
        for record in read_jsonl(path):
            identifier = record["claim_id"]
            if identifier in retrieved:
                raise ValueError(f"Duplicate retrieval record: {identifier}")
            retrieved[identifier] = record
    rows = []
    seen = set()
    for claim in claims:
        if claim.claim_id in seen:
            raise ValueError(f"Duplicate cached claim: {claim.claim_id}")
        seen.add(claim.claim_id)
        if retrieved is not None and claim.claim_id not in retrieved:
            raise ValueError(f"Missing retrieval record: {claim.claim_id}")
        ids = (retrieved[claim.claim_id]["evidence_ids"] if retrieved is not None else claim.gold_evidence_ids)
        items = corpus.select(ids)
        if not use_images:
            items = [item for item in items if item.modality == "text"]
        items = items[:max_evidence]
        features = encode_claim_graph(claim, items, clip)
        if features.ndim != 2 or len(features) != len(items) + 1 or not np.isfinite(features).all():
            raise ValueError(f"Invalid encoder features for {claim.claim_id}")
        destination = output / f"{safe_id(claim.claim_id)}.npz"
        np.savez_compressed(destination, features=features,
                            node_types=np.asarray([0] + [NODE_TYPES[e.modality] for e in items], np.int64))
        rows.append({"claim_id": claim.claim_id, "split": claim.split, "label": claim.label,
                     "file": destination.name, "evidence_ids": [item.evidence_id for item in items],
                     "claim_sha256": fingerprint(claim.public_context())})
    write_jsonl(output / "manifest.jsonl", rows)
    write_json(output / "metadata.json", {"encoder": clip.identity, "corpus_sha256": corpus.sha256,
               "source": "retrieved" if retrieved is not None else "gold", "max_evidence": max_evidence,
               "records_sha256": fingerprint(rows), "count": len(rows),
               "use_images": use_images, "dimension": features.shape[-1] if rows else None,
               "retrieval_inputs": [str(Path(path).resolve()) for path in paths]})


class CachedGraphDataset(Dataset):
    def __init__(self, directory, split, use_images=True):
        self.directory = Path(directory)
        self.rows = [row for row in read_jsonl(self.directory / "manifest.jsonl") if row["split"] == split]
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        self.use_images = use_images
        if not self.rows:
            raise ValueError(f"No graph features for split {split}")
        if len({row["claim_id"] for row in self.rows}) != len(self.rows):
            raise ValueError(f"Duplicate cached claim IDs for split {split}")
        if any(row["label"] not in VERDICTS for row in self.rows):
            raise ValueError("Supervised graph training requires canonical three-state labels")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.directory / row["file"]) as arrays:
            features = torch.from_numpy(arrays["features"].copy()).float()
            types = torch.from_numpy(arrays["node_types"].copy()).long()
        if (features.ndim != 2 or types.ndim != 1 or len(features) != len(types)
                or len(types) == 0 or types[0] != 0 or not torch.isfinite(features).all()
                or ((types < 0) | (types > 2)).any()):
            raise ValueError(f"Malformed graph feature artifact: {row['file']}")
        if not self.use_images:
            keep = types != NODE_TYPES["image"]
            features, types = features[keep], types[keep]
        return {"features": features, "node_types": types, "label": VERDICTS.index(row["label"]),
                "claim_id": row["claim_id"]}


def collate_graphs(items):
    batch, count, dimension = len(items), max(len(item["features"]) for item in items), items[0]["features"].shape[-1]
    features = torch.zeros(batch, count, dimension)
    types = torch.full((batch, count), -1, dtype=torch.long)
    mask = torch.zeros(batch, count, dtype=torch.bool)
    for index, item in enumerate(items):
        length = len(item["features"])
        features[index, :length], types[index, :length] = item["features"], item["node_types"]
        mask[index, :length] = True
    return {"features": features, "node_types": types, "node_mask": mask,
            "labels": torch.tensor([item["label"] for item in items]),
            "claim_ids": [item["claim_id"] for item in items]}
