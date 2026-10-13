from __future__ import annotations

from collections import Counter
from pathlib import Path

from torch.utils.data import Dataset

from ..schema import Claim
from ..utils.io import read_jsonl


class ClaimDataset(Dataset):
    def __init__(self, path: str | Path, split: str | None = None):
        self.path = Path(path)
        self.claims = [Claim.from_dict(record) for record in read_jsonl(path)]
        if split is not None:
            self.claims = [claim for claim in self.claims if claim.split == split]
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("Claim IDs must be unique within a dataset artifact")

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, index):
        return self.claims[index]

    def statistics(self) -> dict:
        return {"count": len(self), "labels": dict(Counter(c.label for c in self.claims)),
                "splits": dict(Counter(c.split for c in self.claims))}


def enforce_split_disjointness(paths: list[str | Path]) -> dict:
    seen_ids, seen_text = {}, {}
    duplicates = []
    for path in paths:
        for claim in ClaimDataset(path).claims:
            if claim.claim_id in seen_ids:
                raise ValueError(f"Claim ID occurs in multiple splits: {claim.claim_id}")
            seen_ids[claim.claim_id] = str(path)
            normalized = " ".join(claim.text.lower().split())
            if normalized in seen_text:
                duplicates.append({"claim_id": claim.claim_id, "other": seen_text[normalized]})
            seen_text[normalized] = claim.claim_id
    return {"unique_ids": len(seen_ids), "duplicate_texts": duplicates}
