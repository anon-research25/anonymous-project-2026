from __future__ import annotations

from pathlib import Path

from ..schema import Evidence
from ..utils.io import file_sha256, read_jsonl


class EvidenceCorpus:
    """A permitted collection, independent of claim labels and explanations."""

    def __init__(self, path: str | Path, image_root: str | Path = "."):
        self.path = Path(path)
        self.image_root = Path(image_root)
        self.items: dict[str, Evidence] = {}
        for record in read_jsonl(path):
            item = Evidence(**record).resolved(image_root)
            if item.evidence_id in self.items:
                raise ValueError(f"Duplicate corpus ID: {item.evidence_id}")
            self.items[item.evidence_id] = item
        self.sha256 = file_sha256(path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, identifier: str) -> Evidence:
        if identifier not in self.items:
            raise KeyError(f"Evidence {identifier!r} is not in the permitted collection")
        return self.items[identifier]

    def select(self, identifiers) -> list[Evidence]:
        return [self[identifier] for identifier in dict.fromkeys(identifiers)]

    def modality(self, modality: str) -> list[Evidence]:
        return [item for item in self.items.values() if item.modality == modality]
