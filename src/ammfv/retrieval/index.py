"""Chunked exact cosine retrieval or optional FAISS inner-product indexes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils.io import write_json


class DenseIndex:
    def __init__(self, vectors: np.ndarray, identifiers: list[str], chunk_size=65536,
                 backend="numpy"):
        if vectors.ndim != 2 or len(vectors) != len(identifiers):
            raise ValueError("Index row/ID mismatch")
        self.vectors, self.identifiers = vectors, identifiers
        self.chunk_size, self.backend = chunk_size, backend
        self.faiss = None
        if backend == "faiss" and len(vectors):
            import faiss
            self.faiss = faiss.IndexFlatIP(vectors.shape[1])
            self.faiss.add(np.ascontiguousarray(vectors, dtype=np.float32))
        elif backend not in {"numpy", "faiss"}:
            raise ValueError(f"Unknown index backend: {backend}")

    def search(self, query: np.ndarray, k: int, exclude: set[str] | None = None):
        exclude = exclude or set()
        if k <= 0 or len(self.identifiers) == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        query = query / max(np.linalg.norm(query), 1e-12)
        candidates = []
        if self.faiss is not None:
            count = min(len(self.identifiers), k + len(exclude))
            scores, rows = self.faiss.search(query[None], count)
            candidates = [(int(row), float(score)) for row, score in zip(rows[0], scores[0])
                          if row >= 0 and self.identifiers[row] not in exclude]
        else:
            for start in range(0, len(self.vectors), self.chunk_size):
                scores = np.asarray(self.vectors[start:start + self.chunk_size]) @ query
                for offset, identifier in enumerate(self.identifiers[start:start + len(scores)]):
                    if identifier in exclude:
                        scores[offset] = -np.inf
                count = min(k, len(scores))
                local = np.argpartition(-scores, count - 1)[:count]
                candidates.extend((start + int(row), float(scores[row])) for row in local
                                  if np.isfinite(scores[row]))
        candidates.sort(key=lambda item: (-item[1], self.identifiers[item[0]]))
        return [(self.identifiers[row], score) for row, score in candidates[:k]]

    def save(self, directory: str | Path, metadata: dict):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        write_json(directory / "ids.json", self.identifiers)
        write_json(directory / "metadata.json", {**metadata, "rows": len(self.identifiers),
                                                 "dimension": self.vectors.shape[1]})

    @classmethod
    def load(cls, directory: str | Path, backend="numpy", expected_corpus_sha256=None):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if expected_corpus_sha256 and metadata["corpus_sha256"] != expected_corpus_sha256:
            raise ValueError("Index belongs to a different evidence collection")
        instance = cls(np.load(directory / "vectors.npy", mmap_mode="r"),
                       json.loads((directory / "ids.json").read_text(encoding="utf-8")),
                       backend=backend)
        instance.metadata = metadata
        return instance


def build_indexes(corpus, bge, clip, output, backend="numpy"):
    texts, images = corpus.modality("text"), corpus.modality("image")
    for name, items, vectors, encoder in (
        ("text", texts, bge.encode([item.text for item in texts]), bge),
        ("image", images, clip.encode_images([item.image_path for item in images]), clip),
    ):
        DenseIndex(vectors, [item.evidence_id for item in items], backend=backend).save(
            Path(output) / name, {"corpus_sha256": corpus.sha256,
                                  "encoder": encoder.identity, "modality": name})
