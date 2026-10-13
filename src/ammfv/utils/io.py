"""Atomic artifacts and content fingerprints for resumable research workflows."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    atomic_text(path, "".join(canonical_json(record) + "\n" for record in records))


def append_jsonl(path: str | Path, record: dict) -> None:
    """Single-process append; distributed workers must use distinct shard files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()


def safe_id(identifier: str) -> str:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]


def model_artifact_fingerprint(path: str | Path) -> dict:
    """Hash actual local checkpoint assets, not mutable checkpoint path strings."""
    path = Path(path)
    if path.is_file():
        return {"path": str(path), "sha256": file_sha256(path)}
    if not path.is_dir():
        return {"path": str(path), "exists": False}
    extensions = {".json", ".safetensors", ".bin", ".model", ".txt"}
    hashes = {str(file.relative_to(path)): file_sha256(file)
              for file in sorted(path.rglob("*")) if file.is_file() and file.suffix in extensions}
    return {"path": str(path), "files": hashes, "sha256": fingerprint(hashes)}
