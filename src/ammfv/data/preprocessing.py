"""Evidence preparation that preserves source provenance and document offsets."""
from __future__ import annotations

import re
from dataclasses import replace

from ..schema import Claim, Evidence
from ..utils.io import fingerprint


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(document: Evidence, tokenizer, max_tokens=320, overlap_tokens=40):
    """Tokenizer windows preserve source-ID and token-offset metadata.

    The original item ID remains in source.document_id. Claim evidence references
    are updated by remap_claim_evidence after chunking and deduplication.
    """
    if document.modality != "text":
        return [document]
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be in [0,max_tokens)")
    tokens = tokenizer.encode(document.text, add_special_tokens=False)
    chunks = []
    for number, start in enumerate(range(0, len(tokens), max_tokens - overlap_tokens)):
        end = min(start + max_tokens, len(tokens))
        chunks.append(Evidence(
            evidence_id=f"{document.evidence_id}:chunk:{number:04d}", modality="text",
            text=tokenizer.decode(tokens[start:end], skip_special_tokens=True),
            source={**document.source, "document_id": document.evidence_id,
                    "token_start": start, "token_end": end,
                    "tokenizer_name": tokenizer.name_or_path}))
        if end == len(tokens):
            break
    return chunks


def deduplicate_passages(evidence):
    """Return retained items plus alias mapping; do not rewrite gold IDs silently."""
    retained, seen, aliases = [], {}, {}
    for item in evidence:
        if item.modality == "text":
            key = fingerprint(normalize_whitespace(item.text))
        else:
            # Image paths alone do not prove equal pixels; preserve distinct items.
            retained.append(item)
            continue
        if key in seen:
            aliases[item.evidence_id] = seen[key]
        else:
            seen[key] = item.evidence_id
            retained.append(item)
    return retained, aliases


def remap_claim_evidence(claim: Claim, document_chunks: dict, aliases: dict) -> Claim:
    """Expand document-level annotations into retained, deduplicated chunk IDs."""
    identifiers = []
    for document_id in claim.gold_evidence_ids:
        if document_id not in document_chunks:
            raise ValueError(f"Claim {claim.claim_id} references absent document {document_id}")
        identifiers.extend(aliases.get(identifier, identifier)
                           for identifier in document_chunks[document_id])
    return replace(claim, gold_evidence_ids=tuple(dict.fromkeys(identifiers)))
