import argparse
from pathlib import Path

from transformers import AutoTokenizer

from ammfv.data.corpus import EvidenceCorpus
from ammfv.data.dataset import ClaimDataset
from ammfv.data.preprocessing import chunk_document, deduplicate_passages, remap_claim_evidence
from ammfv.utils.io import write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--claims", nargs="+", default=[], help="Canonical claim files to remap")
    parser.add_argument("--claims-output-dir", help="Remapped split output directory")
    args = parser.parse_args()
    corpus = EvidenceCorpus(args.input, args.image_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows, document_chunks = [], {}
    for item in corpus.items.values():
        chunks = chunk_document(item, tokenizer, args.max_tokens, args.overlap)
        rows.extend(chunks)
        document_chunks[item.evidence_id] = [chunk.evidence_id for chunk in chunks]
    aliases = {}
    if args.deduplicate:
        rows, aliases = deduplicate_passages(rows)
    write_jsonl(args.output, (item.to_dict() for item in rows))
    claim_outputs = {}
    target = Path(args.claims_output_dir) if args.claims_output_dir else Path(args.output).parent / "claims"
    destinations = set()
    for path in args.claims:
        destination = target / Path(path).name
        if destination in destinations:
            raise ValueError(f"Claim files have duplicate output names: {destination}")
        destinations.add(destination)
        claims = [remap_claim_evidence(claim, document_chunks, aliases)
                  for claim in ClaimDataset(path).claims]
        write_jsonl(destination, (claim.to_dict() for claim in claims))
        claim_outputs[path] = str(destination)
    write_json(args.output + ".mapping.json", {"document_chunks": document_chunks, "duplicate_aliases": aliases,
                                              "remapped_claim_files": claim_outputs})


if __name__ == "__main__":
    main()
