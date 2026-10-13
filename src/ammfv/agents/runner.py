from __future__ import annotations

import json
from pathlib import Path

from ..utils.io import file_sha256, fingerprint, write_json
from .contracts import ROLE_FIELDS, parse_json_object, validate_role
from .serialization import EvidenceSerializer


class AgentOutputError(RuntimeError):
    """A generated role response fails its structured output contract."""


class RoleRunner:
    """Versioned prompts, shared backbone, bounded formatting retries and call logs."""

    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.prompt_dir = Path(__file__).parent / "prompts" / config["prompt_version"]
        self.serializer = EvidenceSerializer(backend.tokenizer, config["evidence_text_tokens"], config["caption_tokens"])
        self.calls: list[dict] = []
        self.image_hashes = {}
        self.cache = Path(config["cache_dir"]) if config.get("cache_dir") else None

    def reset_trace(self):
        self.calls = []

    def prompt_manifest(self):
        return {path.name: file_sha256(path) for path in self.prompt_dir.glob("*.txt")}

    def call(self, role, context, evidence):
        common = (self.prompt_dir / "common.txt").read_text(encoding="utf-8")
        instructions = common + "\n\n" + (self.prompt_dir / f"{role}.txt").read_text(encoding="utf-8")
        fields = {key: kind.__name__ for key, kind in ROLE_FIELDS[role].items()}
        instructions += "\nRequired JSON field types: " + json.dumps(fields) + "."
        user, images, preprocessing = self.serializer.request(context, evidence)
        for path in images:
            if path not in self.image_hashes:
                self.image_hashes[path] = file_sha256(path)
        key = fingerprint({"backend": self.backend.identity, "instructions": instructions,
                           "user": user, "images": [self.image_hashes[path] for path in images]})
        cache_path = self.cache / f"{key}.json" if self.cache else None
        permitted = {item.evidence_id for item in evidence}
        request_record = {"role": role, "context": context, "user_message": user,
                          "prompt_sha256": fingerprint(instructions), "cache_key": key,
                          "evidence_ids": sorted(permitted), "image_paths": images,
                          "serialization": preprocessing}
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            result = validate_role(role, cached["result"], permitted, context)
            self.calls.append({**request_record, "cached": True, **cached})
            return result
        attempts = []
        repair = ""
        for attempt in range(self.config["format_retries"] + 1):
            try:
                generation = self.backend.generate(instructions, user + repair, images)
            except Exception as error:
                self.calls.append({**request_record, "cached": False, "attempts": attempts,
                                   "failed": True, "generation_error_type": type(error).__name__,
                                   "generation_error": str(error)})
                raise
            attempt_record = {"attempt": attempt, "raw_output": generation.text,
                              "input_tokens": generation.input_tokens, "output_tokens": generation.output_tokens,
                              "image_preprocessing": generation.image_preprocessing}
            try:
                result = validate_role(role, parse_json_object(generation.text), permitted, context)
            except (ValueError, TypeError, KeyError) as error:
                attempt_record["contract_error"] = str(error)
                attempts.append(attempt_record)
                repair = ("\nFORMAT_REPAIR_INSTRUCTION: Your previous answer failed the output contract: "
                          + str(error) + ". Re-assess the same supplied context and emit valid JSON. "
                          "No new evidence or factual premises are provided by this repair instruction.")
                continue
            attempts.append(attempt_record)
            completed = {"result": result, "attempts": attempts}
            self.calls.append({**request_record, "cached": False, **completed})
            if cache_path:
                write_json(cache_path, completed)
            return result
        self.calls.append({**request_record, "cached": False, "attempts": attempts, "failed": True})
        raise AgentOutputError(f"{role} failed structured output contract after {len(attempts)} attempts")
