"""Persistent evidence identities and canonical three-state verification records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class Strategy(str, Enum):
    SIMPLE = "S"
    MODERATE = "M"
    COMPLEX = "C"


VERDICTS = tuple(item.value for item in Verdict)
STRATEGIES = tuple(item.value for item in Strategy)
NODE_TYPES = {"claim": 0, "text": 1, "image": 2}


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    modality: str
    text: str = ""
    image_path: str | None = None
    caption: str = ""
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.evidence_id or self.evidence_id.startswith("CF:"):
            raise ValueError("Observed evidence requires a non-counterfactual ID")
        if self.modality not in {"text", "image"}:
            raise ValueError(f"Unsupported modality {self.modality}")
        if self.modality == "text" and not self.text.strip():
            raise ValueError(f"Empty passage: {self.evidence_id}")
        if self.modality == "image" and not self.image_path:
            raise ValueError(f"Missing image path: {self.evidence_id}")

    def resolved(self, image_root: str | Path) -> Evidence:
        record = self.to_dict()
        if self.image_path and not Path(self.image_path).is_absolute():
            record["image_path"] = str((Path(image_root) / self.image_path).resolve())
        return Evidence(**record)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    split: str
    dataset: str
    label: str | None = None
    gold_evidence_ids: tuple[str, ...] = ()
    explanation: str | None = None
    benchmark_label: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.claim_id or not self.text.strip():
            raise ValueError("Claim ID and text must be nonempty")
        if self.label is not None and self.label not in VERDICTS:
            raise ValueError(f"Noncanonical claim label: {self.label}")
        if self.split not in {"train", "validation", "test", "unlabeled"}:
            raise ValueError(f"Unknown split: {self.split}")

    @classmethod
    def from_dict(cls, record: dict) -> Claim:
        record = dict(record)
        record["gold_evidence_ids"] = tuple(record.get("gold_evidence_ids", []))
        return cls(**record)

    def public_context(self) -> dict:
        """The only claim fields visible to models at inference; excludes annotations."""
        return {"claim_id": self.claim_id, "text": self.text}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    strategy: str
    evidence: list[Evidence]
    scores: dict[str, float] = field(default_factory=dict)
    hops: list[dict] = field(default_factory=list)
    direct_prediction: dict | None = None


@dataclass
class VerificationResult:
    claim_id: str
    verdict: str
    explanation: str
    strategy: str
    evidence_ids: list[str]
    pre_debate_verdict: str | None = None
    initial_assessments: list[dict] = field(default_factory=list)
    debate_invoked: bool = False
    rounds: int = 0
    termination_reason: str = ""
    consensus_reached: bool = False
    epistemic_mode: str = "evidence_grounded"
    confidence: float | None = None
    limitations: list[str] = field(default_factory=list)
    trace_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
