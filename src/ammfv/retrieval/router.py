from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..schema import STRATEGIES


def success_priority_label(gold: str, predictions: dict[str, str]) -> str:
    """Paper routing rule: S succeeds -> S; else M succeeds -> M; else C.

    C includes all-strategy failures. These are strategy-relative labels,
    not independent annotations of intrinsic claim complexity.
    """
    if predictions.get("S") == gold:
        return "S"
    if predictions.get("M") == gold:
        return "M"
    return "C"


class ClaimRouter:
    def __init__(self, checkpoint: str, device="cpu", max_length=256):
        if not checkpoint:
            raise ValueError("Adaptive inference needs a trained router checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        self.device, self.max_length = device, max_length
        labels = tuple(self.model.config.id2label[index] for index in range(3))
        if labels != STRATEGIES:
            raise ValueError(f"Router checkpoint label order must be {STRATEGIES}; found {labels}")

    @torch.inference_mode()
    def predict(self, claim_text: str) -> tuple[str, dict[str, float]]:
        tokens = self.tokenizer(claim_text, truncation=True, max_length=self.max_length,
                                return_tensors="pt").to(self.device)
        probabilities = self.model(**tokens).logits.softmax(-1)[0].cpu().tolist()
        return STRATEGIES[max(range(3), key=probabilities.__getitem__)], dict(zip(STRATEGIES, probabilities))
