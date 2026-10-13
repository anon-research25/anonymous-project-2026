from __future__ import annotations

from ..utils.io import canonical_json


class EvidenceSerializer:
    """Preserve provenance and distinguish direct images from supplied captions."""

    def __init__(self, tokenizer, evidence_text_tokens=320, caption_tokens=100):
        self.tokenizer = tokenizer
        self.text_budget, self.caption_budget = evidence_text_tokens, caption_tokens

    def truncate(self, text, budget):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.decode(tokens[:budget], skip_special_tokens=True), {
            "original_tokens": len(tokens), "retained_tokens": min(len(tokens), budget),
            "truncated": len(tokens) > budget}

    def serialize(self, evidence):
        records, images, preprocessing = [], [], {}
        for item in evidence:
            record = {"evidence_id": item.evidence_id, "modality": item.modality,
                      "source_metadata": item.source}
            if item.modality == "text":
                record["observed_passage"], preprocessing[item.evidence_id] = self.truncate(item.text, self.text_budget)
            else:
                record["image_attachment_index"] = len(images)
                record["supplied_caption_not_direct_visual_observation"], preprocessing[item.evidence_id] = self.truncate(
                    item.caption, self.caption_budget)
                images.append(item.image_path)
            records.append(record)
        return records, images, preprocessing

    def request(self, context, evidence):
        records, images, preprocessing = self.serialize(evidence)
        user = ("CONTROLLER_CONTEXT_BEGIN\n" + canonical_json(context) + "\nCONTROLLER_CONTEXT_END\n"
                "OBSERVED_EVIDENCE_DATA_BEGIN\n" + canonical_json(records) + "\nOBSERVED_EVIDENCE_DATA_END\n"
                "Image attachments follow in their zero-based attachment-index order. "
                "Return only the requested JSON object.")
        return user, images, preprocessing
