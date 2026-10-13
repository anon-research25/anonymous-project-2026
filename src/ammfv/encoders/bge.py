from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BGEEncoder:
    """Dense BGE text retrieval, with normalized CLS embeddings and query prefix."""

    def __init__(self, model_name="BAAI/bge-large-en-v1.5", revision="main", device="cpu",
                 batch_size=32, max_length=512,
                 query_prefix="Represent this sentence for searching relevant passages: "):
        self.model_name, self.revision = model_name, revision
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.query_prefix = query_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
        self.model.requires_grad_(False)
        self.dimension = self.model.config.hidden_size
        self.identity = {"model": model_name, "revision": revision,
                         "commit": getattr(self.model.config, "_commit_hash", None),
                         "max_length": max_length, "query_prefix": query_prefix}

    @torch.inference_mode()
    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        outputs = []
        if query:
            texts = [self.query_prefix + text for text in texts]
        for start in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                    truncation=True, max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            features = self.model(**tokens).last_hidden_state[:, 0]
            outputs.append(F.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, self.dimension), np.float32)
