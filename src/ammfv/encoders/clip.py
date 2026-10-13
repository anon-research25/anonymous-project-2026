from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from ..schema import Evidence


class CLIPEncoder:
    """CLIP ViT-L/14 features; evidence text capped at 77 CLIP tokens.

    Image resizing implements the paper's 224x224 encoder preprocessing. Agent
    image contexts are handled separately and retain their original resolution.
    Captions remain text metadata and never substitute for an image embedding.
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14", revision="main",
                 device="cpu", batch_size=32, max_length=77, image_size=224):
        self.model_name, self.revision = model_name, revision
        self.device, self.batch_size = device, batch_size
        self.max_length, self.image_size = max_length, image_size
        self.processor = CLIPProcessor.from_pretrained(model_name, revision=revision)
        self.model = CLIPModel.from_pretrained(model_name, revision=revision).to(device).eval()
        self.model.requires_grad_(False)
        self.dimension = self.model.config.projection_dim
        self.identity = {"model": model_name, "revision": revision,
                         "commit": getattr(self.model.config, "_commit_hash", None),
                         "max_length": max_length, "image_size": image_size}

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.processor(text=texts[start:start + self.batch_size],
                                    padding=True, truncation=True, max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            features = self.model.get_text_features(**inputs)
            outputs.append(F.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, self.dimension), np.float32)

    @torch.inference_mode()
    def encode_images(self, paths: list[str | Path]) -> np.ndarray:
        outputs = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            for path in paths[start:start + self.batch_size]:
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    images.append(rgb.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC))
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            outputs.append(F.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, self.dimension), np.float32)

    def encode_evidence(self, evidence: list[Evidence]) -> np.ndarray:
        result = np.empty((len(evidence), self.dimension), dtype=np.float32)
        for modality in ("text", "image"):
            positions = [index for index, item in enumerate(evidence) if item.modality == modality]
            if positions:
                values = ([evidence[i].text for i in positions] if modality == "text"
                          else [evidence[i].image_path for i in positions])
                result[positions] = (self.encode_text(values) if modality == "text"
                                     else self.encode_images(values))
        return result
