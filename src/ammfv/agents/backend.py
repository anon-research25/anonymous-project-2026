"""A single frozen LLaVA-NeXT checkpoint shared by all generative roles."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, LlavaNextForConditionalGeneration


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    image_preprocessing: dict | None = None


class ContextBudgetError(ValueError):
    """The fully expanded multimodal request exceeds the context window."""


class LlavaBackend:
    """Greedy local inference with native image processing and exact token accounting.

    CLIP's 77-token limit applies to graph features, not to LLaVA evidence contexts.
    Image preprocessing fits the context budget when needed and records retained
    dimensions. Expanded multimodal token sequences are always kept intact.
    """

    def __init__(self, config):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config["model"], revision=config["revision"])
        self.processor.tokenizer.padding_side = "left"
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[config["dtype"]]
        load_options = {"torch_dtype": dtype, "revision": config["revision"],
                        "low_cpu_mem_usage": True}
        if config.get("device_map"):
            load_options["device_map"] = config["device_map"]
        if config.get("load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(config["model"], **load_options).eval()
        if not config.get("device_map"):
            self.model.to(config["device"])
        self.model.requires_grad_(False)
        if getattr(self.processor, "patch_size", None) is None:
            self.processor.patch_size = self.model.config.vision_config.patch_size
        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        if getattr(self.processor, "num_additional_image_tokens", None) is None:
            self.processor.num_additional_image_tokens = 1
        self.identity = {"config": config, "commit": getattr(self.model.config, "_commit_hash", None)}
        self.lock = threading.Lock()

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @torch.inference_mode()
    def generate(self, instructions: str, user: str, image_paths: list[str]) -> Generation:
        # Mistral templates may not accept a system role. Preserve role instructions
        # in the single user message, before the controller-delimited evidence data.
        content = [{"type": "text", "text": instructions + "\n\n" + user}]
        content.extend({"type": "image"} for _ in image_paths)
        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        kwargs = {"text": prompt, "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs)
        input_tokens = inputs["input_ids"].shape[-1]
        image_preprocessing = {"mode": "native", "original_sizes": [list(item.size) for item in images]}
        if images and input_tokens + self.config["max_new_tokens"] > self.config["context_tokens"]:
            # Reprocess complete images rather than truncating expanded image-token sequences.
            image_size = self.model.config.vision_config.image_size
            bounded_images = []
            for item in images:
                bounded = item.copy()
                bounded.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                bounded_images.append(bounded)
            kwargs["images"] = bounded_images
            inputs = self.processor(**kwargs)
            input_tokens = inputs["input_ids"].shape[-1]
            image_preprocessing.update(mode="context_fit", retained_sizes=[list(item.size) for item in bounded_images])
        if input_tokens + self.config["max_new_tokens"] > self.config["context_tokens"]:
            raise ContextBudgetError(f"Prompt {input_tokens} + generation {self.config['max_new_tokens']} "
                                     f"exceeds budget {self.config['context_tokens']}")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with self.lock:
            output = self.model.generate(**inputs, do_sample=False,
                                         max_new_tokens=self.config["max_new_tokens"],
                                         num_beams=1, use_cache=True)
        generated = output[0, input_tokens:]
        return Generation(self.processor.decode(generated, skip_special_tokens=True),
                          input_tokens, len(generated), image_preprocessing)
