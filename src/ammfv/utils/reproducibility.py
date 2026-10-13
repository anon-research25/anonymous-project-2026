from __future__ import annotations

import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic


def environment_record() -> dict:
    packages = {}
    for name in ("torch", "transformers", "numpy", "accelerate", "sacrebleu", "rouge-score"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def claim_seed(seed: int, identifier: str) -> int:
    from .io import fingerprint
    return (seed + int(fingerprint(identifier)[:8], 16)) % (2**31)
