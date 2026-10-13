"""Local metrics and complete run manifests; remote logging is opt-in externally."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from .io import append_jsonl, fingerprint, write_json
from .reproducibility import environment_record


def setup_logger(output: str | Path) -> logging.Logger:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ammfv.{output.resolve()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(output / "run.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


class MetricWriter:
    def __init__(self, output: str | Path, tensorboard: bool = False):
        self.output = Path(output)
        self.tensorboard = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard = SummaryWriter(str(self.output / "tensorboard"))

    def log(self, step: int, metrics: dict) -> None:
        append_jsonl(self.output / "metrics.jsonl", {"step": step, **metrics})
        if self.tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard.add_scalar(key, value, step)

    def close(self):
        if self.tensorboard:
            self.tensorboard.close()


def save_run_manifest(output: str | Path, config: dict, command: str, inputs: dict | None = None):
    write_json(Path(output) / "run_manifest.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "config": config, "config_sha256": fingerprint(config),
        "environment": environment_record(), "inputs": inputs or {},
        "method": "AMMFV", "manifest_version": 1,
    })
