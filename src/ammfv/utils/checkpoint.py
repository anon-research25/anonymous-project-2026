"""Graph checkpoints contain model, optimizer, scheduler, RNG and provenance."""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .io import fingerprint, write_json


def capture_rng_state():
    """Capture local Python, NumPy, CPU and CUDA random generators."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    epoch=0, step=0, config=None, best_metric=None, provenance=None,
                    rank_rng_states=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    state = {"format_version": 1, "model": module.state_dict(),
             "optimizer": optimizer.state_dict() if optimizer else None,
             "scheduler": scheduler.state_dict() if scheduler else None,
             "scaler": scaler.state_dict() if scaler else None,
             "epoch": epoch, "step": step, "config": config,
             "best_metric": best_metric, "provenance": provenance,
             "rng": capture_rng_state(), "rank_rng_states": rank_rng_states}
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    write_json(path.with_suffix(".json"), {"epoch": epoch, "step": step,
               "config_sha256": fingerprint(config), "best_metric": best_metric,
               "provenance": provenance})


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    restore_rng=False, expected_graph_config=None):
    # Resume artifacts include Python/NumPy RNG objects. Only load your trusted files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if expected_graph_config and state["config"]["graph"] != expected_graph_config:
        raise ValueError("Checkpoint graph architecture/config differs from requested config")
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(state["model"], strict=True)
    for name, component in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if component is not None and state.get(name) is not None:
            component.load_state_dict(state[name])
    if restore_rng:
        rng = state["rng"]
        if state.get("rank_rng_states"):
            from torch import distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            world = dist.get_world_size() if dist.is_initialized() else 1
            if len(state["rank_rng_states"]) != world:
                raise ValueError("Checkpoint RNG state requires the original distributed world size")
            rng = state["rank_rng_states"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return state
