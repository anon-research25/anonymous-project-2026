"""torchrun support for graph training and independent inference shards."""
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


def initialize() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dist.init_process_group(backend)
    return rank, world, local


def barrier():
    if dist.is_initialized():
        dist.barrier()


def sum_values(values: list[float], device: str) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()


@contextmanager
def main_process_first():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        barrier()
    yield
    if rank == 0:
        barrier()
