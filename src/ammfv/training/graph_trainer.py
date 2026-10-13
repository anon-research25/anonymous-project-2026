"""Mini-batch graph optimization with AMP, accumulation, DDP and exact dev totals."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ..data.features import CachedGraphDataset, collate_graphs
from ..graph.model import HeterogeneousGraphVerifier
from ..utils.checkpoint import capture_rng_state, load_checkpoint, save_checkpoint
from ..utils.distributed import barrier, initialize, sum_values
from ..utils.io import fingerprint, write_json
from ..utils.logging import MetricWriter, save_run_manifest, setup_logger
from ..utils.reproducibility import seed_everything
from .losses import SupervisedVeracityLoss
from .schedules import annealed_temperature, cosine_warmup


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.inference_mode()
def evaluate_graph(model, loader, device, classes=3):
    """DDP validation uses unwrapped model and disjoint unpadded rank subsets."""
    model.eval()
    confusion = torch.zeros(classes, classes, dtype=torch.float64, device=device)
    loss_sum, count = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["features"], batch["node_types"], batch["node_mask"])
        loss = torch.nn.functional.cross_entropy(output["logits"], batch["labels"], reduction="sum")
        prediction = output["logits"].argmax(-1)
        cells = torch.bincount(batch["labels"] * classes + prediction, minlength=classes * classes)
        confusion += cells.reshape(classes, classes)
        loss_sum += loss.item()
        count += len(prediction)
    if dist.is_initialized():
        dist.all_reduce(confusion)
    loss_sum, count = sum_values([loss_sum, count], device)
    tp = confusion.diag()
    precision = tp / confusion.sum(0).clamp_min(1)
    recall = tp / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"validation/loss": loss_sum / max(count, 1),
            "validation/accuracy": tp.sum().item() / max(count, 1),
            "validation/macro_f1": f1.mean().item()}


def train_graph(config, features_dir, resume=None):
    rank, world, local = initialize()
    cfg = config["graph_training"]
    device = f"cuda:{local}" if torch.cuda.is_available() and cfg["device"].startswith("cuda") else "cpu"
    seed_everything(config["seed"] + rank, config["deterministic"])
    output_dir = Path(cfg["output_dir"])
    logger = setup_logger(output_dir / f"rank_{rank}")
    train_data = CachedGraphDataset(features_dir, "train", config["retrieval"]["use_images"])
    dev_data = CachedGraphDataset(features_dir, "validation", config["retrieval"]["use_images"])
    metadata_hash = fingerprint(train_data.metadata)
    if train_data.metadata.get("dimension", config["graph"]["input_dim"]) != config["graph"]["input_dim"]:
        raise ValueError("Cached CLIP dimension does not match graph.input_dim")
    sampler = DistributedSampler(train_data, world, rank, shuffle=True, seed=config["seed"]) if world > 1 else None
    loader = DataLoader(train_data, batch_size=cfg["batch_size"], sampler=sampler,
                        shuffle=sampler is None, num_workers=cfg["workers"], collate_fn=collate_graphs)
    # DistributedSampler pads validation sets; use disjoint slices instead.
    dev_loader = DataLoader(Subset(dev_data, list(range(rank, len(dev_data), world))),
                            batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_graphs,
                            num_workers=cfg["workers"])
    model = HeterogeneousGraphVerifier(config["graph"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = math.ceil(len(loader) / cfg["accumulation_steps"]) * cfg["epochs"]
    scheduler = cosine_warmup(optimizer, total_steps, cfg["warmup_ratio"])
    use_amp = device.startswith("cuda") and cfg["precision"] != "float32"
    amp_dtype = torch.float16 if cfg["precision"] == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    start_epoch, step, best = 0, 0, -1.0
    if resume:
        saved = load_checkpoint(resume, model, optimizer, scheduler, scaler, True, config["graph"])
        if saved["provenance"]["feature_metadata_sha256"] != metadata_hash:
            raise ValueError("Cannot resume graph training with different cached features")
        start_epoch, step, best = saved["epoch"] + 1, saved["step"], saved["best_metric"]
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local] if device.startswith("cuda") else None)
    module = model.module if hasattr(model, "module") else model
    criterion = SupervisedVeracityLoss()
    writer = MetricWriter(output_dir, cfg["tensorboard"]) if rank == 0 else None
    if rank == 0:
        save_run_manifest(output_dir, config, "train-graph", {"features": train_data.metadata})
    for epoch in range(start_epoch, cfg["epochs"]):
        if sampler:
            sampler.set_epoch(epoch)
        module.set_sampling_temperature(annealed_temperature(
            epoch, config["graph"]["temperature"], cfg["min_temperature"], cfg["temperature_decay"]))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            group_start = index // cfg["accumulation_steps"] * cfg["accumulation_steps"]
            actual_group_size = min(cfg["accumulation_steps"], len(loader) - group_start)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
                prediction = model(batch["features"], batch["node_types"], batch["node_mask"])
                raw_loss = criterion(prediction["logits"], batch["labels"])
                loss = raw_loss / actual_group_size
            scaler.scale(loss).backward()
            if (index + 1) % cfg["accumulation_steps"] == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = scaler.get_scale() >= old_scale
                if updated:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += int(updated)
                if writer and step % cfg["logging_steps"] == 0:
                    writer.log(step, {"train/loss": raw_loss.item(), "train/grad_norm": float(norm),
                                      "train/lr": scheduler.get_last_lr()[0], "epoch": epoch})
        metrics = evaluate_graph(module, dev_loader, device)
        rank_rng_states = [capture_rng_state()]
        if dist.is_initialized():
            rank_rng_states = [None] * world
            dist.all_gather_object(rank_rng_states, capture_rng_state())
        if rank == 0:
            writer.log(step, metrics)
            logger.info("epoch=%s metrics=%s", epoch, metrics)
            improved = metrics["validation/macro_f1"] > best
            best = max(best, metrics["validation/macro_f1"])
            arguments = {"optimizer": optimizer, "scheduler": scheduler, "scaler": scaler,
                         "epoch": epoch, "step": step, "config": config, "best_metric": best,
                         "rank_rng_states": rank_rng_states,
                         "provenance": {"feature_metadata_sha256": metadata_hash,
                                        "clip_encoder": train_data.metadata["encoder"]}}
            save_checkpoint(output_dir / "last.pt", model, **arguments)
            if improved:
                save_checkpoint(output_dir / "best.pt", model, **arguments)
                write_json(output_dir / "best_metrics.json", metrics)
        barrier()
    if writer:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()
