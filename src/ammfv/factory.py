"""Centralized model construction; encoder-only commands never load LLaVA."""
from __future__ import annotations

from pathlib import Path
import copy

from .agents.backend import LlavaBackend
from .agents.runner import RoleRunner
from .config import validate_config
from .data.corpus import EvidenceCorpus
from .encoders.bge import BGEEncoder
from .encoders.clip import CLIPEncoder
from .graph.model import HeterogeneousGraphVerifier
from .retrieval.controller import ExpertController
from .retrieval.experts import DirectExpert, MultiStepExpert, SingleStepExpert
from .retrieval.index import DenseIndex
from .retrieval.router import ClaimRouter
from .utils.checkpoint import load_checkpoint
from .utils.io import model_artifact_fingerprint


def make_corpus(config):
    return EvidenceCorpus(config["data"]["corpus"], config["data"]["image_root"])


def make_clip(config):
    cfg = config["encoders"]["clip"]
    return CLIPEncoder(cfg["model"], cfg["revision"], cfg["device"], cfg["batch_size"],
                       cfg["max_length"], cfg["image_size"])


def make_bge(config):
    cfg = config["encoders"]["bge"]
    return BGEEncoder(cfg["model"], cfg["revision"], cfg["device"], cfg["batch_size"],
                      cfg["max_length"], cfg["query_prefix"])


def make_runner(config):
    return RoleRunner(LlavaBackend(config["llm"]), config["agents"])


def make_pipeline(config, load_router=True, direct_only=False):
    from .pipeline import AMMFVPipeline
    if direct_only:
        config = copy.deepcopy(config)
        config["graph"]["enabled"] = False
        config["routing"].update(adaptive=False, fixed_strategy="S", gold_policy="route")
    validate_config(config)
    corpus = make_corpus(config)
    clip = None if direct_only else make_clip(config)
    runner = make_runner(config)
    experts = {"S": DirectExpert(runner)}
    if config["retrieval"]["evidence_mode"] == "retrieved" and not direct_only:
        bge = make_bge(config)
        directory = Path(config["retrieval"]["index_dir"])
        indexes = [DenseIndex.load(directory / modality, config["retrieval"]["index_backend"], corpus.sha256)
                   for modality in ("text", "image")]
        single = SingleStepExpert(corpus, *indexes, bge, clip, config["retrieval"]["text_k"],
                                  config["retrieval"]["image_k"], config["retrieval"]["use_images"])
        experts["M"] = single
        experts["C"] = MultiStepExpert(single, runner, config["retrieval"]["max_hops"],
                                       config["retrieval"]["queries_per_hop"], config["retrieval"]["max_evidence"],
                                       config["retrieval"].get("planner_context_size", 7))
    gold_bypass = (config["retrieval"]["evidence_mode"] == "gold" and config["routing"]["gold_policy"] == "bypass")
    router = (ClaimRouter(config["routing"]["checkpoint"], config["routing"]["device"],
                          config["routing"]["max_length"])
              if load_router and config["routing"]["adaptive"] and not gold_bypass and not direct_only else None)
    graph = None
    if config["graph"]["enabled"] and not direct_only:
        graph = HeterogeneousGraphVerifier(config["graph"]).to(config["graph"]["device"])
        checkpoint = config["graph"]["checkpoint"]
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("Graph inference requires a trained checkpoint; use train-graph first")
        # Runtime budgets/device/checkpoint paths can change without architecture
        # changes. State-dict loading validates parameter shapes, encoder provenance
        # prevents mismatched shared-space representations.
        state = load_checkpoint(checkpoint, graph)
        architecture_keys = ("input_dim", "hidden_dim", "layers", "metric", "heads", "meta_paths")
        trained_graph = state["config"]["graph"]
        if any(trained_graph[key] != config["graph"][key] for key in architecture_keys):
            raise ValueError("Graph checkpoint architecture differs from active graph configuration")
        trained_encoder = state["provenance"]["clip_encoder"]
        if trained_encoder != clip.identity:
            raise ValueError("Graph checkpoint encoder differs from active CLIP encoder")
        graph.eval()
    controller = ExpertController(config, corpus, experts, router)
    pipeline = AMMFVPipeline(config, corpus, clip, graph, controller, runner)
    pipeline.assets_manifest = {
        "clip": clip.identity if clip is not None else None, "llm": runner.backend.identity,
        "bge": experts["M"].bge.identity if "M" in experts else None,
        "graph": model_artifact_fingerprint(config["graph"]["checkpoint"]) if graph is not None else None,
        "router": model_artifact_fingerprint(config["routing"]["checkpoint"]) if router is not None else None,
    }
    return pipeline
