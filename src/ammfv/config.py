"""Composable YAML configuration with explicit, recorded command-line overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None,
                _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Resolve ``extends`` relative to the YAML file; data paths remain cwd-relative."""
    config: dict = {}
    visited: set[Path] = set()

    def layers(current, stack):
        current = Path(current).resolve()
        if current in stack:
            raise ValueError(f"Cyclic configuration inheritance: {stack + (current,)}")
        if current in visited:
            return
        with current.open(encoding="utf-8") as handle:
            own = yaml.safe_load(handle) or {}
        if not isinstance(own, dict):
            raise ValueError(f"Configuration must be a mapping: {current}")
        parents = own.pop("extends", [])
        if isinstance(parents, str):
            parents = [parents]
        for parent in parents:
            yield from layers(current.parent / parent, stack + (current,))
        visited.add(current)
        yield own

    for layer in layers(path, _stack):
        config = deep_merge(config, layer)
    for expression in overrides or []:
        name, separator, raw = expression.partition("=")
        if not separator:
            raise ValueError(f"Expected key=value, got {expression!r}")
        keys = name.split(".")
        node = config
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Override traverses a scalar: {name}")
        node[keys[-1]] = yaml.safe_load(raw)
    return _expand_environment(config)


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def validate_config(config: dict) -> None:
    """Check semantic constraints, without loading models or touching datasets."""
    if config["retrieval"]["evidence_mode"] not in {"retrieved", "gold"}:
        raise ValueError("evidence_mode must be retrieved or gold")
    if config["graph"]["neighbors"] < 1 or config["debate"]["max_rounds"] < 1:
        raise ValueError("Neighborhood and debate budgets must be positive")
    if config["graph"]["temperature"] <= 0:
        raise ValueError("Gumbel temperature must be positive")
    if config["debate"]["critique_exchanges"] < 1:
        raise ValueError("At least one standard/counterfactual exchange is required")
    if config["context"]["reference_size"] < 1:
        raise ValueError("reference_size must be positive")
    if config["context"]["max_agents"] is not None and config["context"]["max_agents"] < 1:
        raise ValueError("max_agents must be null or positive")
    if config["debate"]["confidence"] not in {"none", "agreement"}:
        raise ValueError("confidence must be none or agreement")
    if config["llm"]["do_sample"] or config["llm"]["temperature"] != 0:
        raise ValueError("Paper inference uses greedy decoding with temperature 0")
    if config["routing"]["gold_policy"] not in {"bypass", "route"}:
        raise ValueError("Record an explicit routing gold_policy")
    if config["graph"]["metric"] not in {"cosine", "weighted_cosine"}:
        raise ValueError("Unknown graph similarity metric")
    if config["retrieval"]["max_evidence"] < 1 or config["retrieval"]["max_hops"] < 1:
        raise ValueError("Retrieval evidence and hop budgets must be positive")
    if config["retrieval"].get("planner_context_size", 7) < 1:
        raise ValueError("planner_context_size must be positive")
    if config["routing"]["fixed_strategy"] not in {"S", "M", "C"}:
        raise ValueError("fixed_strategy must be S, M or C")
    for name in ("graph_training", "router_training"):
        for setting in ("batch_size", "accumulation_steps", "epochs", "logging_steps"):
            if config[name][setting] < 1:
                raise ValueError(f"{name}.{setting} must be positive")
    if config["llm"]["max_new_tokens"] >= config["llm"]["context_tokens"]:
        raise ValueError("The generation budget must leave room for prompt tokens")
