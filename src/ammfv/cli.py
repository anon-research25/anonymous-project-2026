"""All data, training, inference, evaluation and analysis entry points."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .utils.io import append_jsonl, read_jsonl, write_json, write_jsonl


def configuration(args):
    return load_config(args.config, args.overrides)


def claims_for(config, split=None, input_path=None):
    from .data.dataset import ClaimDataset
    split = split or config["data"]["split"]
    path = input_path or config["data"]["claims"][split]
    return ClaimDataset(path, split=None if input_path else split)


def command_prepare(args):
    from .data.adapters import prepare_claims, prepare_corpus
    import yaml
    adapter = yaml.safe_load(Path(args.adapter).read_text(encoding="utf-8"))
    if args.kind == "claims":
        prepare_claims(args.input, args.output, adapter["dataset"], args.split,
                       adapter["claims"]["fields"], adapter["label_mapping"], adapter["claims"].get("records_key"))
    else:
        prepare_corpus(args.input, args.output, adapter["corpus"]["fields"], adapter["corpus"].get("records_key"))


def command_prepare_dataset(args):
    """Convert a native benchmark release into all canonical dataset artifacts."""
    from .data.benchmarks import prepare_cosmos, prepare_mocheg
    config = configuration(args)
    output = args.output or str(Path(config["data"]["corpus"]).parent)
    if config["data"]["dataset"] == "mocheg":
        prepare_mocheg(args.root, output)
    elif config["data"]["dataset"] == "cosmos":
        prepare_cosmos(args.root, output, config["seed"], args.pairs_per_image)
    else:
        raise ValueError("Native conversion supports MOCHEG and COSMOS")


def command_inventory(args):
    from .data.dataset import enforce_split_disjointness
    from .factory import make_corpus
    config = configuration(args)
    corpus = make_corpus(config)
    datasets = {split: claims_for(config, split) for split in ("train", "validation", "test")}
    report = {"datasets": {split: dataset.statistics() for split, dataset in datasets.items()},
              "splits": enforce_split_disjointness(list(config["data"]["claims"].values())),
              "corpus": {"size": len(corpus), "sha256": corpus.sha256,
                         "text": len(corpus.modality("text")), "image": len(corpus.modality("image"))}}
    missing = []
    for dataset in datasets.values():
        for claim in dataset.claims:
            missing.extend({"claim_id": claim.claim_id, "evidence_id": identifier}
                           for identifier in claim.gold_evidence_ids if identifier not in corpus.items)
    report["missing_gold_ids"] = missing
    write_json(args.output, report)


def command_build_index(args):
    from .factory import make_bge, make_clip, make_corpus
    from .retrieval.index import build_indexes
    config = configuration(args)
    build_indexes(make_corpus(config), make_bge(config), make_clip(config),
                  config["retrieval"]["index_dir"], config["retrieval"]["index_backend"])


def command_features(args):
    from .data.features import cache_graph_features
    from .factory import make_clip, make_corpus
    config = configuration(args)
    corpus, clip = make_corpus(config), make_clip(config)
    claims = [claim for split in args.splits for claim in claims_for(config, split).claims]
    cache_graph_features(claims, corpus, clip, args.output or config["data"]["features_dir"],
                         args.retrieval_records, config["retrieval"]["max_evidence"],
                         config["retrieval"]["use_images"])


def command_retrieve(args):
    from .factory import make_pipeline
    config = configuration(args)
    config["graph"]["enabled"] = False
    pipeline = make_pipeline(config, load_router=args.strategy is None, direct_only=args.strategy == "S")
    rows = []
    for claim in claims_for(config, args.split).claims:
        pipeline.runner.reset_trace()
        acquired = pipeline.controller.acquire(claim, args.strategy)
        rows.append({"claim_id": claim.claim_id, "strategy": acquired.strategy,
                     "evidence_ids": [e.evidence_id for e in acquired.evidence],
                     "scores": acquired.scores, "hops": acquired.hops,
                     "direct_prediction": acquired.direct_prediction})
    write_jsonl(args.output, rows)


def command_router_labels(args):
    from .factory import make_pipeline
    from .training.router_labels import generate_router_labels
    config = configuration(args)
    pipeline = make_pipeline(config, load_router=False)
    generate_router_labels(pipeline, claims_for(config, "train").claims, args.output, args.resume)


def command_train_router(args):
    from .training.router_trainer import train_router
    train_router(configuration(args), args.labels, args.resume)


def command_train_graph(args):
    from .training.graph_trainer import train_graph
    config = configuration(args)
    train_graph(config, args.features or config["data"]["features_dir"], args.resume)


def command_infer(args):
    from .factory import make_pipeline
    from .inference import run_inference
    from .utils.reproducibility import seed_everything
    config = configuration(args)
    seed_everything(config["seed"], config["deterministic"])
    shard = args.shard_index if args.shard_index is not None else int(os.environ.get("RANK", 0))
    count = args.shard_count if args.shard_count is not None else int(os.environ.get("WORLD_SIZE", 1))
    if "LOCAL_RANK" in os.environ:
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        config["llm"]["device"], config["graph"]["device"], config["routing"]["device"] = device, device, device
        config["llm"]["device_map"] = None
        for settings in config["encoders"].values():
            if settings["device"].startswith("cuda"):
                settings["device"] = device
    pipeline = make_pipeline(config, load_router=args.strategy is None, direct_only=args.strategy == "S")
    path = run_inference(pipeline, claims_for(config, args.split, args.input), args.resume,
                         shard, count, args.strategy)
    print(path)


def command_merge(args):
    from .inference import merge_shards
    expected = None
    if args.claims:
        from .data.dataset import ClaimDataset
        expected = [claim.claim_id for claim in ClaimDataset(args.claims).claims]
    merge_shards(args.inputs, args.output, expected)


def command_evaluate(args):
    from .evaluation.evaluator import evaluate_predictions
    config = configuration(args)
    metrics = evaluate_predictions(claims_for(config, args.split), args.predictions,
                                   config["evaluation"], args.output)
    print(json.dumps(metrics["classification"], indent=2))


def command_retrieval_metrics(args):
    from .evaluation.retrieval import retrieval_metrics
    config = configuration(args)
    write_json(args.output, retrieval_metrics(claims_for(config, args.split).claims, list(read_jsonl(args.records))))


def command_analyze(args):
    from .analysis.traces import summarize_predictions
    summarize_predictions(read_jsonl(args.predictions), args.output)


def command_trace(args):
    from .analysis.traces import export_trace_html
    export_trace_html(args.trace, args.output)


def command_plot(args):
    from .analysis.plots import draw_graph, plot_sensitivity, plot_training
    if args.kind == "graph":
        draw_graph(args.input, args.output)
    elif args.kind == "training":
        plot_training(args.input, args.output)
    else:
        plot_sensitivity(list(read_jsonl(args.input)), args.parameter, args.metric, args.output)


def command_tables(args):
    from .analysis.tables import collect_metric_files
    collect_metric_files(args.inputs, args.output, args.latex_output)


def command_baseline(args):
    from .baselines.llava import LlavaVerifier
    from .baselines.summarization import greedy_rouge_oracle, lead3
    from .factory import make_corpus, make_runner
    config = configuration(args)
    corpus = make_corpus(config)
    acquisition = {r["claim_id"]: r for r in read_jsonl(args.evidence_records)}
    runner = make_runner(config) if args.method in {"plain", "pe", "icl1"} else None
    demonstration = json.loads(Path(args.demonstration).read_text(encoding="utf-8")) if args.demonstration else None
    verifier = LlavaVerifier(runner, args.method, demonstration) if runner else None
    if Path(args.output).exists():
        raise FileExistsError("Select a fresh baseline output artifact")
    for claim in claims_for(config, args.split).claims:
        record = acquisition[claim.claim_id]
        evidence = corpus.select(record["evidence_ids"])
        if not config["retrieval"]["use_images"]:
            evidence = [e for e in evidence if e.modality == "text"]
        if verifier:
            row = {**verifier.verify(claim, evidence).to_dict(), "status": "ok"}
        else:
            if args.method == "oracle" and claim.explanation is None:
                raise ValueError("ORACLE requires a reference explanation")
            explanation = lead3(evidence) if args.method == "lead3" else greedy_rouge_oracle(evidence, claim.explanation)
            row = {"claim_id": claim.claim_id, "explanation": explanation, "method": args.method,
                   "reference_used": args.method == "oracle", "task": "explanation_only"}
        append_jsonl(args.output, row)


def command_explanation_eval(args):
    from .evaluation.explanations import explanation_metrics
    config = configuration(args)
    rows = {r["claim_id"]: r for r in read_jsonl(args.predictions)}
    claims = [c for c in claims_for(config, args.split).claims if c.explanation is not None]
    hypotheses = [rows.get(c.claim_id, {}).get("explanation", "") for c in claims]
    write_json(args.output, explanation_metrics(hypotheses, [c.explanation for c in claims],
                                               config["evaluation"]["rouge_stemmer"], config["evaluation"]["bleu_tokenizer"]))


def command_export_external(args):
    from .baselines.external import export_evidence_bundle
    from .factory import make_corpus
    config = configuration(args)
    export_evidence_bundle(claims_for(config, args.split).claims, make_corpus(config),
                           list(read_jsonl(args.evidence_records)), args.output, args.method)


def command_import_external(args):
    from .baselines.external import import_external_predictions
    import yaml
    mapping = yaml.safe_load(Path(args.mapping).read_text(encoding="utf-8"))
    import_external_predictions(args.input, args.output, args.method, mapping,
                                [c.claim_id for c in claims_for(configuration(args), args.split).claims])


def build_parser():
    parser = argparse.ArgumentParser(prog="ammfv", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, function, configured=True):
        command = commands.add_parser(name, help=function.__doc__)
        command.set_defaults(function=function)
        if configured:
            command.add_argument("--config", default="configs/datasets/mocheg.yaml")
            command.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
        return command

    def split(command):
        command.add_argument("--split", choices=["train", "validation", "test", "unlabeled"], default=None)

    prepare = add("prepare", command_prepare, False)
    prepare.add_argument("--kind", choices=["claims", "corpus"], required=True)
    prepare.add_argument("--adapter", required=True)
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split", choices=["train", "validation", "test", "unlabeled"], default="train")
    prepare_dataset = add("prepare-dataset", command_prepare_dataset)
    prepare_dataset.add_argument("--root", required=True)
    prepare_dataset.add_argument("--output")
    prepare_dataset.add_argument("--pairs-per-image", type=int, default=1)
    inventory = add("inventory", command_inventory)
    inventory.add_argument("--output", required=True)
    add("build-index", command_build_index)
    features = add("cache-features", command_features)
    features.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation"])
    features.add_argument("--output")
    features.add_argument("--retrieval-records", nargs="+", help="One or more train/dev acquisition JSONL files")
    retrieve = add("retrieve", command_retrieve)
    split(retrieve)
    retrieve.add_argument("--strategy", choices=["S", "M", "C"])
    retrieve.add_argument("--output", required=True)
    labels = add("router-labels", command_router_labels)
    labels.add_argument("--output", required=True)
    labels.add_argument("--resume", action="store_true")
    train_router = add("train-router", command_train_router)
    train_router.add_argument("--labels", required=True)
    train_router.add_argument("--resume")
    train_graph = add("train-graph", command_train_graph)
    train_graph.add_argument("--features")
    train_graph.add_argument("--resume")
    infer = add("infer", command_infer)
    split(infer)
    infer.add_argument("--input", help="Canonical JSONL, optionally unlabeled")
    infer.add_argument("--strategy", choices=["S", "M", "C"])
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--shard-index", type=int)
    infer.add_argument("--shard-count", type=int)
    merge = add("merge-shards", command_merge, False)
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--claims")
    merge.add_argument("--output", required=True)
    evaluate = add("evaluate", command_evaluate)
    split(evaluate)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    retrieval_metrics = add("retrieval-metrics", command_retrieval_metrics)
    split(retrieval_metrics)
    retrieval_metrics.add_argument("--records", required=True)
    retrieval_metrics.add_argument("--output", required=True)
    analyze = add("analyze", command_analyze, False)
    analyze.add_argument("--predictions", required=True)
    analyze.add_argument("--output", required=True)
    trace = add("trace-html", command_trace, False)
    trace.add_argument("--trace", required=True)
    trace.add_argument("--output", required=True)
    plot = add("plot", command_plot, False)
    plot.add_argument("--kind", choices=["graph", "training", "sensitivity"], required=True)
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--parameter", default="reference_size")
    plot.add_argument("--metric", default="label_accuracy")
    tables = add("tables", command_tables, False)
    tables.add_argument("--inputs", nargs="+", required=True)
    tables.add_argument("--output", required=True)
    tables.add_argument("--latex-output")
    baseline = add("baseline", command_baseline)
    split(baseline)
    baseline.add_argument("--method", choices=["plain", "pe", "icl1", "lead3", "oracle"], required=True)
    baseline.add_argument("--evidence-records", required=True)
    baseline.add_argument("--demonstration")
    baseline.add_argument("--output", required=True)
    explanation_eval = add("evaluate-explanations", command_explanation_eval)
    split(explanation_eval)
    explanation_eval.add_argument("--predictions", required=True)
    explanation_eval.add_argument("--output", required=True)
    export = add("export-external", command_export_external)
    split(export)
    export.add_argument("--evidence-records", required=True)
    export.add_argument("--method", required=True)
    export.add_argument("--output", required=True)
    importer = add("import-external", command_import_external)
    split(importer)
    importer.add_argument("--input", required=True)
    importer.add_argument("--mapping", required=True)
    importer.add_argument("--method", required=True)
    importer.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)
