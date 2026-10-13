"""Diagnostic graph-head predictions, distinct from final AMMFV debate verdicts."""
import argparse

import torch
from torch.utils.data import DataLoader

from ammfv.config import load_config
from ammfv.data.features import CachedGraphDataset, collate_graphs
from ammfv.evaluation.classification import classification_metrics
from ammfv.graph.model import HeterogeneousGraphVerifier
from ammfv.schema import VERDICTS
from ammfv.training.graph_trainer import move_batch
from ammfv.utils.checkpoint import load_checkpoint
from ammfv.utils.io import write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets/mocheg.yaml")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = CachedGraphDataset(args.features or config["data"]["features_dir"], args.split,
                              config["retrieval"]["use_images"])
    loader = DataLoader(data, batch_size=config["graph_training"]["batch_size"], collate_fn=collate_graphs)
    device = config["graph"]["device"]
    model = HeterogeneousGraphVerifier(config["graph"]).to(device).eval()
    state = load_checkpoint(config["graph"]["checkpoint"], model)
    if state["provenance"]["clip_encoder"] != data.metadata["encoder"]:
        raise ValueError("Graph diagnostic features do not match checkpoint encoder")
    rows, labels, predictions = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch["features"], batch["node_types"], batch["node_mask"])
            probabilities = output["probabilities"].cpu().tolist()
            predicted = output["logits"].argmax(-1).cpu().tolist()
            gold = batch["labels"].cpu().tolist()
            for identifier, score, prediction, label in zip(batch["claim_ids"], probabilities, predicted, gold):
                rows.append({"claim_id": identifier, "graph_verdict": VERDICTS[prediction],
                             "probabilities": score, "gold": VERDICTS[label], "output_kind": "graph_head_diagnostic"})
                labels.append(VERDICTS[label])
                predictions.append(VERDICTS[prediction])
    write_jsonl(args.output, rows)
    write_json(args.output + ".metrics.json", classification_metrics(labels, predictions, list(VERDICTS)))


if __name__ == "__main__":
    main()
