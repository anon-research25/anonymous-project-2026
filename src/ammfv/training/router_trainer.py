from __future__ import annotations

from pathlib import Path
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from ..schema import STRATEGIES
from ..utils.io import file_sha256, read_jsonl, write_json
from ..utils.logging import save_run_manifest
from ..utils.reproducibility import seed_everything


class RouterDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = list(read_jsonl(path))
        self.tokenizer, self.max_length = tokenizer, max_length
        if any(row["split"] != "train" or row["strategy_label"] not in STRATEGIES for row in self.rows):
            raise ValueError("Router dataset must contain generated labels on TRAIN claims only")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        tokens["labels"] = STRATEGIES.index(row["strategy_label"])
        return tokens


def train_router(config, labels_path, resume=None):
    """Hold out generated TRAIN labels; benchmark dev/test never generate strategy labels."""
    cfg = config["router_training"]
    seed_everything(config["seed"], config["deterministic"])
    primary = int(os.environ.get("RANK", "0")) == 0
    output = Path(cfg["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], revision=cfg["revision"], num_labels=3,
        id2label=dict(enumerate(STRATEGIES)), label2id={label: i for i, label in enumerate(STRATEGIES)})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    dataset = RouterDataset(labels_path, tokenizer, cfg["max_length"])
    generator = np.random.default_rng(config["seed"])
    indices = generator.permutation(len(dataset))
    held_out = int(len(indices) * cfg["held_out_fraction"])
    if held_out < 1 or held_out >= len(indices):
        raise ValueError("Router needs at least one training and one held-out generated TRAIN record")
    from torch.utils.data import Subset
    training = Subset(dataset, indices[held_out:].tolist())
    validation = Subset(dataset, indices[:held_out].tolist())
    if primary:
        save_run_manifest(output, config, "train-router", {"labels_sha256": file_sha256(labels_path)})
        write_json(output / "partition.json", {
            "train_claim_ids": [dataset.rows[i]["claim_id"] for i in indices[held_out:]],
            "held_out_train_claim_ids": [dataset.rows[i]["claim_id"] for i in indices[:held_out]],
            "note": "Both partitions originate in the benchmark training split."})

    def metrics(prediction):
        predicted = prediction.predictions.argmax(-1)
        return {"accuracy": accuracy_score(prediction.label_ids, predicted),
                "macro_f1": f1_score(prediction.label_ids, predicted, labels=[0, 1, 2], average="macro", zero_division=0)}

    arguments = TrainingArguments(
        output_dir=str(output / "trainer"), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["accumulation_steps"], learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="epoch", save_strategy="epoch", logging_steps=cfg["logging_steps"],
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=2, fp16=cfg["precision"] == "float16", bf16=cfg["precision"] == "bfloat16",
        seed=config["seed"], data_seed=config["seed"], report_to=[],
        dataloader_num_workers=cfg["workers"])
    trainer = Trainer(model=model, args=arguments, train_dataset=training, eval_dataset=validation,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / "best"))
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "best")
        write_json(output / "validation_metrics.json", metrics)
