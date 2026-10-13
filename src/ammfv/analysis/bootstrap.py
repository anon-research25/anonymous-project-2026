"""Optional descriptive uncertainty for measured predictions, never paper scores."""
from __future__ import annotations

import numpy as np

from ..evaluation.classification import classification_metrics


def paired_accuracy_interval(gold, pre, post, samples=2000, confidence=0.95, seed=42):
    if len(gold) != len(pre) or len(pre) != len(post) or not gold:
        raise ValueError("Bootstrap needs equally sized nonempty paired lists")
    differences = np.asarray([int(after == label) - int(before == label)
                              for label, before, after in zip(gold, pre, post)], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(samples)]
    tail = (1 - confidence) / 2
    return {"delta_accuracy_pp": float(differences.mean() * 100),
            "interval_pp": [float(np.quantile(estimates, tail) * 100),
                            float(np.quantile(estimates, 1 - tail) * 100)],
            "confidence": confidence, "bootstrap_samples": samples, "seed": seed,
            "unit": "claim", "paired": True}


def classification_interval(gold, predictions, classes, metric="f1", samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    gold, predictions = list(gold), list(predictions)
    if not gold:
        raise ValueError("Bootstrap requires nonempty predictions")
    estimates = []
    for _ in range(samples):
        indices = rng.integers(0, len(gold), size=len(gold))
        score = classification_metrics([gold[i] for i in indices], [predictions[i] for i in indices], classes)
        estimates.append(score[metric])
    point = classification_metrics(gold, predictions, classes)[metric]
    return {"metric": metric, "estimate": point, "interval95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "bootstrap_samples": samples, "seed": seed}
