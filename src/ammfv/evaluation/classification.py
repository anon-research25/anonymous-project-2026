from __future__ import annotations

from collections import Counter

import numpy as np


def classification_metrics(gold, predicted, classes, averaging="macro"):
    """Class-wise F1 is averaged directly, never derived from averaged P/R.

    Failures/abstentions are always included as incorrect predictions; recall
    denominators include their ground-truth examples. They are extra confusion
    columns, not a spurious additional benchmark class.
    """
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Metric inputs must have equal, nonzero length")
    if any(label not in classes for label in gold):
        raise ValueError("Ground truth outside evaluation class set")
    rows = []
    for label in classes:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        support = sum(g == label for g in gold)
        predicted_count = sum(p == label for p in predicted)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": label, "precision": precision, "recall": recall,
                     "f1": f1, "support": support, "predicted_count": predicted_count, "tp": tp})
    if averaging == "macro":
        weights = np.ones(len(classes)) / len(classes)
    elif averaging == "weighted":
        weights = np.asarray([row["support"] for row in rows]) / len(gold)
    elif averaging == "micro":
        tp = sum(row["tp"] for row in rows)
        predicted_count = sum(row["predicted_count"] for row in rows)
        precision, recall = tp / max(predicted_count, 1), tp / len(gold)
        aggregate = {"precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
        weights = None
    else:
        raise ValueError("Supported averaging: macro, weighted, micro")
    if weights is not None:
        aggregate = {metric: float(sum(weight * row[metric] for weight, row in zip(weights, rows)))
                     for metric in ("precision", "recall", "f1")}
    columns = list(classes) + sorted(set(predicted) - set(classes))
    confusion = [[sum(g == label and p == column for g, p in zip(gold, predicted))
                  for column in columns] for label in classes]
    return {"label_accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold),
            **aggregate, "averaging": averaging, "n": len(gold), "per_class": rows,
            "confusion": {"rows": classes, "columns": columns, "matrix": confusion},
            "prediction_counts": dict(Counter(predicted))}
