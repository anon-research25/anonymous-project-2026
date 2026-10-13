from __future__ import annotations

import csv
import json
from pathlib import Path

from ..utils.io import atomic_text


def collect_metric_files(paths, output, latex_output=None):
    rows = []
    for path in paths:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        metric = document["classification"]
        rows.append({"run": Path(path).parent.name, "LA": 100 * metric["label_accuracy"],
                     "Pre": 100 * metric["precision"], "Rec": 100 * metric["recall"],
                     "F1": 100 * metric["f1"], "n": metric["n"],
                     "averaging": metric["averaging"], "status": document["status"]})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "LA", "Pre", "Rec", "F1", "n", "averaging", "status"])
        writer.writeheader()
        writer.writerows(rows)
    if latex_output:
        lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Run & LA & Pre & Rec & F1 \\", r"\midrule"]
        for row in rows:
            name = row["run"].replace("_", r"\_")
            lines.append(f"{name} & {row['LA']:.2f} & {row['Pre']:.2f} & {row['Rec']:.2f} & {row['F1']:.2f} " + r"\\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        atomic_text(latex_output, "\n".join(lines) + "\n")
    return rows
