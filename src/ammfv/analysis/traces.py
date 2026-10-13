"""Self-contained HTML traces and descriptive routing/debate/cost reports."""
from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..utils.io import atomic_text, write_json


def export_trace_html(trace_path, output):
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    title = html.escape(trace["claim"]["claim_id"])
    sections = []
    for key in ("claim", "routing", "retrieval", "graph", "deliberation", "result", "calls"):
        if key in trace:
            text = html.escape(json.dumps(trace[key], ensure_ascii=False, indent=2))
            sections.append(f"<details {'open' if key in {'claim','result'} else ''}><summary>{key}</summary><pre>{text}</pre></details>")
    document = ("<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>AMMFV trace {title}</title>"
                "<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;"
                "background:#f8fafc;color:#172334}details{background:white;padding:16px;margin:12px 0;"
                "border:1px solid #dce3ea;border-radius:8px}summary{font-weight:600;cursor:pointer}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}</style></head><body>"
                f"<h1>AMMFV trace: {title}</h1>" + "".join(sections) + "</body></html>")
    atomic_text(output, document)


def summarize_predictions(records, output):
    records = list(records)
    ok = [r for r in records if r.get("status", "ok") == "ok"]
    latency = [r["elapsed_seconds"] for r in records if "elapsed_seconds" in r]
    report = {"n": len(records), "successes": len(ok), "failures": len(records) - len(ok),
              "strategies": dict(Counter(r.get("strategy") for r in ok)),
              "epistemic_modes": dict(Counter(r.get("epistemic_mode", "evidence_grounded") for r in ok)),
              "verdicts": dict(Counter(r["verdict"] for r in ok)),
              "round_counts": dict(Counter(r.get("rounds", 0) for r in ok)),
              "termination": dict(Counter(r.get("termination_reason") for r in ok)),
              "debate_invocations": sum(r.get("debate_invoked", False) for r in ok),
              "latency_seconds": {"mean": float(np.mean(latency)),
                                  "p50": float(np.percentile(latency, 50)),
                                  "p95": float(np.percentile(latency, 95))} if latency else None}
    token_counts, cached = [], 0
    for record in ok:
        if record.get("trace_path") and Path(record["trace_path"]).is_file():
            trace = json.loads(Path(record["trace_path"]).read_text(encoding="utf-8"))
            for call in trace.get("calls", []):
                cached += call.get("cached", False)
                if not call.get("cached", False):
                    token_counts.extend(call.get("attempts", []))
    report["generation"] = {"uncached_attempts": len(token_counts), "cached_calls": cached,
                             "input_tokens": sum(a["input_tokens"] for a in token_counts),
                             "output_tokens": sum(a["output_tokens"] for a in token_counts)}
    write_json(output, report)
    return report
