#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any


def add_metric(metrics_accum: dict[str, list[float]], metric_name: str, value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            add_metric(metrics_accum, metric_name, item)
        return
    try:
        v = float(value)
    except Exception:
        return
    metrics_accum.setdefault(metric_name, []).append(v)


def binom_stderr(xs: list[float]) -> float:
    if not xs:
        return 0.0
    n = len(xs)
    p = sum(xs) / n
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate lm_eval-style IFEval per-doc metrics from a jsonl file.")
    ap.add_argument("jsonl", help="Path to ifeval_seed*.jsonl")
    ap.add_argument("--out", help="Output metrics json path. Defaults to <jsonl>.ifeval_metrics.json")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out) if args.out else jsonl_path.with_suffix(".ifeval_metrics.json")

    metrics_accum: dict[str, list[float]] = {}
    num_samples = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_metrics = obj.get("doc_metrics")
            if not isinstance(doc_metrics, dict):
                raise ValueError(f"Missing or invalid doc_metrics at line {line_no}")
            num_samples += 1
            for metric_name, value in doc_metrics.items():
                add_metric(metrics_accum, metric_name, value)

    final = {
        "dataset": "ifeval",
        "num_samples": num_samples,
        "inst_level_loose_acc": sum(metrics_accum.get("inst_level_loose_acc", [])) / max(len(metrics_accum.get("inst_level_loose_acc", [])), 1),
        "inst_level_strict_acc": sum(metrics_accum.get("inst_level_strict_acc", [])) / max(len(metrics_accum.get("inst_level_strict_acc", [])), 1),
        "prompt_level_loose_acc": sum(metrics_accum.get("prompt_level_loose_acc", [])) / max(len(metrics_accum.get("prompt_level_loose_acc", [])), 1),
        "prompt_level_strict_acc": sum(metrics_accum.get("prompt_level_strict_acc", [])) / max(len(metrics_accum.get("prompt_level_strict_acc", [])), 1),
        "prompt_level_loose_acc_stderr": binom_stderr(metrics_accum.get("prompt_level_loose_acc", [])),
        "prompt_level_strict_acc_stderr": binom_stderr(metrics_accum.get("prompt_level_strict_acc", [])),
    }

    out = {
        "final": final,
        "all_metrics_raw": {k: len(v) for k, v in metrics_accum.items()},
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
