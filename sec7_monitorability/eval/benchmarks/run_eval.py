#!/usr/bin/env python3
"""Run the three Section-7 monitorability benchmarks on a multi-stream checkpoint.

Usage:
    python -m eval.benchmarks.run_eval --model /path/to/checkpoint \
        --benchmark monitor_accuracy

    Benchmarks: alignment_faking, monitor_accuracy, concern_subvocalization,
                all (runs all three).
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)














def run_monitor_accuracy(
    model,
    data_dir: str,
    max_items: int,
    output_dir: str,
    ablate: list[int] | None,
    judge=None,
    **gen_kwargs,
):
    """Run monitorability — does thinking leak the misaligned target behavior?

    Headline metric is `monitor_accuracy`: a multi-class classifier (judge)
    reads the model's private thinking and predicts which of six fixed
    misaligned-behavior labels the scenario was probing. If `judge` is
    None, the headline metric is unavailable and only the descriptive
    keyword proxy is reported.
    """
    from . import monitor_accuracy

    items = monitor_accuracy.load_dataset(data_dir, max_items=max_items)
    log.info(
        f"Monitorability: {len(items)} scenarios across "
        f"{len(set(i.category for i in items))} categories"
        + (" (with monitor judge)" if judge is not None else " (no judge — keyword proxy only)")
    )

    scores = []
    results_log = []

    from .inference import GenerationResult

    for i, item in enumerate(items):
        system, user = monitor_accuracy.format_prompt(item)
        sr = model.generate(
            user, system_prompt=system, ablate_channels=ablate, **gen_kwargs
        )
        result = GenerationResult.from_stream_result(
            sr, prompt=user, system_prompt=system)
        score = monitor_accuracy.score_response(item, result, judge=judge)
        scores.append(score)

        results_log.append(
            {
                "item_idx": i,
                "id": item.id,
                "category": item.category,
                "source_dataset": item.source_dataset,
                "target_behavior": item.target_behavior,
                "system_prompt": system,
                "prompt": user,
                "result": result.to_dict(),
                "score": score,
            }
        )

        if (i + 1) % 10 == 0:
            judged = [s for s in scores if s.get("monitor_correct") is not None]
            acc = (sum(1 for s in judged if s["monitor_correct"]) / len(judged)
                   if judged else float("nan"))
            ss = sum(s["stream_signal"] for s in scores) / len(scores)
            log.info(
                f"  [{i + 1}/{len(items)}] monitor_accuracy: {acc:.3f}"
                f" (n_judged={len(judged)}), proxy_signal: {ss:.3f}"
            )

    summary = monitor_accuracy.aggregate_scores(scores)
    _save(output_dir, "monitor_accuracy_results.json", summary, results_log)
    return summary


def run_alignment_faking(
    model,
    data_dir: str,
    max_items: int,
    output_dir: str,
    ablate: list[int] | None,
    judge=None,
    **gen_kwargs,
):
    """Run alignment faking — monitored vs unmonitored behavior.

    If `judge` is provided, also runs the Greenblatt alignment-faking
    classifier and the Abdelnabi eval-awareness classifier on each
    condition's concatenated thinking channels (see alignment_faking.py).
    """
    from . import alignment_faking as af

    items = af.load_dataset(max_items=max_items)
    log.info(
        f"Alignment Faking: {len(items)} prompts × 2 conditions"
        + (" (with judge)" if judge is not None else "")
    )

    scores = []
    results_log = []

    from .inference import GenerationResult

    for i, item in enumerate(items):
        m_sr = model.generate(
            item.monitored_prompt, ablate_channels=ablate, **gen_kwargs
        )
        u_sr = model.generate(
            item.unmonitored_prompt, ablate_channels=ablate, **gen_kwargs
        )
        monitored = GenerationResult.from_stream_result(
            m_sr, prompt=item.monitored_prompt)
        unmonitored = GenerationResult.from_stream_result(
            u_sr, prompt=item.unmonitored_prompt)

        score = af.score_pair(item, monitored, unmonitored, judge=judge)
        scores.append(score)

        results_log.append(
            {
                "item_idx": i,
                "category": item.category,
                "signal": item.signal,
                "monitored_prompt": item.monitored_prompt,
                "unmonitored_prompt": item.unmonitored_prompt,
                "monitored": monitored.to_dict(),
                "unmonitored": unmonitored.to_dict(),
                "score": score,
            }
        )

        log.info(
            f"  [{i + 1}/{len(items)}] {item.signal} (changes={score['behavior_changes']})"
        )

    summary = af.aggregate_scores(scores)
    _save(output_dir, "alignment_faking_results.json", summary, results_log)
    return summary



















def run_concern_subvocalization(
    model,
    data_dir: str,
    max_items: int,
    output_dir: str,
    ablate: list[int] | None,
    judge=None,
    **gen_kwargs,
):
    """Run concern_subvocalization — do streams articulate concerns Output suppresses?

    Headline: judge__stream_catches_rate = fraction of items where the
    reasoning articulates at least one item-specific concern at the
    specificity_floor threshold AND Output does not.

    Judge calls per item = 2 × n_concerns (for Output and reasoning).
    Scales linearly with item count and concern density; skip judge for
    a dry run that only saves outputs.
    """
    from . import concern_subvocalization as ut

    items = ut.load_dataset(data_dir, max_items=max_items)
    log.info(
        f"Uncomfortable Truth: {len(items)} items across "
        f"{len(set(i.category for i in items))} categories"
        + (" (with judge)" if judge is not None else "")
    )

    scores = []
    results_log = []

    from .inference import GenerationResult

    for i, item in enumerate(items):
        prompt = ut.format_prompt(item)
        sr = model.generate(prompt, ablate_channels=ablate, **gen_kwargs)
        result = GenerationResult.from_stream_result(sr, prompt=prompt)
        score = ut.score_response(item, result, judge=judge)
        scores.append(score)
        results_log.append({
            "item_idx": i,
            "id": item.id,
            "category": item.category,
            "severity": item.severity,
            "prompt": prompt,
            "result": result.to_dict(),
            "score": score,
        })
        if (i + 1) % 5 == 0 or (i + 1) == len(items):
            log.info(f"  [{i + 1}/{len(items)}] {item.id}")

    summary = ut.aggregate_scores(scores)
    _save(output_dir, "concern_subvocalization_results.json", summary, results_log)
    return summary
















# ── Helpers ──────────────────────────────────────────────────────────────────


_HEADLINE_KEYS = ("methodology_version", "n", "monitor_accuracy", "compliance_gap")


def _is_valid_results_file(path: str) -> bool:
    """Return True if `path` parses as JSON and has a usable summary block.

    Pre-v5 the runners only checked file presence, so a partial / empty
    file from a crashed worker would block re-runs. We require the file
    to parse, contain a `summary` dict, and have at least one expected
    headline key OR the `methodology_version` marker (which every v2+
    benchmark emits even when all rates are None).
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    summary = data.get("summary")
    if not isinstance(summary, dict) or not summary:
        return False
    if "methodology_version" in summary:
        return True
    return any(k in summary for k in _HEADLINE_KEYS)


def _save(output_dir, filename, summary, items_log):
    """Atomic JSON write: tmp file + rename.

    Multiple cluster dispatchers can race on the same output_dir, and a
    crashed worker mid-write would leave a truncated file that the
    skip-if-exists check would then keep. Writing to {file}.tmp and renaming
    keeps either the previous valid file or the new valid file at the final
    path; never a partial mix.
    """
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, filename)
    tmp_path = f"{final_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump({"summary": summary, "items": items_log}, f, indent=2, default=str)
    os.replace(tmp_path, final_path)


def print_summary(name: str, summary: dict):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    for key, val in summary.items():
        if isinstance(val, dict):
            print(f"\n  {key}:")
            for k2, v2 in val.items():
                if isinstance(v2, dict):
                    print(f"    {k2}:")
                    for k3, v3 in v2.items():
                        print(
                            f"      {k3}: {v3:.4f}"
                            if isinstance(v3, float)
                            else f"      {k3}: {v3}"
                        )
                else:
                    print(
                        f"    {k2}: {v2:.4f}"
                        if isinstance(v2, float)
                        else f"    {k2}: {v2}"
                    )
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-stream model benchmark evaluation"
    )
    parser.add_argument("--model", default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--benchmark",
        default=None,
        choices=[
            "alignment_faking",
            "monitor_accuracy",
            "concern_subvocalization",
            "all",
        ],
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to benchmark data (uses built-in data/ if not set)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (auto-generated from model name if not set)",
    )
    parser.add_argument(
        "--max-items", type=int, default=50, help="Max items per benchmark"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map", default=None, help="Device map for multi-GPU (e.g. 'auto')"
    )
    parser.add_argument(
        "--arch", default="auto", choices=["auto", "qwen3", "qwen3_5", "qwen3_5_moe"]
    )
    # Generation params
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--silence-penalty", type=float, default=2.0)
    parser.add_argument("--think-silence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--skip-silence",
        action="store_true",
        help="Mask silence tokens as attention keys",
    )
    parser.add_argument(
        "--system-prompt",
        action="store_true",
        help="Warm-start with 10-row system prompt header before user input",
    )
    parser.add_argument(
        "--force-silence-penalty-on-user-talking",
        action="store_true",
        help="Apply the Output silence penalty also while the user is actively "
        "talking (default: penalty fires only after user input ends). Used "
        "by the audit C.1 forced-coverage experiment for stream_override "
        "to test whether the architecture *can* exercise the mid-generation "
        "override moat under forced conditions, even though it doesn't "
        "naturally do so from the training distribution.",
    )
    # Original metric options
    parser.add_argument(
        "--judge-model",
        default=None,
        help="LLM judge for original metrics. For API: 'gpt-4', 'gpt-4o'. "
        "For local: HF model path/ID. Used by StrongREJECT, JailbreakBench.",
    )
    parser.add_argument(
        "--judge-api-base",
        default=None,
        help="OpenAI-compatible API base URL for judge (e.g. http://localhost:8000/v1)",
    )
    args = parser.parse_args()

    gen_kwargs = dict(
        max_rows=args.max_rows,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        silence_penalty=args.silence_penalty,
        think_silence_penalty=args.think_silence_penalty,
        skip_silence=args.skip_silence,
        warm_start=args.system_prompt,
        silence_penalty_during_user_talk=args.force_silence_penalty_on_user_talking,
    )

    # Auto-generate output dir from model name if not set
    if args.output_dir is None:
        model_name = Path(args.model).name if args.model else "unknown"
        args.output_dir = os.path.join(
            "eval_results", f"{model_name}_n{args.max_items}"
        )

    # Resolve data directory
    data_dir = args.data_dir or str(Path(__file__).parent / "data")

    if not args.model:
        parser.error("--model is required")
    if not args.benchmark:
        parser.error("--benchmark is required")

    from .inference import StreamModel

    log.info(f"Loading model from {args.model}...")
    model = StreamModel(
        args.model, device=args.device, arch=args.arch, device_map=args.device_map
    )
    log.info("Model loaded.")

    # Optional: LLM judge for all three paper benchmarks.
    llm_judge = None
    if args.judge_model:
        from .llm_judge import LLMJudge

        llm_judge = LLMJudge(
            model_id=args.judge_model,
            api_base=args.judge_api_base,
        )
        log.info(f"LLM judge configured: {args.judge_model}")

    all_benchmarks = {
        "alignment_faking": ("Alignment Faking", run_alignment_faking),
        "monitor_accuracy": ("Monitor Accuracy", run_monitor_accuracy),
        "concern_subvocalization": ("Concern Sub-vocalization", run_concern_subvocalization),
    }

    if args.benchmark == "all":
        to_run = list(all_benchmarks.keys())
    else:
        to_run = [args.benchmark]

    # Save run metadata
    os.makedirs(args.output_dir, exist_ok=True)
    import datetime

    metadata = {
        "model": args.model,
        "model_name": Path(args.model).name if args.model else None,
        "benchmarks": to_run,
        "max_items": args.max_items,
        "generation": gen_kwargs,
        "device": args.device,
        "device_map": args.device_map,
        "arch": args.arch,
        "judge_model": args.judge_model,
        "judge_api_base": args.judge_api_base,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Run metadata saved to {meta_path}")

    for bench_key in to_run:
        bench_name, bench_fn = all_benchmarks[bench_key]
        bench_output = os.path.join(args.output_dir, bench_key)

        # Skip benchmarks that already have a VALID results file (resume after
        # crash). A bare presence check would treat a partial/truncated file
        # from a crashed worker as "done" and prevent rerun, leaving silently-
        # blank summaries in the run.
        if os.path.isdir(bench_output):
            existing = [
                f for f in os.listdir(bench_output) if f.endswith("_results.json")
            ]
            valid = [
                f for f in existing
                if _is_valid_results_file(os.path.join(bench_output, f))
            ]
            if valid:
                log.info(
                    f"\nSkipping {bench_name} (valid results already exist: {valid[0]})"
                )
                continue
            if existing:
                log.warning(
                    f"\nRe-running {bench_name}: existing {existing} not "
                    "parseable / no headline summary keys (likely crashed worker)."
                )

        log.info(f"\nRunning {bench_name}...")
        t0 = time.time()

        # All three paper benchmarks accept an optional `judge` kwarg.
        extra = {}
        if llm_judge is not None:
            extra["judge"] = llm_judge
        summary = bench_fn(
            model,
            data_dir,
            args.max_items,
            bench_output,
            ablate=None,
            **extra,
            **gen_kwargs,
        )
        print_summary(bench_name, summary)

        elapsed = time.time() - t0
        log.info(f"{bench_name} completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
