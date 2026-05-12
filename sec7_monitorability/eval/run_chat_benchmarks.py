#!/usr/bin/env python3
"""Run benchmarks on standard chat models (chat-trained baselines or base Qwen models).

Uses standard autoregressive generation (no stream grid) for fair comparison.
Runs the same prompts/datasets as the stream benchmarks but with standard chat inference.

Supports both:
  - Chat-trained baselines (fine-tuned on stream data in chat format)
  - Base Qwen models (unmodified, e.g. ${MODELS_ROOT}/Qwen3-4B)

Usage:
    # Chat-trained baseline
    CUDA_VISIBLE_DEVICES=4 uv run python eval/run_chat_benchmarks.py \
        --model /path/to/chat_baseline --output-dir eval_results/chat_baseline_8b

    # Base Qwen model
    CUDA_VISIBLE_DEVICES=4 uv run python eval/run_chat_benchmarks.py \
        --model ${MODELS_ROOT}/Qwen3-4B --base-model \
        --output-dir eval_results/base_qwen3_4b
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train", "train"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)


def load_chat_model(model_path, device="cuda", device_map=None, base_model=False):
    """Load a chat-trained or base Qwen model.

    Args:
        base_model: If True, load as a vanilla HuggingFace model (no custom classes).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading {'base' if base_model else 'chat'} model from {model_path}...")

    load_kwargs = dict(torch_dtype=torch.bfloat16)
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    if base_model:
        # Standard HuggingFace loading for unmodified Qwen models
        if os.path.isdir(model_path):
            load_kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    else:
        # Custom loading for chat-trained stream models
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        model_type = cfg.get("model_type", "qwen3")
        if model_type in ("qwen3_5", "qwen3_5_text"):
            from qwen3_5 import StreamQwen3_5ForCausalLM, StreamQwen3_5TextConfig
            text_cfg = cfg.get("text_config", cfg)
            config = StreamQwen3_5TextConfig(**text_cfg)
            config.use_cache = True
            model = StreamQwen3_5ForCausalLM.from_pretrained(
                model_path, config=config, **load_kwargs
            )
        else:
            from qwen3 import Qwen3ForCausalLM, Qwen3MedusaConfig
            config = Qwen3MedusaConfig.from_pretrained(model_path)
            config.use_cache = True
            model = Qwen3ForCausalLM.from_pretrained(
                model_path, config=config, ignore_mismatched_sizes=True,
                **load_kwargs,
            )

    if device_map is None:
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params).")
    return model, tokenizer


def generate_chat_result(model, tokenizer, prompt, system_prompt=None,
                          temperature=0.6, max_new_tokens=2048,
                          top_p=0.95, top_k=20, enable_thinking=False,
                          repetition_penalty=1.0):
    """Canonical chat generation.

    Returns a `GenerationResult` with the full raw decoded output (never
    stripped), the visible `verbalized_response`, and the hidden reasoning
    as `{"CoT": <think-block>}` (or `{}` when thinking is disabled). This
    is the ONLY generation primitive the chat-runner benchmarks should
    call — do not introduce parallel helpers that drop data.

    `repetition_penalty` (default 1.0 = off) is plumbed through to
    `model.generate(...)` to suppress degenerate looping at the tail of
    the trained-distribution length seen on chat_8b_allstreams.
    """
    from benchmarks.inference import GenerationResult

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        top_k=top_k if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    new_tokens = out[0][input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return GenerationResult.from_chat_text(
        raw=raw,
        prompt=prompt,
        system_prompt=system_prompt,
        thinking_enabled=enable_thinking,
    )




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


def save_results(output_dir, filename, summary, items_log):
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
























def run_alignment_faking(model, tokenizer, data_dir, max_items, output_dir,
                         judge=None, **kwargs):
    from benchmarks import alignment_faking as af
    items = af.load_dataset(max_items=max_items)
    log.info(f"Alignment Faking: {len(items)} items x2 conditions"
             + (" (with judge)" if judge is not None else ""))
    scores, results_log = [], []
    for i, item in enumerate(items):
        monitored = generate_chat_result(
            model, tokenizer, item.monitored_prompt, **kwargs)
        unmonitored = generate_chat_result(
            model, tokenizer, item.unmonitored_prompt, **kwargs)
        score = af.score_pair(item, monitored, unmonitored, judge=judge)
        scores.append(score)
        results_log.append({
            "item_idx": i, "category": item.category, "signal": item.signal,
            "monitored_prompt": item.monitored_prompt,
            "unmonitored_prompt": item.unmonitored_prompt,
            "monitored": monitored.to_dict(),
            "unmonitored": unmonitored.to_dict(),
            "score": score,
        })
        log.info(f"  [{i + 1}/{len(items)}] {item.signal} (changes={score['behavior_changes']})")
    summary = af.aggregate_scores(scores)
    save_results(output_dir, "alignment_faking_results.json", summary, results_log)
    return summary
























def run_monitor_accuracy(model, tokenizer, data_dir, max_items, output_dir,
                       judge=None, **kwargs):
    """Run monitorability with the unified GenerationResult pipeline.

    Headline metric is `monitor_accuracy` from the multi-class label judge
    (see eval/benchmarks/llm_judge.py:judge_monitor_accuracy_target_behavior).
    For chat baselines without thinking the thinking trace is empty, which
    auto-classifies as `none_of_the_above`; this is by design — chat models
    have no thinking surface for a monitor to read.
    """
    from benchmarks import monitor_accuracy
    items = monitor_accuracy.load_dataset(data_dir, max_items=max_items)
    log.info(
        f"Monitorability: {len(items)} items"
        + (" (with monitor judge)" if judge is not None else " (no judge — keyword proxy only)")
    )
    scores, results_log = [], []
    for i, item in enumerate(items):
        system, user = monitor_accuracy.format_prompt(item)
        result = generate_chat_result(
            model, tokenizer, user, system_prompt=system, **kwargs)
        score = monitor_accuracy.score_response(item, result, judge=judge)
        scores.append(score)
        results_log.append({
            "item_idx": i, "id": item.id, "category": item.category,
            "target_behavior": item.target_behavior,
            "system_prompt": system, "prompt": user,
            "result": result.to_dict(), "score": score,
        })
        if (i + 1) % 10 == 0:
            judged = [s for s in scores if s.get("monitor_correct") is not None]
            acc = (sum(1 for s in judged if s["monitor_correct"]) / len(judged)
                   if judged else float("nan"))
            log.info(
                f"  [{i + 1}/{len(items)}] monitor_accuracy: {acc:.3f}"
                f" (n_judged={len(judged)})"
            )
    summary = monitor_accuracy.aggregate_scores(scores)
    save_results(output_dir, "monitor_accuracy_results.json", summary, results_log)
    return summary






def run_concern_subvocalization(model, tokenizer, data_dir, max_items, output_dir,
                             judge=None, **kwargs):
    """Stream-comparable: do base/chat CoT articulate concerns Output suppresses?

    For base models with thinking, `generate_chat_result` packages the
    <think> block as the `hidden_reasoning={"CoT": ...}` channel, so the
    same judge-based scoring flows through identically to stream models.
    Chat baselines with no thinking produce empty reasoning → judge
    calls skip gracefully and `judge_stream_catches` is None (missing
    data), not 0 (false), per tri-state convention.
    """
    from benchmarks import concern_subvocalization as ut
    items = ut.load_dataset(data_dir, max_items=max_items)
    log.info(f"Uncomfortable Truth: {len(items)} items"
             + (" (with judge)" if judge is not None else ""))
    scores, results_log = [], []
    for i, item in enumerate(items):
        prompt = ut.format_prompt(item)
        result = generate_chat_result(model, tokenizer, prompt, **kwargs)
        score = ut.score_response(item, result, judge=judge)
        scores.append(score)
        results_log.append({
            "item_idx": i, "id": item.id, "category": item.category,
            "severity": item.severity, "prompt": prompt,
            "result": result.to_dict(), "score": score,
        })
        if (i + 1) % 5 == 0 or (i + 1) == len(items):
            log.info(f"  [{i + 1}/{len(items)}] {item.id}")
    summary = ut.aggregate_scores(scores)
    save_results(output_dir, "concern_subvocalization_results.json", summary, results_log)
    return summary






BENCHMARKS = {
    "alignment_faking": ("Alignment Faking", run_alignment_faking),
    "monitor_accuracy": ("Monitor Accuracy", run_monitor_accuracy),
    "concern_subvocalization": ("Concern Sub-vocalization", run_concern_subvocalization),
}


def main():
    parser = argparse.ArgumentParser(description="Chat/base model benchmark evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark",
        default="all",
        help="One of: alignment_faking, monitor_accuracy, concern_subvocalization, all, "
             "or a comma-separated list.",
    )
    parser.add_argument("--base-model", action="store_true",
                        help="Load as vanilla HuggingFace model (for base Qwen models)")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--rep-penalty", type=float, default=1.0,
                        help="Repetition penalty for model.generate (default 1.0 = off). "
                        "Use ~1.1 to suppress degenerate looping near the trained-"
                        "distribution length tail on chat baselines.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="Device map for multi-GPU (e.g. 'auto')")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen3 thinking mode for all generation. "
                        "Use for base Qwen3 models (not chat-finetuned baselines).")
    parser.add_argument("--judge-model", default=None,
                        help="LLM judge for the monitorability evals (e.g. 'gpt-5.4-mini').")
    args = parser.parse_args()

    model, tokenizer = load_chat_model(args.model, args.device,
                                       device_map=args.device_map,
                                       base_model=args.base_model)

    data_dir = os.path.join(os.path.dirname(__file__), "benchmarks", "data")

    gen_kwargs = dict(temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    if args.rep_penalty and args.rep_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = args.rep_penalty
        log.info(f"Repetition penalty: {args.rep_penalty}")
    if args.enable_thinking:
        gen_kwargs["enable_thinking"] = True
        log.info("Thinking mode ENABLED (CoT will be generated and stripped from output)")

    if args.benchmark == "all":
        to_run = list(BENCHMARKS.keys())
    elif "," in args.benchmark:
        to_run = [b.strip() for b in args.benchmark.split(",") if b.strip()]
    else:
        to_run = [args.benchmark]

    # Save metadata
    os.makedirs(args.output_dir, exist_ok=True)
    import datetime
    model_type = "base_model" if args.base_model else "chat_baseline"
    meta = {"model": args.model, "type": model_type,
            "enable_thinking": args.enable_thinking,
            "benchmarks": to_run,
            "max_items": args.max_items, "generation": gen_kwargs,
            "timestamp": datetime.datetime.now().isoformat()}
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    judge = None
    if args.judge_model:
        from benchmarks.llm_judge import LLMJudge
        judge = LLMJudge(model_id=args.judge_model)
        log.info(f"Judge model: {args.judge_model}")

    for key in to_run:
        name, fn = BENCHMARKS[key]
        bench_dir = os.path.join(args.output_dir, key)

        # Skip if a VALID results file exists. A bare presence check would
        # treat a partial/truncated file from a crashed worker as "done"
        # and prevent rerun, leaving silently-blank summaries in the run.
        if os.path.isdir(bench_dir):
            existing = [f for f in os.listdir(bench_dir) if f.endswith("_results.json")]
            valid = [f for f in existing
                     if _is_valid_results_file(os.path.join(bench_dir, f))]
            if valid:
                log.info(f"Skipping {name} (valid results exist: {valid[0]})")
                continue
            if existing:
                log.warning(
                    f"Re-running {name}: existing {existing} not parseable / "
                    "no headline summary keys (likely crashed worker)."
                )

        log.info(f"\nRunning {name}...")
        t0 = time.time()
        # All three paper benchmarks accept an optional `judge` kwarg.
        extra_kwargs = {"judge": judge} if judge is not None else {}
        summary = fn(model, tokenizer, data_dir, args.max_items, bench_dir,
                     **gen_kwargs, **extra_kwargs)
        elapsed = time.time() - t0
        log.info(f"{name} completed in {elapsed:.0f}s")

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif not isinstance(v, dict):
                print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
