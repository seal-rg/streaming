import json
import os
import random
import sys
import time
from pathlib import Path
from types import MethodType

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("${SEC6_ROOT}/train")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3 import Qwen3ForMedusa  # noqa: E402
from qwen3.modeling_qwen3_new_28_fast import Qwen3ForCausalLM as FastMethodSource  # noqa: E402


def pick_split(ds_dict, preferred=("test", "validation", "dev", "train")):
    for s in preferred:
        if s in ds_dict:
            return s
    return list(ds_dict.keys())[0]


def build_logicnli_input(ex):
    premise = ex.get("premise", "") or ""
    hyp = ex.get("hypothesis", "") or ""
    return (
        f"Determine whether the hypothesis: {str(hyp).strip()} "
        f"is entailment, contradiction, neutral, or self_contradiction "
        f"based on the provided context below.\n\n"
        f"Context:\nPremise: {str(premise).strip()}"
    )


def load_prompts():
    task = os.environ.get("BENCH_TASK", "logicnli").lower()
    if task != "logicnli":
        raise ValueError(f"Unsupported BENCH_TASK: {task}")
    num_samples = int(os.environ.get("BENCH_NUM_SAMPLES", "100"))
    seed = int(os.environ.get("BENCH_SEED", "42"))
    ds = load_dataset("tasksource/LogicNLI")
    split = os.environ.get("BENCH_SPLIT") or pick_split(ds)
    ds_split = ds[split]
    total = len(ds_split)
    if num_samples <= 0 or num_samples >= total:
        idxs = list(range(total))
    else:
        rng = random.Random(seed)
        idxs = rng.sample(range(total), num_samples)
    prompts = []
    for idx in idxs:
        ex = ds_split[idx]
        prompts.append(
            {
                "index": int(idx),
                "gold": ex.get("label"),
                "prompt": build_logicnli_input(ex),
            }
        )
    return prompts, split


def build_origin():
    model_path = os.environ["ORIGIN_MODEL_PATH"]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    return model, tokenizer


def build_multi(use_fast_method: bool):
    model_path = os.environ["MULTI_MODEL_PATH"]
    model, _ = Qwen3ForMedusa.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )
    model = model.to("cuda")
    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    model.tokenizer = tokenizer
    if use_fast_method:
        fast_fn = FastMethodSource.medusa_generate_interleaved_multihead_stream_user_same_y0
        model.medusa_generate_interleaved_multihead_stream_user_same_y0 = MethodType(fast_fn, model)
    fn_name = "medusa_generate_interleaved_multihead_stream_user_same_y0"
    if not hasattr(model, fn_name):
        raise RuntimeError(f"{fn_name} not found on multi model")
    return model, tokenizer, fn_name


def run_origin(model, tokenizer, prompt: str):
    use_chat = os.environ.get("BENCH_USE_CHAT_TEMPLATE", "0") == "1"
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    toks = tokenizer(text, return_tensors="pt")
    input_ids = toks.input_ids.cuda()
    attention_mask = toks.attention_mask.cuda()
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(os.environ.get("BENCH_ORIGIN_MAX_NEW_TOKENS", "4096")),
        "do_sample": False,
        "use_cache": True,
    }
    t0 = time.perf_counter()
    out = model.generate(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_ids = out[0, input_ids.shape[1] :]
    text_out = tokenizer.decode(
        new_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    return {
        "elapsed_sec": elapsed,
        "generated_tokens": int(new_ids.numel()),
        "text": text_out,
    }


def run_multi(model, fn_name: str, prompt: str):
    fn = getattr(model, fn_name)
    assistant_heads = int(os.environ.get("BENCH_ASSISTANT_HEADS", "2"))
    kwargs = {
        "question_text": prompt,
        "assistant_heads": assistant_heads,
        "assistant_prefix_texts": [""] * assistant_heads,
        "assistant_prefill_texts": [""] * assistant_heads,
        "max_new_tokens": int(os.environ.get("BENCH_MULTI_MAX_NEW_TOKENS", "2048")),
        "max_steps": int(os.environ.get("BENCH_MULTI_MAX_STEPS", "8192")),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50,
        "presence_penalty": 0.0,
        "do_sample": False,
        "stop_on_im_end": True,
        "allow_same_step_visible": False,
    }
    t0 = time.perf_counter()
    out = fn(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tokens = 0
    by_head_tokens = {}
    by_head_text = {}
    for k, v in out.items():
        n = int(v.numel())
        by_head_tokens[int(k)] = n
        total_tokens += n
        by_head_text[int(k)] = model.tokenizer.decode(
            v.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
    return {
        "elapsed_sec": elapsed,
        "generated_tokens": total_tokens,
        "generated_tokens_by_head": by_head_tokens,
        "text_by_head": by_head_text,
    }


def aggregate(rows, key):
    total_time = sum(r[key]["elapsed_sec"] for r in rows)
    total_tokens = sum(r[key]["generated_tokens"] for r in rows)
    return {
        "num_samples": len(rows),
        "total_elapsed_sec": total_time,
        "total_generated_tokens": total_tokens,
        "token_per_sec": total_tokens / total_time if total_time > 0 else None,
        "avg_elapsed_sec": total_time / len(rows) if rows else None,
        "avg_generated_tokens": total_tokens / len(rows) if rows else None,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if "ORIGIN_MODEL_PATH" not in os.environ:
        raise SystemExit("Please set ORIGIN_MODEL_PATH")
    if "MULTI_MODEL_PATH" not in os.environ:
        raise SystemExit("Please set MULTI_MODEL_PATH")

    out_dir = Path(os.environ.get("BENCH_OUT_DIR", "${DATA_ROOT}/para_out/bench_compare_origin_multi"))
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts, split = load_prompts()
    origin_model, origin_tok = build_origin()
    multi_model, _, multi_fn_name = build_multi(use_fast_method=False)
    fast_model, _, fast_fn_name = build_multi(use_fast_method=True)

    rows = []
    for item in prompts:
        prompt = item["prompt"]
        row = {
            "index": item["index"],
            "gold": item["gold"],
            "prompt": prompt,
            "origin": run_origin(origin_model, origin_tok, prompt),
            "baseline_multi": run_multi(multi_model, multi_fn_name, prompt),
            "fast_multi": run_multi(fast_model, fast_fn_name, prompt),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "index": row["index"],
                    "origin_t": row["origin"]["elapsed_sec"],
                    "multi_t": row["baseline_multi"]["elapsed_sec"],
                    "fast_t": row["fast_multi"]["elapsed_sec"],
                },
                ensure_ascii=False,
            )
        )

    summary = {
        "task": "logicnli",
        "split": split,
        "origin": aggregate(rows, "origin"),
        "baseline_multi": aggregate(rows, "baseline_multi"),
        "fast_multi": aggregate(rows, "fast_multi"),
    }
    if summary["baseline_multi"]["token_per_sec"] and summary["origin"]["token_per_sec"]:
        summary["baseline_multi_vs_origin_token_speedup"] = summary["baseline_multi"]["token_per_sec"] / summary["origin"]["token_per_sec"]
    if summary["fast_multi"]["token_per_sec"] and summary["baseline_multi"]["token_per_sec"]:
        summary["fast_multi_vs_baseline_multi_token_speedup"] = (
            summary["fast_multi"]["token_per_sec"] / summary["baseline_multi"]["token_per_sec"]
        )

    result = {"rows": rows, "summary": summary}
    out_json = out_dir / "benchmark_compare_origin_multi_results.json"
    out_jsonl = out_dir / "benchmark_compare_origin_multi_results.jsonl"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_jsonl}")


if __name__ == "__main__":
    main()
