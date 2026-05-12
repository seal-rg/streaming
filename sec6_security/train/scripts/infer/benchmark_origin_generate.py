import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_prompt():
    return os.environ.get(
        "BENCH_PROMPT",
        (
            "Summarize the technical reason why dense attention masks can slow "
            "autoregressive inference, especially when grouped-query attention and "
            "multi-stream attention are used."
        ),
    )


def build_inputs(tokenizer):
    prompt = make_prompt()
    if os.environ.get("BENCH_USE_CHAT_TEMPLATE", "0") == "1":
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    toks = tokenizer(text, return_tensors="pt")
    return toks.input_ids.cuda(), toks.attention_mask.cuda(), text


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmark_origin_generate.py")

    model_path = os.environ.get("ORIGIN_MODEL_PATH")
    if not model_path:
        raise SystemExit("Please set ORIGIN_MODEL_PATH")

    max_new_tokens = int(os.environ.get("BENCH_MAX_NEW_TOKENS", "2048"))
    temperature = float(os.environ.get("BENCH_TEMPERATURE", "0.0"))
    top_p = float(os.environ.get("BENCH_TOP_P", "1.0"))
    top_k = int(os.environ.get("BENCH_TOP_K", "50"))
    do_sample = temperature > 0
    warmup = int(os.environ.get("BENCH_WARMUP", "1"))
    iters = int(os.environ.get("BENCH_ITERS", "3"))
    out_dir = Path(os.environ.get("BENCH_OUT_DIR", "${DATA_ROOT}/para_out/bench_origin_generate"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"

    input_ids, attention_mask, prompt_text = build_inputs(tokenizer)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "use_cache": True,
    }
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    for _ in range(warmup):
        _ = model.generate(**generate_kwargs)
        torch.cuda.synchronize()

    times = []
    final_text = ""
    generated_tokens = 0
    for _ in range(iters):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = model.generate(**generate_kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        new_ids = out[0, input_ids.shape[1] :]
        generated_tokens = int(new_ids.numel())
        final_text = tokenizer.decode(
            new_ids.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        tokps = generated_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else None
        times.append(
            {
                "elapsed_ms": elapsed_ms,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
                "generated_tokens": generated_tokens,
                "token_per_sec": tokps,
            }
        )

    elapsed = [x["elapsed_ms"] for x in times]
    tps = [x["token_per_sec"] for x in times if x["token_per_sec"] is not None]
    mem = [x["peak_mem_gb"] for x in times]
    result = {
        "model_path": model_path,
        "device": torch.cuda.get_device_name(0),
        "prompt": prompt_text,
        "input_tokens": int(input_ids.shape[1]),
        "max_new_tokens": max_new_tokens,
        "iters": iters,
        "warmup": warmup,
        "mean_ms": sum(elapsed) / len(elapsed),
        "p50_ms": sorted(elapsed)[len(elapsed) // 2],
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "mean_peak_mem_gb": sum(mem) / len(mem),
        "mean_generated_tokens": sum(x["generated_tokens"] for x in times) / len(times),
        "mean_token_per_sec": (sum(tps) / len(tps)) if tps else None,
        "last_generated_text": final_text,
    }
    print(json.dumps(result, ensure_ascii=False))
    out_path = out_dir / "benchmark_origin_generate_results.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
