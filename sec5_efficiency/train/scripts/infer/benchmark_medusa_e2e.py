import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType

import torch
import transformers

ROOT = Path('${SEC5_ROOT}/train')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3 import Qwen3ForMedusa  # noqa: E402
from qwen3.modeling_qwen3_new_28_fast import Qwen3ForCausalLM as FastMethodSource  # noqa: E402


def build_model(model_path: str, use_fast_method: bool):
    dtype_name = os.environ.get('BENCH_DTYPE', 'float16').lower()
    if dtype_name == 'bfloat16':
        dtype = torch.bfloat16
    elif dtype_name == 'float32':
        dtype = torch.float32
    else:
        dtype = torch.float16

    model, _ = Qwen3ForMedusa.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map='auto' if torch.cuda.is_available() else None,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )
    if torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '<pad>'
    model.tokenizer = tokenizer

    if use_fast_method:
        fast_fn = FastMethodSource.medusa_generate_interleaved_multihead_stream_user_same_y0
        model.medusa_generate_interleaved_multihead_stream_user_same_y0 = MethodType(fast_fn, model)

    candidate_fns = [
        'medusa_generate_interleaved_multihead_stream_user_same_y0',
        'medusa_generate_interleaved_multihead_stream_user',
        'medusa_generate_interleaved_multihead_stream',
    ]
    gen_fn = None
    gen_fn_name = None
    for fn_name in candidate_fns:
        fn = getattr(model, fn_name, None)
        if fn is not None:
            gen_fn = fn
            gen_fn_name = fn_name
            break
    if gen_fn is None:
        raise RuntimeError(f'Medusa function not found. Tried: {candidate_fns}')

    return model, tokenizer, gen_fn_name


def make_prompt():
    return os.environ.get(
        'BENCH_PROMPT',
        (
            'Summarize the technical reason why dense attention masks can slow autoregressive '
            'inference, especially when grouped-query attention and multi-stream attention are used.'
        ),
    )


def make_kwargs(model, gen_fn_name: str):
    fn = getattr(model, gen_fn_name)
    sig = inspect.signature(fn)
    params = sig.parameters
    prompt = make_prompt()
    assistant_heads = int(os.environ.get('BENCH_ASSISTANT_HEADS', '2'))
    max_new_tokens = int(os.environ.get('BENCH_MAX_NEW_TOKENS', '1024'))
    max_steps = int(os.environ.get('BENCH_MAX_STEPS', '4096'))
    temperature = float(os.environ.get('BENCH_TEMPERATURE', '0.0'))
    top_p = float(os.environ.get('BENCH_TOP_P', '1.0'))
    top_k = int(os.environ.get('BENCH_TOP_K', '50'))
    presence_penalty = float(os.environ.get('BENCH_PRESENCE_PENALTY', '0.0'))
    do_sample = temperature > 0

    kwargs = {}
    if 'question_text' in params:
        kwargs['question_text'] = prompt
    if 'assistant_heads' in params:
        kwargs['assistant_heads'] = assistant_heads
    if 'assistant_prefix_texts' in params:
        kwargs['assistant_prefix_texts'] = [''] * assistant_heads
    if 'assistant_prefill_texts' in params:
        kwargs['assistant_prefill_texts'] = [''] * assistant_heads
    if 'max_new_tokens' in params:
        kwargs['max_new_tokens'] = max_new_tokens
    if 'max_steps' in params:
        kwargs['max_steps'] = max_steps
    if 'temperature' in params:
        kwargs['temperature'] = temperature
    if 'top_p' in params:
        kwargs['top_p'] = top_p
    if 'top_k' in params:
        kwargs['top_k'] = top_k
    if 'presence_penalty' in params:
        kwargs['presence_penalty'] = presence_penalty
    if 'do_sample' in params:
        kwargs['do_sample'] = do_sample
    if 'stop_on_im_end' in params:
        kwargs['stop_on_im_end'] = True
    if 'allow_same_step_visible' in params:
        kwargs['allow_same_step_visible'] = False
    return kwargs


def count_generated_tokens(out):
    if isinstance(out, dict):
        total = 0
        by_head = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                n = int(v.numel())
                by_head[int(k)] = n
                total += n
        return total, by_head
    return 0, {}


def decode_generated_text(out, tokenizer):
    texts = {}
    if not isinstance(out, dict):
        return texts
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            ids = v.tolist()
            texts[int(k)] = tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
    return texts


def benchmark_model(model, gen_fn_name: str):
    fn = getattr(model, gen_fn_name)
    kwargs = make_kwargs(model, gen_fn_name)
    warmup = int(os.environ.get('BENCH_WARMUP', '1'))
    iters = int(os.environ.get('BENCH_ITERS', '3'))

    for _ in range(warmup):
        _ = fn(**kwargs)
        torch.cuda.synchronize()

    times = []
    output_size = None
    tokens_by_head = None
    texts_by_head = None
    for _ in range(iters):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn(**kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_tokens, by_head = count_generated_tokens(out)
        token_per_sec = total_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else None
        times.append(
            {
                'elapsed_ms': elapsed_ms,
                'peak_mem_gb': torch.cuda.max_memory_allocated() / (1024**3),
                'token_per_sec': token_per_sec,
                'generated_tokens': total_tokens,
            }
        )
        output_size = total_tokens
        tokens_by_head = by_head
        texts_by_head = decode_generated_text(out, model.tokenizer)

    elapsed = [x['elapsed_ms'] for x in times]
    peak_mem = [x['peak_mem_gb'] for x in times]
    tps = [x['token_per_sec'] for x in times if x['token_per_sec'] is not None]
    gen_tokens = [x['generated_tokens'] for x in times]
    elapsed_sorted = sorted(elapsed)
    return {
        'method_name': gen_fn_name,
        'kwargs': kwargs,
        'iters': iters,
        'warmup': warmup,
        'mean_ms': sum(elapsed) / len(elapsed),
        'p50_ms': elapsed_sorted[len(elapsed_sorted) // 2],
        'min_ms': min(elapsed),
        'max_ms': max(elapsed),
        'mean_peak_mem_gb': sum(peak_mem) / len(peak_mem),
        'mean_token_per_sec': (sum(tps) / len(tps)) if tps else None,
        'mean_generated_tokens': sum(gen_tokens) / len(gen_tokens),
        'last_generated_tokens_by_head': tokens_by_head,
        'last_generated_text_by_head': texts_by_head,
        'output_size': output_size,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for benchmark_medusa_e2e.py')

    multi_model_path = os.environ.get('MULTI_MODEL_PATH')
    if not multi_model_path:
        raise SystemExit('Please set MULTI_MODEL_PATH to the multi-stream checkpoint directory')

    out_dir = Path(os.environ.get('BENCH_OUT_DIR', '${DATA_ROOT}/para_out/bench_medusa_e2e'))
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = [
        ('baseline_multi', False),
        ('fast_multi', True),
    ]

    rows = []
    model_cache = {}
    for label, use_fast_method in layout:
        model, _, gen_fn_name = build_model(multi_model_path, use_fast_method=use_fast_method)
        model_cache[label] = model
        row = {
            'label': label,
            'model_path': multi_model_path,
            'device': torch.cuda.get_device_name(0),
            'use_fast_method': use_fast_method,
            'result': benchmark_model(model, gen_fn_name),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    for model in model_cache.values():
        del model
    torch.cuda.empty_cache()

    keyed = {row['label']: row for row in rows}
    summary = {}
    if 'baseline_multi' in keyed and 'fast_multi' in keyed:
        base = keyed['baseline_multi']['result']['mean_ms']
        fast = keyed['fast_multi']['result']['mean_ms']
        base_tps = keyed['baseline_multi']['result'].get('mean_token_per_sec')
        fast_tps = keyed['fast_multi']['result'].get('mean_token_per_sec')
        summary['baseline_multi_mean_ms'] = base
        summary['fast_multi_mean_ms'] = fast
        summary['fast_multi_speedup_vs_baseline_multi'] = base / fast if fast > 0 else None
        summary['baseline_multi_mean_token_per_sec'] = base_tps
        summary['fast_multi_mean_token_per_sec'] = fast_tps
        summary['fast_multi_token_per_sec_speedup'] = (fast_tps / base_tps) if base_tps and fast_tps else None

    payload = {'rows': rows, 'summary': summary}
    out_path = out_dir / 'benchmark_medusa_e2e_results.json'
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
