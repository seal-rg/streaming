import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def make_causal_mask(q_len: int, k_len: int, device, dtype):
    mask = torch.full((1, 1, q_len, k_len), float('-inf'), device=device, dtype=dtype)
    q_pos = torch.arange(q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    visible = k_pos <= (k_len - q_len + q_pos)
    mask.masked_fill_(visible.unsqueeze(0).unsqueeze(0), 0.0)
    return mask


def make_full_visible_mask(q_len: int, k_len: int, device, dtype):
    return torch.zeros((1, 1, q_len, k_len), device=device, dtype=dtype)


def benchmark_case(name, q_len, k_len, d_head=128, num_q_heads=32, num_kv_heads=32, iters=100, warmup=30, dtype=torch.bfloat16):
    device = 'cuda'
    query = torch.randn((1, num_q_heads, q_len, d_head), device=device, dtype=dtype)
    key = torch.randn((1, num_kv_heads, k_len, d_head), device=device, dtype=dtype)
    value = torch.randn((1, num_kv_heads, k_len, d_head), device=device, dtype=dtype)

    attn_mask = None
    is_causal = False
    enable_gqa = False

    if name.endswith('causal_flag'):
        is_causal = True
    elif name.endswith('dense_causal_mask'):
        attn_mask = make_causal_mask(q_len, k_len, device, dtype)
    elif name.endswith('dense_full_visible'):
        attn_mask = make_full_visible_mask(q_len, k_len, device, dtype)
    else:
        raise ValueError(name)

    if num_q_heads != num_kv_heads:
        enable_gqa = True

    for _ in range(warmup):
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times_sorted = sorted(times)
    mid = len(times_sorted) // 2
    p50 = times_sorted[mid]
    p90 = times_sorted[min(len(times_sorted) - 1, int(len(times_sorted) * 0.9))]
    mean = sum(times) / len(times)
    return {
        'name': name,
        'q_len': q_len,
        'k_len': k_len,
        'num_q_heads': num_q_heads,
        'num_kv_heads': num_kv_heads,
        'mean_ms': mean,
        'p50_ms': p50,
        'p90_ms': p90,
        'iters': iters,
        'warmup': warmup,
        'dtype': str(dtype),
        'gqa': num_q_heads != num_kv_heads,
    }


def format_summary(rows):
    lines = []
    grouped = {}
    for row in rows:
        grouped.setdefault((row['k_len'], row['num_q_heads'], row['num_kv_heads']), []).append(row)
    for key in sorted(grouped):
        k_len, num_q_heads, num_kv_heads = key
        lines.append(f'=== K={k_len}, q_heads={num_q_heads}, kv_heads={num_kv_heads} ===')
        bucket = sorted(grouped[key], key=lambda x: x['name'])
        for row in bucket:
            lines.append(
                f"{row['name']}: mean={row['mean_ms']:.3f} ms, p50={row['p50_ms']:.3f} ms, p90={row['p90_ms']:.3f} ms"
            )
        by_name = {r['name']: r for r in bucket}
        ratio_pairs = [
            ('single_q2_dense_causal_mask', 'single_q2_causal_flag'),
            ('multi_q3_dense_causal_mask', 'multi_q3_causal_flag'),
            ('multi_q3_dense_full_visible', 'multi_q3_causal_flag'),
            ('multi_q5_dense_causal_mask', 'multi_q5_causal_flag'),
            ('multi_q5_dense_full_visible', 'multi_q5_causal_flag'),
            ('multi_q9_dense_causal_mask', 'multi_q9_causal_flag'),
            ('multi_q9_dense_full_visible', 'multi_q9_causal_flag'),
        ]
        for slow_name, base_name in ratio_pairs:
            if slow_name in by_name and base_name in by_name:
                ratio = by_name[slow_name]['mean_ms'] / by_name[base_name]['mean_ms']
                lines.append(f'ratio {slow_name} / {base_name} = {ratio:.3f}x')
        if 'single_q2_causal_flag' in by_name and 'multi_q5_causal_flag' in by_name:
            ratio = by_name['multi_q5_causal_flag']['mean_ms'] / by_name['single_q2_causal_flag']['mean_ms']
            lines.append(f'ratio multi_q5_causal_flag / single_q2_causal_flag = {ratio:.3f}x')
        if 'single_q2_dense_causal_mask' in by_name and 'multi_q5_dense_causal_mask' in by_name:
            ratio = by_name['multi_q5_dense_causal_mask']['mean_ms'] / by_name['single_q2_dense_causal_mask']['mean_ms']
            lines.append(f'ratio multi_q5_dense_causal_mask / single_q2_dense_causal_mask = {ratio:.3f}x')
        if 'single_q2_causal_flag' in by_name and 'multi_q3_causal_flag' in by_name:
            ratio = by_name['multi_q3_causal_flag']['mean_ms'] / by_name['single_q2_causal_flag']['mean_ms']
            lines.append(f'ratio multi_q3_causal_flag / single_q2_causal_flag = {ratio:.3f}x')
        if 'single_q2_dense_causal_mask' in by_name and 'multi_q3_dense_causal_mask' in by_name:
            ratio = by_name['multi_q3_dense_causal_mask']['mean_ms'] / by_name['single_q2_dense_causal_mask']['mean_ms']
            lines.append(f'ratio multi_q3_dense_causal_mask / single_q2_dense_causal_mask = {ratio:.3f}x')
        if 'single_q2_causal_flag' in by_name and 'multi_q9_causal_flag' in by_name:
            ratio = by_name['multi_q9_causal_flag']['mean_ms'] / by_name['single_q2_causal_flag']['mean_ms']
            lines.append(f'ratio multi_q9_causal_flag / single_q2_causal_flag = {ratio:.3f}x')
        if 'single_q2_dense_causal_mask' in by_name and 'multi_q9_dense_causal_mask' in by_name:
            ratio = by_name['multi_q9_dense_causal_mask']['mean_ms'] / by_name['single_q2_dense_causal_mask']['mean_ms']
            lines.append(f'ratio multi_q9_dense_causal_mask / single_q2_dense_causal_mask = {ratio:.3f}x')
        lines.append('')
    return '\n'.join(lines)


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this benchmark')

    out_dir = Path(os.environ.get('BENCH_OUT_DIR', '${DATA_ROOT}/para_out/bench_sdpa_mask'))
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0)
    print(f'device={device_name}')

    cases = [
        ('single_q2_causal_flag', 2),
        ('single_q2_dense_causal_mask', 2),
        ('multi_q3_causal_flag', 3),
        ('multi_q3_dense_causal_mask', 3),
        ('multi_q3_dense_full_visible', 3),
        ('multi_q5_causal_flag', 5),
        ('multi_q5_dense_causal_mask', 5),
        ('multi_q5_dense_full_visible', 5),
        ('multi_q9_causal_flag', 9),
        ('multi_q9_dense_causal_mask', 9),
        ('multi_q9_dense_full_visible', 9),
    ]

    results = []
    for num_q_heads, num_kv_heads in [(32, 32), (32, 8)]:
        for k_len in (1024, 4096, 8192, 16384):
            for name, q_len in cases:
                row = benchmark_case(
                    name=name,
                    q_len=q_len,
                    k_len=k_len,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                )
                results.append(row)
                print(json.dumps(row, ensure_ascii=False))

    summary = format_summary(results)
    print('\n' + summary)

    payload = {
        'device': device_name,
        'results': results,
        'summary': summary,
    }
    out_path = out_dir / 'benchmark_sdpa_mask_results.json'
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
