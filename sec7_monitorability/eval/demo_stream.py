#!/usr/bin/env python3
"""
Stream inference demo — generates 10 parallel cognitive streams in real time.

The model predicts all 10 channels simultaneously per timestep (row) using a
block-causal attention mask where same-row tokens cannot see each other.

Usage:
    python eval/demo_stream.py --model /path/to/checkpoint "Hey, what's up?"
    python eval/demo_stream.py --model /path/to/checkpoint  # interactive mode
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from stream_inference import (
    CHANNELS,
    C,
    generate,
    load_model,
)

# ANSI colors
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

CHANNEL_STYLES = {
    0: BOLD + BLUE,  # User
    1: BOLD + GREEN,  # Output
    9: BOLD + YELLOW,  # Synthesis
}


def load_training_sample(data_path, sample_idx=0):
    """Load a training sample and return its rows as list of [C] token id lists."""
    import json

    jsonl_path = os.path.join(data_path, "dataset.jsonl")
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                sample = json.loads(line)
                break
    token_ids = sample["token_ids"]  # [C][R]
    num_rows = sample["num_rows"]
    rows = []
    for r in range(num_rows):
        rows.append([token_ids[c][r] for c in range(C)])
    return rows


def decode_token(tokenizer, tok_id, silence_token):
    """Decode a single token, cleaning up whitespace markers."""
    if tok_id == silence_token:
        return "-"
    text = tokenizer.decode([tok_id])
    return text.strip() or "-"


def print_header(col_widths):
    pad = "    "  # matches row number prefix width
    header = " | ".join(
        (CHANNEL_STYLES.get(i, CYAN) + ch + RESET).ljust(
            w + len(CHANNEL_STYLES.get(i, CYAN)) + len(RESET)
        )
        for i, (ch, w) in enumerate(zip(CHANNELS, col_widths))
    )
    sep = "-+-".join("-" * w for w in col_widths)
    print(pad + header)
    print(pad + DIM + sep + RESET)


def print_row(tokenizer, row_idx, row, col_widths, silence_token, is_prefill=False):
    cells = []
    for i, tok_id in enumerate(row):
        text = decode_token(tokenizer, tok_id, silence_token)[: col_widths[i]]
        if is_prefill:
            style = DIM
        elif text == "-":
            style = DIM
        else:
            style = CHANNEL_STYLES.get(i, "")
        cells.append(style + text.ljust(col_widths[i]) + RESET)
    prefix = DIM + f"{row_idx:3d} " + RESET
    print(prefix + " | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description="Stream LLM inference demo")
    parser.add_argument("prompt", nargs="?", default=None, help="User prompt")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument(
        "--pre-think", type=int, default=1, help="Silence rows before user input begins"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--silence-penalty",
        type=float,
        default=10.0,
        help="Logit penalty on silence for Output after user input (0=off)",
    )
    parser.add_argument(
        "--think-silence-penalty",
        type=float,
        default=0.0,
        help="Logit penalty on silence for thinking channels 2-9 (0=off)",
    )
    parser.add_argument(
        "--prefill-sample",
        type=int,
        default=None,
        metavar="IDX",
        help="Prefill with 25%% of training sample IDX from data dir",
    )
    parser.add_argument(
        "--data-dir",
        default="${DATA_ROOT}/v12_processed",
        help="Path to training data (for --prefill-sample)",
    )
    parser.add_argument(
        "--skip-silence",
        action="store_true",
        help="Use block_causal_skip_silence mask (silence tokens not attended to as keys)",
    )
    parser.add_argument(
        "--system-prompt",
        action="store_true",
        help="Warm-start with 10-row system prompt header before user input",
    )
    parser.add_argument(
        "--arch", default="auto", choices=["auto", "qwen3", "qwen3_5", "qwen3_5_moe"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default=None,
        help="Device map for multi-GPU (e.g. 'auto')",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer, silence_token = load_model(
        args.model, args.device, arch=args.arch, device_map=args.device_map
    )
    print(f"Silence token ID: {silence_token}")
    print("Ready.\n")

    if args.prefill_sample is not None and args.prompt is not None:
        parser.error("--prefill-sample and prompt are mutually exclusive")

    gen_kwargs = dict(
        max_rows=args.max_rows,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        silence_penalty=args.silence_penalty,
        think_silence_penalty=args.think_silence_penalty,
        skip_silence=args.skip_silence,
        warm_start=args.system_prompt,
    )

    # Prefill mode: load training sample, prefill 25%, generate the rest
    if args.prefill_sample is not None:
        all_rows = load_training_sample(args.data_dir, args.prefill_sample)
        n_prefill = len(all_rows) // 4
        prefill_rows = all_rows[:n_prefill]
        print(
            f"Prefilling with {n_prefill}/{len(all_rows)} rows from sample {args.prefill_sample}\n"
        )

        col_widths = [max(len(ch), 10) for ch in CHANNELS]
        print()
        print_header(col_widths)

        t0 = time.time()
        n_rows = 0
        for row_idx, row, is_prefill in generate(
            model,
            tokenizer,
            "",
            silence_token,
            pre_think=args.pre_think,
            prefill_rows=prefill_rows,
            **gen_kwargs,
        ):
            print_row(tokenizer, row_idx, row, col_widths, silence_token, is_prefill)
            n_rows += 1

        elapsed = time.time() - t0
        print()
        print(
            DIM
            + f"  {n_rows} rows ({n_prefill} prefill + {n_rows - n_prefill} generated) "
            f"in {elapsed:.1f}s" + RESET
        )
        return

    # Interactive / single-prompt mode
    while True:
        if args.prompt is not None:
            user_text = args.prompt
        else:
            try:
                user_text = input(BOLD + "You: " + RESET).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_text:
                continue

        col_widths = [max(len(ch), 10) for ch in CHANNELS]
        print()
        print_header(col_widths)

        t0 = time.time()
        n_rows = 0
        for row_idx, row, is_prefill in generate(
            model,
            tokenizer,
            user_text,
            silence_token,
            pre_think=args.pre_think,
            **gen_kwargs,
        ):
            print_row(tokenizer, row_idx, row, col_widths, silence_token, is_prefill)
            n_rows += 1

        elapsed = time.time() - t0
        print()
        print(
            DIM + f"  {n_rows} rows in {elapsed:.1f}s "
            f"({n_rows / elapsed:.1f} rows/s)" + RESET
        )
        print()

        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()
