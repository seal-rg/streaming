#!/usr/bin/env python3
"""Test interrupt handling: inject user tokens mid-generation.

Uses the gen.send() protocol in stream_inference.generate().
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from stream_inference import CHANNELS, C, load_model, generate


def run_interrupt_test(
    model, tokenizer, silence_token, initial_prompt, interrupt_text, interrupt_row, max_rows=200, **gen_kwargs
):
    """Generate with an interrupt injected at a specific row."""
    # First, generate initial rows with the prompt
    # Then switch to interactive mode and inject interrupt

    # Tokenize the interrupt
    interrupt_tokens = tokenizer.encode(interrupt_text, add_special_tokens=False)

    # We'll run in non-interactive mode but simulate interrupt by
    # concatenating: initial_prompt tokens, then silence until interrupt_row,
    # then interrupt tokens
    # Actually, easier: use user_text="" (interactive mode) and send() tokens

    # Tokenize initial prompt
    initial_tokens = tokenizer.encode(initial_prompt, add_special_tokens=False)

    # Build the full user token sequence:
    # Row 0: silence (pre-think)
    # Rows 1..len(initial): initial prompt
    # Rows len(initial)+1..interrupt_row-1: silence
    # Rows interrupt_row..interrupt_row+len(interrupt): interrupt
    # Rest: silence

    user_token_sequence = []
    user_token_sequence.append(silence_token)  # pre-think row 0
    user_token_sequence.extend(initial_tokens)
    # Pad with silence until interrupt_row
    while len(user_token_sequence) < interrupt_row:
        user_token_sequence.append(silence_token)
    # Add interrupt tokens
    user_token_sequence.extend(interrupt_tokens)

    # Now run with the combined user text
    # We need to construct this as a single prompt... but that won't trigger
    # the interrupt feel. Instead, let's use gen.send() protocol.

    gen = generate(
        model, tokenizer, "",  # empty = interactive mode
        silence_token,
        max_rows=max_rows,
        pre_think=0,  # we handle pre-think ourselves
        **gen_kwargs,
    )

    rows = []
    token_idx = 0
    try:
        # First call with None (starts the generator)
        row_idx, row, is_prefill = next(gen)
        rows.append((row_idx, row, is_prefill))

        while True:
            # Determine what user token to send
            if is_prefill:
                send_token = silence_token
            elif token_idx < len(user_token_sequence):
                send_token = user_token_sequence[token_idx]
                token_idx += 1
            else:
                send_token = silence_token

            row_idx, row, is_prefill = gen.send(send_token)
            rows.append((row_idx, row, is_prefill))
    except StopIteration:
        pass

    return rows


def format_grid(rows, tokenizer, silence_token):
    """Format rows as a readable grid."""
    # Header
    header = "    " + " | ".join(f"{ch:10s}" for ch in CHANNELS)
    sep = "    " + "+".join(["-" * 12] * C)
    lines = [header, sep]

    for row_idx, token_ids, is_prefill in rows:
        prefix = f"{'P' if is_prefill else ' '}{row_idx:3d}"
        cells = []
        for c, tid in enumerate(token_ids):
            tok = tokenizer.decode([tid]).strip()
            if tid == silence_token:
                tok = "-"
            cells.append(f"{tok:10s}")
        lines.append(f"{prefix} {' | '.join(cells)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Test interrupt handling")
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial-prompt", required=True)
    parser.add_argument("--interrupt-text", required=True)
    parser.add_argument("--interrupt-row", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--silence-penalty", type=float, default=5.0)
    parser.add_argument("--skip-silence", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Save output to file")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer, silence_token = load_model(args.model, args.device)
    print(f"Silence token ID: {silence_token}")
    print(f"Initial prompt: {args.initial_prompt!r}")
    print(f"Interrupt at row {args.interrupt_row}: {args.interrupt_text!r}")
    print("Ready.\n")

    rows = run_interrupt_test(
        model, tokenizer, silence_token,
        initial_prompt=args.initial_prompt,
        interrupt_text=args.interrupt_text,
        interrupt_row=args.interrupt_row,
        max_rows=args.max_rows,
        temperature=args.temperature,
        silence_penalty=args.silence_penalty,
        skip_silence=args.skip_silence,
    )

    grid = format_grid(rows, tokenizer, silence_token)
    print(grid)
    print(f"\n  {len(rows)} rows generated")

    if args.output:
        with open(args.output, "w") as f:
            f.write(grid)
            f.write(f"\n\n  {len(rows)} rows generated\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
