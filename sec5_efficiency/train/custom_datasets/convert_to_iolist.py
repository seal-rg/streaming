#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert {input, Solver, AltSolver, Verifier, ...} JSONL
to IO-list format expected by SingleFileDatasetProcessor:

Output per line:
{
  "channels": [
    {"input": <input_text>, "output": <solver_text>},
    ...
  ],
  "sample_id": "...",
  "meta": {...}   # optional
}

Usage examples:
  python3 convert_to_iolist.py \
    --in_jsonl /path/raw.jsonl \
    --out_jsonl /path/converted.jsonl \
    --fixed_heads 3 \
    --output_keys Solver,AltSolver,Verifier \
    --pad_mode drop \
    --keep_meta

  python3 convert_to_iolist.py \
    --in_jsonl raw.jsonl --out_jsonl conv.jsonl \
    --fixed_heads 3 --pad_mode repeat_last
"""

import argparse
import json
from typing import Dict, Any, List, Optional


DEFAULT_OUTPUT_KEYS = ["Solver", "AltSolver", "Verifier"]


def pick_outputs(item: Dict[str, Any], keys: List[str]) -> List[str]:
    outs: List[str] = []
    for k in keys:
        v = item.get(k, None)
        if isinstance(v, str) and v.strip():
            outs.append(v.strip())
    return outs


def build_iolist_item(
    item: Dict[str, Any],
    fixed_heads: int,
    input_key: str,
    output_keys: List[str],
    pad_mode: str,
    keep_meta: bool,
    sample_id_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    inp = item.get(input_key, "")
    if not isinstance(inp, str) or not inp.strip():
        return None
    inp = inp.strip()

    outs = pick_outputs(item, output_keys)
    if len(outs) == 0:
        return None

    # Ensure exactly fixed_heads outputs
    if len(outs) < fixed_heads:
        if pad_mode == "drop":
            return None
        elif pad_mode == "repeat_last":
            last = outs[-1]
            outs = outs + [last] * (fixed_heads - len(outs))
        elif pad_mode == "repeat_first":
            first = outs[0]
            outs = outs + [first] * (fixed_heads - len(outs))
        else:
            raise ValueError(f"Unknown pad_mode: {pad_mode}")
    elif len(outs) > fixed_heads:
        outs = outs[:fixed_heads]

    channels = [{"input": inp, "output": o} for o in outs]

    # sample_id
    sid = None
    if sample_id_key:
        v = item.get(sample_id_key, None)
        if isinstance(v, (str, int)):
            sid = str(v)
    if sid is None:
        # fall back to existing sample_id if present
        v = item.get("sample_id", None)
        if isinstance(v, (str, int)):
            sid = str(v)

    out_item: Dict[str, Any] = {"channels": channels}
    if sid is not None:
        out_item["sample_id"] = sid

    if keep_meta:
        meta = {}
        for k in ["gold", "_quality"]:
            if k in item:
                meta[k] = item[k]
        if meta:
            out_item["meta"] = meta

    return out_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--fixed_heads", type=int, required=True)

    ap.add_argument("--input_key", type=str, default="input",
                    help="Where to read the question/input text from (default: input).")

    ap.add_argument("--output_keys", type=str, default=",".join(DEFAULT_OUTPUT_KEYS),
                    help="Comma-separated keys for outputs, e.g. Solver,AltSolver,Verifier")

    ap.add_argument("--pad_mode", type=str, default="drop",
                    choices=["drop", "repeat_last", "repeat_first"],
                    help="If outputs < fixed_heads: drop or repeat to pad.")

    ap.add_argument("--keep_meta", action="store_true",
                    help="If set, keep gold/_quality in out_item['meta'].")

    ap.add_argument("--sample_id_key", type=str, default=None,
                    help="Optional key to use as sample_id (besides existing sample_id).")

    args = ap.parse_args()

    output_keys = [k.strip() for k in args.output_keys.split(",") if k.strip()]
    if not output_keys:
        raise ValueError("Empty --output_keys")

    kept = 0
    dropped = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                dropped += 1
                continue

            out_item = build_iolist_item(
                item=item,
                fixed_heads=args.fixed_heads,
                input_key=args.input_key,
                output_keys=output_keys,
                pad_mode=args.pad_mode,
                keep_meta=args.keep_meta,
                sample_id_key=args.sample_id_key,
            )

            if out_item is None:
                dropped += 1
                continue

            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. kept={kept}, dropped={dropped}")
    print(f"Output: {args.out_jsonl}")


if __name__ == "__main__":
    main()
