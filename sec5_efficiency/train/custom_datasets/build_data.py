#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_streaming_gsm8k_symbolic_vllm.py

Sentence-level streaming supervision builder for hamishivi/gsm8k-symbolic (or local JSONL)
using a local vLLM teacher WITH chat templates.

Policy (final):
- Split user question into sentences: input_sentences
- For each sentence t:
  * teacher outputs one incremental reply (1–3 sentences)
  * ONLY at the LAST sentence: teacher must provide final answer and END with:
      "the answer is \\boxed{<number>}"
  * Before the last sentence: teacher must NOT output \\boxed{...} nor "the answer is"
- No extra final pass beyond the last sentence.

Output JSONL each line:
{
  "input_sentences": [...],
  "output_sentences": [...],   # same length as input_sentences
  "meta": {...}                # includes reference_solution if present
}

Usage (HF dataset):
  python build_streaming_gsm8k_symbolic_vllm.py \
    --hf_dataset hamishivi/gsm8k-symbolic \
    --hf_split train \
    --output_jsonl out.jsonl \
    --model /path/to/teacher \
    --tensor_parallel_size 1 \
    --max_samples 1000

Usage (local JSONL):
  python build_streaming_gsm8k_symbolic_vllm.py \
    --input_jsonl data.jsonl \
    --output_jsonl out.jsonl \
    --model /path/to/teacher
"""

from __future__ import annotations

import os
import re
import json
import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Reduce vLLM logging noise (best-effort)
# ----------------------------
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.worker").setLevel(logging.ERROR)

# ============================================================
# Sentence splitting (SAFE version: no f-string regex pitfalls)
# ============================================================

_CN_PUNC = "。！？；"
_EN_PUNC = ".?!;"
_SPLIT_CHARS = _CN_PUNC + _EN_PUNC
_SPLIT_RE = re.compile(r"([" + re.escape(_SPLIT_CHARS) + r"])")

def split_sentences(text: str, min_len: int = 15, max_len: int = 320) -> List[str]:
    """
    GSM8K-friendly sentence splitter:
    1) split by newlines
    2) split by strong punctuation (kept)
    3) merge too-short fragments
    4) split too-long by commas/semicolons

    Safe: avoids f-string + backslash regex issues.
    """
    if not text:
        return []

    t = text.strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Split on newlines first
    lines = [ln.strip() for ln in re.split(r"\n+", t) if ln.strip()]

    rough: List[str] = []
    for ln in lines:
        parts = _SPLIT_RE.split(ln)
        for i in range(0, len(parts), 2):
            chunk = parts[i]
            punc = parts[i + 1] if i + 1 < len(parts) else ""
            seg = (chunk + punc).strip()
            if seg:
                rough.append(seg)

    # Merge too-short fragments
    merged: List[str] = []
    buf = ""
    for s in rough:
        if not buf:
            buf = s
            continue
        if len(buf) < min_len:
            buf = (buf + " " + s).strip()
        else:
            merged.append(buf.strip())
            buf = s
    if buf:
        merged.append(buf.strip())

    # Split too-long fragments by commas/semicolons
    final: List[str] = []
    for s in merged:
        if len(s) <= max_len:
            final.append(s.strip())
            continue

        subs = re.split(r"(,|，|;|；)", s)
        tmp = ""
        for i in range(0, len(subs), 2):
            chunk = subs[i]
            punc = subs[i + 1] if i + 1 < len(subs) else ""
            seg = (chunk + punc).strip()
            if not seg:
                continue
            if not tmp:
                tmp = seg
            elif len(tmp) + 1 + len(seg) <= max_len:
                tmp = (tmp + " " + seg).strip()
            else:
                final.append(tmp.strip())
                tmp = seg
        if tmp:
            final.append(tmp.strip())

    return [s for s in final if s]


# ============================================================
# Dataset adapters: HF gsm8k-symbolic and local JSONL messages
# ============================================================

def extract_messages(row: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Supports:
      - HF sample dict: {"messages":[{role,content},...], ...}
      - local json line as dict with "messages"
      - local json line as a list of messages
    """
    if isinstance(row, dict) and isinstance(row.get("messages"), list):
        return row["messages"]
    if isinstance(row, list) and all(isinstance(m, dict) for m in row):
        return row
    return None

def extract_user_and_meta(messages: List[Dict[str, Any]], row: Any = None) -> Tuple[str, Dict[str, Any]]:
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "") or ""
    assistant = next((m.get("content", None) for m in messages if m.get("role") == "assistant"), None)

    meta: Dict[str, Any] = {}
    if isinstance(row, dict):
        for k in ("id", "_id", "uid", "task_id"):
            if k in row:
                meta[k] = row[k]
    if isinstance(assistant, str) and assistant.strip():
        meta["reference_solution"] = assistant.strip()
    return user.strip(), meta

def iter_hf_dataset(name: str, split: str):
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    for ex in ds:
        yield ex

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ============================================================
# vLLM Teacher (chat-template aware)
# ============================================================

@dataclass
class VLLMTeacherConfig:
    model: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    seed: int = 1234

    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 160


class VLLMTeacher:
    def __init__(self, cfg: VLLMTeacherConfig):
        self.cfg = cfg
        from vllm import LLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)

        kwargs = dict(
            model=cfg.model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            seed=cfg.seed,
            trust_remote_code=True,
        )
        if cfg.max_model_len is not None:
            kwargs["max_model_len"] = cfg.max_model_len

        # Some vLLM versions support disable_log_stats; ignore if not accepted.
        try:
            kwargs["disable_log_stats"] = True
        except Exception:
            pass

        self.llm = LLM(**kwargs)

    def generate_from_messages(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
            stop=stop or None,
        )

        out = self.llm.generate([prompt], params)
        return out[0].outputs[0].text.strip()


# ============================================================
# Output format enforcement: "the answer is \\boxed{...}"
# ============================================================

BOX_RE = re.compile(r"\\boxed\{[^}]+\}")
ANSWER_BOX_RE = re.compile(r"the answer is\s*\\boxed\{[^}]+\}\s*$", re.IGNORECASE)

def strip_early_answer(text: str) -> str:
    """
    Non-last step: remove any accidental \\boxed{...} or 'the answer is ...' leakage.
    """
    x = (text or "").strip()
    if "\\boxed" in x:
        x = x.split("\\boxed")[0].strip()
    x = re.split(r"\bthe answer is\b", x, flags=re.IGNORECASE)[0].strip()
    return x

def enforce_last_format(text: str) -> str:
    """
    Last step: ensure reply ends with: 'the answer is \\boxed{...}' and nothing after.
    Best-effort:
      - If already ends properly -> return
      - If contains \\boxed{...} -> rewrite suffix
      - Else return as-is (caller will retry)
    """
    x = (text or "").strip()
    if ANSWER_BOX_RE.search(x):
        return x

    boxes = list(BOX_RE.finditer(x))
    if boxes:
        last_box = boxes[-1].group(0)
        prefix = x[:boxes[-1].start()].rstrip()
        if prefix:
            prefix = prefix + " "
        x2 = (prefix + f"the answer is {last_box}").strip()
        return x2

    return x


# ============================================================
# Prompts with few-shot examples (chat format)
# ============================================================

def build_messages(input_sents: List[str], prev_out: List[str], t: int) -> List[Dict[str, str]]:
    seen = input_sents[: t + 1]
    cur = input_sents[t]
    is_last = (t == len(input_sents) - 1)

    fewshot = r"""
Example 1
Seen:
- Ravi has 45 apples in his basket.
Reply (not last):
Ravi starts with 45 apples.

Seen:
- Ravi has 45 apples in his basket.
- He gives one-fifth of his apples to his friend.
Reply (not last):
He will give away one-fifth of 45, so we should compute that amount next.

Seen:
- Ravi has 45 apples in his basket.
- He gives one-fifth of his apples to his friend.
- How many apples does Ravi have left?
Reply (last):
One-fifth of 45 is 9, so 45 − 9 = 36 apples left; the answer is \boxed{36}.

Example 2
Seen:
- A pack has 100 chocolates.
Reply (not last):
We have 100 chocolates total.

Seen:
- A pack has 100 chocolates.
- Lucy gets 20% and Sarah gets 30%.
Reply (not last):
Together they take 50%, leaving 50% for others.

Seen:
- A pack has 100 chocolates.
- Lucy gets 20% and Sarah gets 30%.
- The remainder is shared equally among four friends. How many does each get?
Reply (last):
50% of 100 is 50, and 50 ÷ 4 = 12.5 each; the answer is \boxed{12.5}.
""".strip()

    system = f"""You are a teacher replying while reading a math word problem sentence by sentence.

MUST-FOLLOW output rules:
- Output ONLY the teacher reply. Do NOT include meta commentary (no "We need to...", no "Rules:", no "assistantfinal").
- Be concise (1–3 sentences).
- If NOT the last sentence: DO NOT output \\boxed{{...}} and DO NOT write "the answer is".
- If the last sentence: you MUST end EXACTLY with: the answer is \\boxed{{<number>}}
- Do not output anything after that final phrase.

Few-shot examples:
{fewshot}
"""

    user = f"""Seen sentences:
{chr(10).join([f"- {s}" for s in seen])}

Previous replies:
{chr(10).join([f"- {s}" for s in prev_out]) if prev_out else "- (none)"}

CURRENT sentence:
- {cur}

Is this the last sentence? {"YES" if is_last else "NO"}

Write the teacher reply now.
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def build_retry_messages(input_sents: List[str], last_reply: str) -> List[Dict[str, str]]:
    full = "\n".join([f"- {s}" for s in input_sents])
    system = "You rewrite text to match a strict required ending format."
    user = f"""Rewrite the reply so it ends EXACTLY with: the answer is \\boxed{{<number>}}.

FULL problem:
{full}

Previous final reply:
{last_reply}

Requirements:
- Output ONLY the rewritten reply.
- End with: the answer is \\boxed{{<number>}}
- Do not output anything after that.

Now write the corrected final reply:
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ============================================================
# Core generation: one call per sentence, enforce last format
# ============================================================

DEFAULT_STOP = ["<|eot_id|>", "</s>", "<|endoftext|>", "assistantfinal"]

def generate_streaming_last_only_answer(teacher: VLLMTeacher, input_sents: List[str], do_retry: bool = True) -> List[str]:
    outputs: List[str] = []
    last_idx = len(input_sents) - 1

    for t in range(len(input_sents)):
        msgs = build_messages(input_sents, outputs, t)
        y = teacher.generate_from_messages(msgs, stop=DEFAULT_STOP).strip()

        if t != last_idx:
            y = strip_early_answer(y)
            outputs.append(y)
            continue

        # last sentence: enforce required ending
        y = enforce_last_format(y)

        if do_retry and not ANSWER_BOX_RE.search(y):
            y2 = teacher.generate_from_messages(build_retry_messages(input_sents, y), stop=DEFAULT_STOP).strip()
            y2 = enforce_last_format(y2)
            if ANSWER_BOX_RE.search(y2):
                y = y2

        outputs.append(y)

    return outputs


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_dataset", type=str, help="e.g. hamishivi/gsm8k-symbolic")
    src.add_argument("--input_jsonl", type=str, help="local jsonl")

    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--max_samples", type=int, default=None)

    # sentence splitting
    ap.add_argument("--min_sent_len", type=int, default=15)
    ap.add_argument("--max_sent_len", type=int, default=320)

    # vLLM
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)

    # decoding
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=160)

    # behavior
    ap.add_argument("--no_retry", action="store_true", help="Disable one-time last-sentence rewrite retry.")
    ap.add_argument("--drop_if_bad_last", action="store_true",
                    help="Drop samples whose last reply still doesn't end with 'the answer is \\\\boxed{...}'.")

    args = ap.parse_args()

    teacher = VLLMTeacher(
        VLLMTeacherConfig(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
    )

    if args.hf_dataset:
        iterator = iter_hf_dataset(args.hf_dataset, args.hf_split)
        src_name = f"{args.hf_dataset}:{args.hf_split}"
    else:
        iterator = iter_jsonl(args.input_jsonl)
        src_name = args.input_jsonl

    n = 0
    kept = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as fw:
        for row in iterator:
            msgs = extract_messages(row)
            if not msgs:
                continue

            user_text, meta = extract_user_and_meta(msgs, row=row)
            if not user_text:
                continue

            input_sents = split_sentences(user_text, min_len=args.min_sent_len, max_len=args.max_sent_len)
            if not input_sents:
                continue

            output_sents = generate_streaming_last_only_answer(
                teacher=teacher,
                input_sents=input_sents,
                do_retry=(not args.no_retry),
            )

            if args.drop_if_bad_last:
                if not output_sents or not ANSWER_BOX_RE.search(output_sents[-1].strip()):
                    continue

            record = {
                "input_sentences": input_sents,
                "output_sentences": output_sents,
                "meta": {
                    **meta,
                    "source": src_name,
                },
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

            kept += 1
            n += 1
            if args.max_samples is not None and n >= args.max_samples:
                break

    print(f"Done. Wrote {kept} samples to {args.output_jsonl}")

if __name__ == "__main__":
    main()
