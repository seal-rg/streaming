#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval tasksource/proofwriter from a LOCAL sampled JSONL (e.g., 1k test samples).

UPDATED: follow GSM8K script logic
- Use model.medusa_generate_interleaved_v2_stream_user(...)
- Decode out.get(0) as final text

Input JSONL format:
  each line: {"index":..., "split":..., "dataset":..., "data": {...raw HF example...}}

Output JSONL schema:
{
  "problem": "...",
  "channels": [{"input": "...", "output": "..."}],
  "_quality": {...}
}

Label extraction:
  Prefer "Answer: <label>" variants, else last True/False/Unknown.
"""

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple, Iterator
import warnings

import torch
import transformers

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_medusa import Qwen2ForMedusa


# ============================================================
# Label canonicalization + extraction
# ============================================================
LABEL_CANON = {"true": "True", "false": "False", "unknown": "Unknown"}

def canon_label(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("entailed", "entails", "entailment", "yes"):
        return "True"
    if s in ("contradicted", "contradiction", "no"):
        return "False"
    if s in ("unknown", "uncertain", "maybe"):
        return "Unknown"
    if s in LABEL_CANON:
        return LABEL_CANON[s]
    if s == "t":
        return "True"
    if s == "f":
        return "False"
    if s == "u":
        return "Unknown"
    return None

_ANSWER_BLOCK_RE  = re.compile(r"(?is)\banswer\b\s*(?::|=|-|is)\s*(true|false|unknown)\b")
_ANSWER_BLOCK_RE2 = re.compile(r"(?is)\banswer\s*:\s*(true|false|unknown)\b")
_FINAL_LABEL_RE   = re.compile(r"(?is)\bfinal\s*(?:label|answer)\s*:\s*(true|false|unknown)\b")
_LAST_LABEL_RE    = re.compile(r"(?is)\b(true|false|unknown)\b")

def extract_label(text: str) -> Optional[str]:
    if not text:
        return None
    m = _ANSWER_BLOCK_RE.search(text) or _ANSWER_BLOCK_RE2.search(text)
    if m:
        return LABEL_CANON[m.group(1).lower()]
    m = _FINAL_LABEL_RE.search(text)
    if m:
        return LABEL_CANON[m.group(1).lower()]
    allm = _LAST_LABEL_RE.findall(text)
    if allm:
        return LABEL_CANON[allm[-1].lower()]
    return None


# ============================================================
# Problem builder
# ============================================================
def build_problem(conclusion: str, premises: List[str]) -> str:
    conclusion = (conclusion or "").strip()
    premises = [str(p).strip() for p in (premises or []) if str(p).strip()]
    return (
        f'Determine whether the conclusion "{conclusion}" is true, false, or unknown. '
        f"based on the given premises below.\n\n"
        f"Premises:\n"
        + "\n".join(premises)
    )


# ============================================================
# Local JSONL reader + ProofWriter field adapter
# ============================================================
def iter_samples_from_jsonl(path: str) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                yield obj.get("index", lineno - 1), obj["data"]
            else:
                yield obj.get("index", lineno - 1), obj

def get_conclusion_and_premises(ex: Dict[str, Any]) -> Tuple[str, List[str], Optional[str]]:
    premises: List[str] = []
    if "premises" in ex:
        if isinstance(ex["premises"], list):
            premises = ex["premises"]
        elif isinstance(ex["premises"], str):
            premises = [ln.strip() for ln in ex["premises"].splitlines() if ln.strip()]

    if not premises:
        for k in ("context", "facts", "theory"):
            if k in ex:
                if isinstance(ex[k], list):
                    premises = ex[k]
                    break
                if isinstance(ex[k], str):
                    premises = [ln.strip() for ln in ex[k].splitlines() if ln.strip()]
                    break

    conclusion = ""
    for k in ("conclusion", "hypothesis", "query", "question", "statement"):
        if k in ex and isinstance(ex[k], str) and ex[k].strip():
            conclusion = ex[k].strip()
            break

    gold_raw = None
    for k in ("label", "answer", "gold", "target"):
        if k in ex and ex[k] is not None:
            gold_raw = ex[k]
            break

    return conclusion, premises, gold_raw


# ============================================================
# Medusa Inference (GSM8K-style)
# ============================================================
class MedusaInference:
    def __init__(self, model_path: str, device: str = "auto", torch_dtype: str = "auto"):
        self.model_path = model_path

        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        if self.device == "auto":
            self.device = "cpu"

        if torch_dtype == "auto":
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        elif torch_dtype == "float16":
            self.dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.system_message = "You are a helpful assistant."

        print("[Init] model_path:", model_path)
        print("[Init] device:", self.device)
        print("[Init] dtype:", self.dtype)

        self.model, _ = Qwen2ForMedusa.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device != "cpu" else None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        try:
            self.model.config.use_cache = True
        except Exception:
            pass

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

        self._setup_special_token_ids()

    def _setup_special_token_ids(self):
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id   = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_start_id is None or self.im_start_id < 0:
            raise ValueError("Tokenizer missing <|im_start|>")
        if self.im_end_id is None or self.im_end_id < 0:
            raise ValueError("Tokenizer missing <|im_end|>")

        nl = self.tokenizer.encode("\n", add_special_tokens=False)
        if not nl:
            raise ValueError("Tokenizer cannot encode newline")
        self.newline_id = nl[0]

        self.system_tokens = self.tokenizer.encode("system", add_special_tokens=False)
        self.user_tokens = self.tokenizer.encode("user", add_special_tokens=False)
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)

        self.system_prefix = [self.im_start_id] + self.system_tokens + [self.newline_id]
        self.user_prefix = [self.im_start_id] + self.user_tokens + [self.newline_id]
        self.assistant_prefix = [self.im_start_id] + self.assistant_tokens + [self.newline_id]

        print(f"[Tokens] im_start={self.im_start_id} im_end={self.im_end_id} nl={self.newline_id}")

    @torch.no_grad()
    def generate_1head(
        self,
        question_text: str,
        final_prefix: str = "",
        max_length_per_head: int = 1024,
        max_steps: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 0,
        do_sample: bool = False,
        stop_on_im_end: bool = True,
        include_im_end_in_cross_channel: bool = True,
        allow_same_step_visible: bool = False,
        disable_assistant_cross_channel: bool = False,
    ) -> str:
        out = self.model.medusa_generate_interleaved_v2_stream_user(
            question_text=question_text,
            assistant_prefix_text=final_prefix,
            max_new_tokens=max_length_per_head,
            max_steps=max_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            stop_on_im_end=stop_on_im_end,
            include_im_end_in_cross_channel=include_im_end_in_cross_channel,
            allow_same_step_visible=allow_same_step_visible,
            disable_assistant_cross_channel=disable_assistant_cross_channel,
        )

        toks = out.get(0, None)
        if toks is None or toks.numel() == 0:
            return ""
        return self.tokenizer.decode(
            toks.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("Eval ProofWriter from local JSONL (GSM8K-style medusa_generate_interleaved_v2_stream_user)")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--in_jsonl", type=str, required=True, help="Local sampled JSONL (e.g., proofwriter_test_1k.jsonl)")
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--summary_json", type=str, required=True)

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    # generation
    ap.add_argument("--max_length_per_head", type=int, default=2048)
    ap.add_argument("--max_steps", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--stop_on_im_end", action="store_true")

    # prompt
    ap.add_argument(
        "--user_instruction",
        type=str,
        default=(
            "Determine whether the conclusion is true, false, or unknown based on the premises. "
            "Explain briefly. End with exactly:\nAnswer:\n<True/False/Unknown>"
        ),
    )
    ap.add_argument("--final_prefix", type=str, default="")

    # medusa flags (match GSM8K defaults)
    ap.add_argument("--include_im_end_in_cross_channel", action="store_true",
                    help="Pass include_im_end_in_cross_channel=True (default False if not set)")
    ap.add_argument("--allow_same_step_visible", action="store_true")
    ap.add_argument("--disable_assistant_cross_channel", action="store_true")

    args = ap.parse_args()

    infer = MedusaInference(args.model_path, device=args.device, torch_dtype=args.torch_dtype)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    fout = open(args.out_jsonl, "w", encoding="utf-8")

    t0 = time.time()
    n = 0
    n_gold = 0
    n_pred = 0
    n_correct = 0

    for idx, ex in iter_samples_from_jsonl(args.in_jsonl):
        conclusion, premises, gold_raw = get_conclusion_and_premises(ex)
        problem = build_problem(conclusion=conclusion, premises=premises)

        gold = canon_label(gold_raw)
        if gold is not None:
            n_gold += 1

        # GSM8K-style: everything goes through question_text
        question_text =  problem

        final_text = infer.generate_1head(
            question_text=question_text,
            final_prefix=args.final_prefix,
            max_length_per_head=args.max_length_per_head,
            max_steps=args.max_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            stop_on_im_end=args.stop_on_im_end,
            include_im_end_in_cross_channel=args.include_im_end_in_cross_channel,
            allow_same_step_visible=args.allow_same_step_visible,
            disable_assistant_cross_channel=args.disable_assistant_cross_channel,
        )

        pred = extract_label(final_text)
        if pred is not None:
            n_pred += 1

        correct = (gold is not None and pred is not None and gold == pred)
        n_correct += int(correct)

        record = {
            "problem": problem,
            "channels": [{"input": problem, "output": final_text}],
            "_quality": {
                "keep": True,
                "quality_score": None,
                "final_label_extracted": pred,
                "gold_label": gold,
                "answer_format_ok": (pred is not None),
                "has_meta_pollution": False,
                "issues": [] if pred is not None else ["no_label_extracted"],
                "summary": "medusa inference (v2_stream_user)",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "judge_model": None,
                "token_stats": {
                    # v2_stream_user does not expose per-head prompt/gen stats here
                    "note": "token stats unavailable in this wrapper; add hooks if needed",
                },
                "index": idx,
            },
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        n += 1
        if n % 50 == 0:
            acc = n_correct / max(n, 1)
            print(f"[ProofWriter] {n} done | acc={acc:.4f} | pred_rate={n_pred/max(n,1):.4f}")

    fout.close()

    elapsed = time.time() - t0
    summary = {
        "dataset": "tasksource/proofwriter",
        "split": "test (local sampled jsonl)",
        "num_samples": n,
        "gold_available_n": n_gold,
        "pred_extracted_n": n_pred,
        "accuracy_over_all_samples": n_correct / max(n, 1),
        "accuracy_over_gold_available": n_correct / max(n_gold, 1),
        "pred_extraction_rate": n_pred / max(n, 1),
        "elapsed_sec": elapsed,
        "samples_per_sec": n / max(elapsed, 1e-9),
        "config": {
            "in_jsonl": args.in_jsonl,
            "max_length_per_head": args.max_length_per_head,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": bool(args.do_sample),
            "stop_on_im_end": bool(args.stop_on_im_end),
            "include_im_end_in_cross_channel": bool(args.include_im_end_in_cross_channel),
            "allow_same_step_visible": bool(args.allow_same_step_visible),
            "disable_assistant_cross_channel": bool(args.disable_assistant_cross_channel),
            "user_instruction": args.user_instruction,
            "final_prefix": args.final_prefix,
        },
        "out_jsonl": args.out_jsonl,
    }

    os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
