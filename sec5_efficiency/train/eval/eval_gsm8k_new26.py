#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
import warnings

import torch
import transformers

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_medusa import Qwen2ForMedusa

_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
_LAST_NUM_RE = re.compile(r"([-+]?\d[\d,]*\.?\d*)")

def normalize_number_str(s: str) -> str:
    s = (s or "").strip().replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return str(int(s))
        if re.fullmatch(r"[-+]?\d*\.\d+", s):
            v = float(s)
            if v.is_integer():
                return str(int(v))
            return str(v)
    except Exception:
        pass
    return s

def extract_gsm8k_gold(answer_text: str) -> Optional[str]:
    if not answer_text:
        return None
    m = _ANSWER_RE.search(answer_text)
    if not m:
        return None
    return normalize_number_str(m.group(1))

def extract_pred_number(text: str) -> Optional[str]:
    if not text:
        return None
    nums = _LAST_NUM_RE.findall(text)
    if not nums:
        return None
    return normalize_number_str(nums[-1])

def is_correct(gold: Optional[str], pred: Optional[str]) -> bool:
    return (gold is not None) and (pred is not None) and (gold == pred)


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

    def build_prompt_1assistant(
        self,
        question_text: str,
        final_prefix: str = "",
    ) -> torch.Tensor:
        sys_ids = self.tokenizer.encode(self.system_message, add_special_tokens=False)
        usr_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
        pre_ids = self.tokenizer.encode(final_prefix, add_special_tokens=False) if final_prefix else []

        ids: List[int] = []
        # system
        ids += self.system_prefix + sys_ids + [self.im_end_id]
        # user (question)
        ids += self.user_prefix + usr_ids + [self.im_end_id]
        # assistant (open)
        ids += self.assistant_prefix + pre_ids
        return torch.tensor([ids], dtype=torch.long)

    def generate_1head(
        self,
        question_text: str,
        final_prefix: str = "",
        max_length_per_head: int = 1024,
        max_steps: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = False,
        stop_on_im_end: bool = True,
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
            include_im_end_in_cross_channel=True,
            allow_same_step_visible=False,
            disable_assistant_cross_channel=False,
        )

        toks = out.get(0, None)
        if toks is None or toks.numel() == 0:
            return ""
        return self.tokenizer.decode(toks.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)


def eval_gsm8k(
    infer: MedusaInference,
    split: str,
    max_samples: int,
    output_jsonl: str,
    output_summary_json: str,
    final_prefix: str,
    max_length_per_head: int,
    max_steps: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    stop_on_im_end: bool,
    verbose_input_decode: bool,
):
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    fout = open(output_jsonl, "w", encoding="utf-8")

    n, n_correct = 0, 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        q = ex["question"]
        gold_text = ex["answer"]
        gold = extract_gsm8k_gold(gold_text)

        text = infer.generate_1head(
            question_text=q,
            final_prefix=final_prefix,
            max_length_per_head=max_length_per_head,
            max_steps=max_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            stop_on_im_end=stop_on_im_end,
            #verbose_input_decode=verbose_input_decode,
        )

        pred = extract_pred_number(text)
        ok = is_correct(gold, pred)

        n += 1
        n_correct += int(ok)

        fout.write(json.dumps({
            "index": i,
            "question": q,
            "gold_final": gold,
            "pred_final": pred,
            "correct": bool(ok),
            "final_text": text,
        }, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"[GSM8K] {i+1}/{len(ds)} acc={n_correct/max(n,1):.4f}")

    fout.close()

    elapsed = time.time() - t0
    summary = {
        "dataset": "gsm8k",
        "split": split,
        "num_samples": n,
        "accuracy": n_correct / max(n, 1),
        "elapsed_sec": elapsed,
        "throughput_samples_per_sec": n / max(elapsed, 1e-9),
        "output_jsonl": output_jsonl,
        "config": {
            "final_prefix": final_prefix,
            "max_length_per_head": max_length_per_head,
            "max_steps": max_steps,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "stop_on_im_end": stop_on_im_end,
        }
    }
    os.makedirs(os.path.dirname(output_summary_json) or ".", exist_ok=True)
    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)

    ap.add_argument("--eval_gsm8k", action="store_true")
    ap.add_argument("--gsm8k_split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--max_samples", type=int, default=100)

    ap.add_argument("--final_prefix", type=str, default="Let me first understand the problem.")
    ap.add_argument("--max_length_per_head", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=2048)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--stop_on_im_end", action="store_true")
    ap.add_argument("--verbose_input_decode", action="store_true")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--output_jsonl", type=str, default="gsm8k_results.jsonl")
    ap.add_argument("--output_summary_json", type=str, default="gsm8k_summary.json")

    args = ap.parse_args()

    infer = MedusaInference(args.model_path, device=args.device, torch_dtype=args.torch_dtype)

    if args.eval_gsm8k:
        eval_gsm8k(
            infer=infer,
            split=args.gsm8k_split,
            max_samples=args.max_samples,
            output_jsonl=args.output_jsonl,
            output_summary_json=args.output_summary_json,
            final_prefix=args.final_prefix,
            max_length_per_head=args.max_length_per_head,
            max_steps=args.max_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            stop_on_im_end=args.stop_on_im_end,
            verbose_input_decode=args.verbose_input_decode,
        )

if __name__ == "__main__":
    main()
