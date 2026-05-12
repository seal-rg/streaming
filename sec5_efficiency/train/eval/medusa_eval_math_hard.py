#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import time
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import time
import argparse
import random
import string
from typing import Any, Dict, List, Optional, Tuple
import warnings
import inspect

import torch
import transformers

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
import transformers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3 import Qwen3ForMedusa  # noqa


MEDUSA_MULTIHEAD_FN_NAME = "medusa_generate_interleaved_multihead_stream_user_same_y0"


def seed_everything(seed: int = 42):
    import numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed
        set_seed(seed)
    except Exception:
        pass


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pick_split(ds_dict, preferred=("test", "validation", "dev", "train")) -> str:
    for s in preferred:
        if s in ds_dict:
            return s
    return list(ds_dict.keys())[0]


def sample_examples(ds, n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(ds):
        return [ds[i] for i in range(len(ds))]
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), n)
    return [ds[i] for i in idxs]


def head_tail(text: str, head: int = 400, tail: int = 400) -> Tuple[str, str]:
    t = text or ""
    return (t[:head], t[-tail:] if len(t) > tail else t)


# ---------------- Boxed extraction ----------------
_BOXED_ANY_RE = re.compile(r"\\boxed\s*\{", re.I)


def _extract_all_boxed_payloads_anywhere(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    s = text

    for m in _BOXED_ANY_RE.finditer(s):
        j = m.end()
        if j <= 0 or s[j - 1] != "{":
            k = s.find("{", j)
            if k == -1:
                continue
            j = k + 1

        payload_start = j
        depth = 1
        k = payload_start
        while k < len(s) and depth > 0:
            ch = s[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth != 0:
            continue

        payload_end = k - 1
        payload = s[payload_start:payload_end].strip()
        if payload:
            out.append(payload)

    return out


def extract_final_boxed(text: str) -> Optional[str]:
    vals = _extract_all_boxed_payloads_anywhere(text)
    if not vals:
        return None
    bad = {"<value>", "value", "<answer>", "answer", "..."}
    for v in reversed(vals):
        v0 = v.strip()
        if v0.lower() in bad:
            continue
        if "<value>" in v0.lower():
            continue
        return v0
    return None


def extract_candidate_answer(text: str) -> str:
    if not text:
        return ""
    boxed = extract_final_boxed(text)
    if boxed is not None:
        return boxed.strip()

    labeled = list(re.finditer(r"(?is)(final\s*answer)\s*[:：]\s*(.+)", text))
    if labeled:
        return labeled[-1].group(2).strip().splitlines()[0].strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


# ---------------- math_verify ----------------
def mv_verify_pred(pred_text: str, gold_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        from math_verify import parse, verify
    except Exception:
        raise RuntimeError('Install: pip install "math-verify[antlr4_13_2]"')

    parsed_pred = parse(pred_text or "")
    parsed_gold = parse(gold_text or "")

    def best_candidate(parsed):
        if parsed is None:
            return None
        if not isinstance(parsed, list):
            return parsed
        sympy_like = None
        str_like = None
        for item in parsed:
            if isinstance(item, str):
                str_like = item
            else:
                sympy_like = item
        return sympy_like if sympy_like is not None else str_like

    pred_cand = best_candidate(parsed_pred)
    gold_cand = best_candidate(parsed_gold)

    if pred_cand is None or gold_cand is None:
        return (
            False,
            str(pred_cand) if pred_cand is not None else None,
            str(gold_cand) if gold_cand is not None else None,
        )

    ok = bool(verify(parsed_gold, parsed_pred))
    return (ok, str(pred_cand), str(gold_cand))


# ---------------- sampling defaults ----------------
def qwen_recommended_sampling(think_mode: str) -> Dict[str, Any]:
    think_mode = think_mode.lower()
    if think_mode == "think":
        return dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)
    return dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)


def _model_forces_non_think(model_name: str) -> bool:
    name = (model_name or "").lower()
    return ("instruct-2507" in name) or ("-instruct-2507" in name)


# ---------------- datasets ----------------
SUPPORTED = {
    "math-ai/aime24": ("math-ai/aime24", None),
    "math-ai/aime25": ("math-ai/aime25", None),
    "math-ai/minervamath": ("math-ai/minervamath", None),
    "math-ai/amc23": ("math-ai/amc23", None),
    "MathArena/hmmt_feb_2025": ("MathArena/hmmt_feb_2025", None),
    "MathArena/aime_2026": ("MathArena/aime_2026", None),
    "MathArena/hmmt_nov_2025": ("MathArena/hmmt_nov_2025", None),
}


def get_problem_gold_and_id(task: str, ex: Dict[str, Any]) -> Tuple[str, str, Optional[Any]]:
    if task == "math-ai/aime24":
        problem = str(ex.get("problem", "") or "")
        solution = str(ex.get("solution", "") or "")
        gold = extract_final_boxed(solution) or solution.strip()
        eid = ex.get("id", None)
        return problem, gold, eid

    if task == "math-ai/aime25":
        problem = str(ex.get("problem", "") or "")
        ans = str(ex.get("answer", "") or "")
        gold = extract_final_boxed(ans) or ans.strip()
        eid = ex.get("id", None)
        return problem, gold, eid

    if task == "math-ai/minervamath":
        problem = str(ex.get("question", "") or "")
        ans = str(ex.get("answer", "") or "")
        gold = extract_final_boxed(ans) or ans.strip()
        eid = ex.get("id", ex.get("idx", None))
        return problem, gold, eid

    if task == "math-ai/amc23":
        problem = str(ex.get("question", "") or "")
        ans = str(ex.get("answer", "") or "")
        gold = extract_final_boxed(ans) or ans.strip()
        eid = ex.get("id", None)
        return problem, gold, eid

    if task == "MathArena/hmmt_feb_2025":
        problem = str(ex.get("problem", "") or "")
        ans = str(ex.get("answer", "") or "")
        gold = extract_final_boxed(ans) or ans.strip()
        eid = ex.get("problem_idx", ex.get("id", None))
        return problem, gold, eid

    if task == "MathArena/aime_2026":
        problem = str(ex.get("problem", "") or "")
        ans = str(ex.get("answer", "") or "")
        gold = extract_final_boxed(ans) or ans.strip()
        eid = ex.get("problem_idx", ex.get("id", None))
        return problem, gold, eid

    if task == "MathArena/hmmt_nov_2025":
        problem = str(ex.get("problem", "") or "")
        ans = str(ex.get("answer", "") or "")
        gold = extract_final_boxed(ans) or ans.strip()
        eid = ex.get("problem_idx", ex.get("id", None))
        return problem, gold, eid

    raise ValueError(f"Unsupported task: {task}")


# ---------------- generator ----------------
@dataclass
class GenResult:
    text_head0: str
    text_by_head: Dict[int, str]
    prompt_tokens: int
    completion_tokens_head0: int
    completion_tokens_by_head: Dict[int, int]
    ttft_s: Optional[float]
    ttft_tokens: Optional[int]
    latency_s: float
    toks_per_s: Optional[float]
    prompt_raw: str
    prompt_formatted: str


class Medusa3HeadFirstOnly:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        assistant_heads: int = 3,
        seed: int = 42,
    ):
        torch.manual_seed(seed)

        self.model_name = model_path
        self.assistant_heads = int(assistant_heads)
        if self.assistant_heads != 3:
            raise ValueError("This script is configured for assistant_heads=3 as requested.")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype == "auto":
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        elif torch_dtype == "float16":
            self.dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        print("[Init] model_path:", model_path)
        print("[Init] device:", self.device)
        print("[Init] dtype:", self.dtype)
        print("[Init] medusa_fn:", MEDUSA_MULTIHEAD_FN_NAME)
        print("[Init] assistant_heads per forward:", self.assistant_heads)

        self.model, _ = Qwen3ForMedusa.from_pretrained(
            model_path,
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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._gen_fn = getattr(self.model, MEDUSA_MULTIHEAD_FN_NAME, None)
        if self._gen_fn is None:
            raise RuntimeError(f"Model does not have `{MEDUSA_MULTIHEAD_FN_NAME}`")

        try:
            self._gen_sig = inspect.signature(self._gen_fn)
            self._gen_params = set(self._gen_sig.parameters.keys())
        except Exception:
            self._gen_sig = None
            self._gen_params = None

        # match your reference defaults
        self.max_new_tokens = 256
        self.max_steps = 2048
        self.temperature = 0.7
        self.top_p = 0.8
        self.top_k = 20
        self.min_p = 0.0
        self.presence_penalty = 0.0
        self.do_sample = False

    def _format_prompt(self, prompt: str, think_mode: str, inject_think_tokens: bool) -> str:
        user_content = prompt
        if inject_think_tokens:
            if think_mode == "think":
                user_content = user_content.rstrip() + "\n/think"
            elif think_mode == "no_think":
                user_content = user_content.rstrip() + "\n/no_think"
        return user_content

    def _call_gen(self, **kwargs):
        if self._gen_params is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in self._gen_params}
        return self._gen_fn(**kwargs)

    @torch.no_grad()
    def generate(self, prompt: str, think_mode: str, inject_think_tokens: bool) -> GenResult:
        formatted = self._format_prompt(prompt, think_mode=think_mode, inject_think_tokens=inject_think_tokens)
        prompt_tokens = len(self.tokenizer.encode(formatted, add_special_tokens=False))
        print(f"[Generate] prompt_tokens={prompt_tokens} formatted_prompt={formatted[:200]}{'...' if len(formatted) > 200 else ''}")
        t0 = time.perf_counter()
        out = self._call_gen(
            question_text=formatted,
            assistant_heads=3,
            assistant_prefix_texts=["", "", ""],
            assistant_prefill_texts=["", "", ""],
            max_new_tokens=int(self.max_new_tokens),
            max_steps=int(self.max_steps),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            top_k=int(self.top_k),
            min_p=float(self.min_p),
            do_sample=bool(self.do_sample),
            stop_on_im_end=True,
            presence_penalty=float(self.presence_penalty),
            allow_same_step_visible=False,
        )
        t1 = time.perf_counter()

        completion_tokens_by_head: Dict[int, int] = {}
        text_by_head: Dict[int, str] = {}

        for h in (0, 1, 2):
            toks = out.get(h, None)
            if toks is None or toks.numel() == 0:
                completion_tokens_by_head[h] = 0
                text_by_head[h] = ""
            else:
                ids = toks.tolist()
                completion_tokens_by_head[h] = len(ids)
                text_by_head[h] = self.tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

        latency_s = float(t1 - t0)
        comp0 = int(completion_tokens_by_head.get(0, 0))
        toks_per_s = float(comp0 / max(1e-6, latency_s))

        return GenResult(
            text_head0=text_by_head.get(0, ""),
            text_by_head=text_by_head,
            prompt_tokens=int(prompt_tokens),
            completion_tokens_head0=int(comp0),
            completion_tokens_by_head=completion_tokens_by_head,
            ttft_s=None,
            ttft_tokens=None,
            latency_s=latency_s,
            toks_per_s=toks_per_s,
            prompt_raw=prompt,
            prompt_formatted=formatted,
        )


# ---------------- evaluator ----------------
def eval_avgk(
    gen: Medusa3HeadFirstOnly,
    task: str,
    examples: List[Dict[str, Any]],
    ks: List[int],
    out_jsonl: str,
    think_mode: str,
    inject_think_tokens: bool,
    base_seed: int,
    dump_full_text_by_head: bool,
) -> Dict[str, Any]:
    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks)

    hit_at = {k: 0 for k in ks}
    total = 0
    skipped = 0

    sum_prompt = 0
    sum_comp0 = 0
    sum_lat = 0.0
    sum_comp_by_head = {0: 0, 1: 0, 2: 0}

    ensure_dir(os.path.dirname(out_jsonl) or ".")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            problem, gold, eid = get_problem_gold_and_id(task, ex)
            if not str(problem).strip() or not str(gold).strip():
                skipped += 1
                continue

            prompt = problem  # keep raw question only

            ok_list: List[bool] = []
            cand_list: List[str] = []

            raw_headtail_by_head_list: List[Dict[int, Dict[str, str]]] = []
            completion_tokens_by_head_list: List[Dict[int, int]] = []
            text_by_head_list: List[Dict[int, str]] = []  # optional, controlled by flag

            per_sample_meta: List[Dict[str, Any]] = []

            for j in range(max_k):
                seed_j = base_seed + i * 1000 + j
                seed_everything(seed_j)

                r = gen.generate(prompt, think_mode=think_mode, inject_think_tokens=inject_think_tokens)

                sum_prompt += int(r.prompt_tokens)
                sum_comp0 += int(r.completion_tokens_head0)
                sum_lat += float(r.latency_s)
                for h in (0, 1, 2):
                    sum_comp_by_head[h] += int(r.completion_tokens_by_head.get(h, 0))

                cand = extract_candidate_answer(r.text_head0)
                ok, pred_best, gold_best = mv_verify_pred(cand, str(gold))

                ok_list.append(bool(ok))
                cand_list.append(cand)

                headtail_by_head: Dict[int, Dict[str, str]] = {}
                for h in (0, 1, 2):
                    htxt, ttxt = head_tail(r.text_by_head.get(h, ""))
                    headtail_by_head[h] = {"head": htxt, "tail": ttxt}

                raw_headtail_by_head_list.append(headtail_by_head)
                completion_tokens_by_head_list.append(dict(r.completion_tokens_by_head))

                if dump_full_text_by_head:
                    # store full decoded text per head for this forward
                    text_by_head_list.append({h: r.text_by_head.get(h, "") for h in (0, 1, 2)})

                per_sample_meta.append({
                    "seed": seed_j,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens_head0": r.completion_tokens_head0,
                    "completion_tokens_by_head": r.completion_tokens_by_head,
                    "latency_s": r.latency_s,
                    "toks_per_s": r.toks_per_s,
                    "pred_best": pred_best,
                    "gold_best": gold_best,
                })

            total += 1

            for k in ks:
                if any(ok_list[:k]):
                    hit_at[k] += 1

            rec: Dict[str, Any] = {
                "task": task,
                "example_id": eid,
                "gold": str(gold),
                "prompt": prompt,
                "ok_list": ok_list,
                "cand_list": cand_list,
                "raw_headtail_list_by_head": raw_headtail_by_head_list,
                "completion_tokens_by_head_list": completion_tokens_by_head_list,
                "per_sample_meta": per_sample_meta,
            }
            if dump_full_text_by_head:
                rec["text_by_head_list"] = text_by_head_list

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    denom_gen = max(1, total * max_k)
    res: Dict[str, Any] = {
        "task": task,
        "total": len(examples),
        "valid": total,
        "skipped": skipped,
        "ks": ks,
        "avg_prompt_tokens": sum_prompt / denom_gen,
        "avg_completion_tokens_head0": sum_comp0 / denom_gen,
        "avg_completion_tokens_by_head": {str(h): sum_comp_by_head[h] / denom_gen for h in (0, 1, 2)},
        "avg_total_tokens_head0": (sum_prompt + sum_comp0) / denom_gen,
        "avg_latency_s": sum_lat / denom_gen,
        "avg_ttft_s": None,
        "avg_ttft_tokens": None,
        "dump_full_text_by_head": bool(dump_full_text_by_head),
    }
    for k in ks:
        res[f"avg@{k}"] = hit_at[k] / max(1, total)
    return res


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=list(SUPPORTED.keys()))
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--n", type=int, default=0, help="Num samples (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")

    parser.add_argument("--think_mode", type=str, default="auto", choices=["auto", "think", "no_think"])
    parser.add_argument("--inject_think_tokens", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=2048)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--presence_penalty", type=float, default=None)
    parser.add_argument("--do_sample", action="store_true")

    parser.add_argument("--use_qwen_best_practices", action="store_true")

    parser.add_argument("--avg_ks", type=str, default="2,4,8")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    # ✅ NEW
    parser.add_argument(
        "--dump_full_text_by_head",
        action="store_true",
        help="Dump full decoded text for head0/1/2 into jsonl (can be very large).",
    )

    args = parser.parse_args()
    ensure_dir(args.out_dir)
    seed_everything(args.seed)

    ks: List[int] = []
    for x in (args.avg_ks or "").split(","):
        x = x.strip()
        if x:
            ks.append(int(x))
    if not ks:
        ks = [2, 4, 8]
    ks = sorted(set(ks))
    if max(ks) <= 0:
        raise ValueError("--avg_ks must contain positive integers.")

    task = args.task
    if task == "math-ai/minervamath":
        ks = [1]

    resolved_mode = args.think_mode
    if _model_forces_non_think(args.model):
        resolved_mode = "no_think"

    if args.use_qwen_best_practices:
        rec = qwen_recommended_sampling(resolved_mode)
        if args.temperature is None: args.temperature = rec["temperature"]
        if args.top_p is None: args.top_p = rec["top_p"]
        if args.top_k is None: args.top_k = rec["top_k"]
        if args.min_p is None: args.min_p = rec["min_p"]
        if args.presence_penalty is None: args.presence_penalty = rec["presence_penalty"]

    # match reference defaults
    if args.temperature is None: args.temperature = 0.7
    if args.top_p is None: args.top_p = 0.8
    if args.top_k is None: args.top_k = 20
    if args.min_p is None: args.min_p = 0.0
    if args.presence_penalty is None: args.presence_penalty = 0.0

    hf_name, hf_config = SUPPORTED[task]
    ds = load_dataset(hf_name, hf_config) if hf_config else load_dataset(hf_name)
    split = args.split or pick_split(ds)
    examples = sample_examples(ds[split], args.n, args.seed)

    gen = Medusa3HeadFirstOnly(
        model_path=args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        assistant_heads=3,
        seed=args.seed,
    )
    gen.max_new_tokens = int(args.max_new_tokens)
    gen.max_steps = int(args.max_steps)
    gen.temperature = float(args.temperature)
    gen.top_p = float(args.top_p)
    gen.top_k = int(args.top_k)
    gen.min_p = float(args.min_p)
    gen.presence_penalty = float(args.presence_penalty)
    gen.do_sample = bool(args.do_sample)

    run_tag = task.replace("/", "__")
    out_jsonl = os.path.join(args.out_dir, f"{run_tag}_{split}_avgk_medusa3h_head0.jsonl")
    summary_path = os.path.join(args.out_dir, f"{run_tag}_{split}_summary_medusa3h_head0.json")

    res = eval_avgk(
        gen=gen,
        task=task,
        examples=examples,
        ks=ks,
        out_jsonl=out_jsonl,
        think_mode=resolved_mode,
        inject_think_tokens=bool(args.inject_think_tokens),
        base_seed=args.seed,
        dump_full_text_by_head=bool(args.dump_full_text_by_head),
    )

    res["_args"] = {
        "model": args.model,
        "task": task,
        "split": split,
        "n": args.n,
        "seed": args.seed,
        "think_mode": resolved_mode,
        "inject_think_tokens": bool(args.inject_think_tokens),

        "assistant_heads_per_forward": 3,
        "judge_head_index": 0,
        "medusa_fn": MEDUSA_MULTIHEAD_FN_NAME,

        "max_new_tokens": gen.max_new_tokens,
        "max_steps": gen.max_steps,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "top_k": gen.top_k,
        "min_p": gen.min_p,
        "presence_penalty": gen.presence_penalty,
        "do_sample": gen.do_sample,

        "avg_ks": ks,
        "out_jsonl": out_jsonl,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "use_qwen_best_practices": bool(args.use_qwen_best_practices),
        "dump_full_text_by_head": bool(args.dump_full_text_by_head),
    }
    if not gen.do_sample:
        res["_warning"] = "do_sample=False => deterministic decoding; avg@k will not differ unless other stochasticity exists."

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"[Load] task={task} split={split} n={len(examples)}")
    print(f"[Saved jsonl]   {out_jsonl}")
    print(f"[Saved summary] {summary_path}")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
