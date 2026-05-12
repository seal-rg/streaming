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

import torch
import transformers

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

# =========================
# Medusa import
# =========================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3 import Qwen3ForMedusa  # noqa


# ============================================================
# Utils
# ============================================================

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


def sample_examples_list(examples: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(examples):
        return examples
    rng = random.Random(seed)
    idxs = rng.sample(range(len(examples)), n)
    return [examples[i] for i in idxs]


def sample_examples_hfds(ds_split, n: int, seed: int) -> List[Dict[str, Any]]:
    L = len(ds_split)
    if n <= 0 or n >= L:
        return [ds_split[i] for i in range(L)]
    rng = random.Random(seed)
    idxs = rng.sample(range(L), n)
    return [ds_split[i] for i in idxs]


# ============================================================
# JSONL loaders (SQuAD + ProofWriter meta wrapper)
# ============================================================

def load_jsonl_generic(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
            else:
                raise ValueError(f"jsonl line {line_no} is not a dict")
    return out


def load_squad_jsonl(path: str) -> List[Dict[str, Any]]:
    exs = load_jsonl_generic(path)
    # sanity (soft): allow extra fields but require these
    for i, ex in enumerate(exs):
        if not isinstance(ex, dict):
            raise ValueError(f"squad jsonl line {i+1} is not dict")
        if "context" not in ex or "question" not in ex or "answers" not in ex:
            raise ValueError(f"Invalid SQuAD jsonl line {i+1}: need context/question/answers")
    return exs


def load_proofwriter_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Supports both:
      A) each line is the actual sample dict (has theory/question/answer)
      B) each line is a wrapper dict with "data": {...sample...}  (your example)
    Returns list of sample dicts.
    """
    raw = load_jsonl_generic(path)
    out: List[Dict[str, Any]] = []
    for i, obj in enumerate(raw, 1):
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
            out.append(obj["data"])
        else:
            out.append(obj)
    # sanity (soft)
    for i, ex in enumerate(out):
        if not isinstance(ex, dict):
            raise ValueError(f"proofwriter jsonl item {i+1} not dict")
        if "theory" not in ex or "question" not in ex or "answer" not in ex:
            raise ValueError(f"Invalid ProofWriter jsonl item {i+1}: need theory/question/answer")
    return out


# ============================================================
# Prompt wrapping (your required format)
# ============================================================

def wrap_input_with_context(question_line: str, context_block: str) -> str:
    q = (question_line or "").strip()
    ctx = (context_block or "").strip()
    return (
        f"{q} based on the provided context below.\n\n"
        f"Context:\n{ctx}"
    )


# ============================================================
# Task-specific prompt builders
# ============================================================

# ---- SQuAD ----
def build_squad_input(ex: Dict[str, Any]) -> str:
    q = (ex.get("question", "") or "").strip()
    ctx = (ex.get("context", "") or "").strip()
    return wrap_input_with_context(q, ctx)

def get_squad_gold_answers(ex: Dict[str, Any]) -> List[str]:
    ans = ex.get("answers", {}) or {}
    texts = ans.get("text", [])
    if isinstance(texts, str):
        texts = [texts]
    return [str(x) for x in texts if str(x).strip()]


# ---- PubMedQA (pqa_labeled) ----
def pubmedqa_build_abstract(ex: Dict[str, Any]) -> str:
    ctx = ex.get("context", None)
    parts: List[str] = []

    def add(x):
        if not x:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                parts.append(s)
        elif isinstance(x, list):
            for y in x:
                add(y)
        elif isinstance(x, dict):
            add(x.get("contexts"))
        else:
            return

    add(ctx)
    if not parts:
        add(ex.get("long_answer", None))
    return "\n".join([p for p in parts if p])

def build_pubmedqa_input(ex: Dict[str, Any]) -> str:
    q = (ex.get("question", "") or "").strip()
    abstract = pubmedqa_build_abstract(ex)
    qline = f"{q} Answer with one of: Yes, No, or Maybe."
    return wrap_input_with_context(qline, abstract)

def get_pubmedqa_gold(ex: Dict[str, Any]) -> Optional[str]:
    g = ex.get("final_decision", None)
    if g is None:
        return None
    s = str(g).strip().lower().rstrip(".")
    return s if s in {"yes", "no", "maybe"} else None


# ---- ProofWriter ----
def _pw_premises_to_text(theory: Any) -> str:
    if isinstance(theory, list):
        return "\n".join([str(x).strip() for x in theory if str(x).strip()])
    return "\n".join([ln.strip() for ln in str(theory or "").splitlines() if ln.strip()])

def build_proofwriter_input(ex: Dict[str, Any]) -> str:
    # your required format uses quoted conclusion, but your examples show unquoted in some; keep quoted like HF code
    query = ex.get("question", ex.get("query", ex.get("conclusion", ""))) or ""
    theory = ex.get("theory", ex.get("premises", ex.get("context", "")))
    ctx = _pw_premises_to_text(theory)
    qline = f'Determine whether the conclusion "{str(query).strip()}" is true, false, or unknown.'
    return wrap_input_with_context(qline, ctx)

def get_proofwriter_gold(ex: Dict[str, Any]) -> Optional[str]:
    g = ex.get("answer", ex.get("label", ex.get("gold", None)))
    if g is None:
        return None
    s = str(g).strip().lower().rstrip(".")
    return s if s in {"true", "false", "unknown"} else None


# ---- LogicNLI ----
def build_logicnli_input(ex: Dict[str, Any]) -> str:
    premise = ex.get("premise", "") or ""
    hyp = ex.get("hypothesis", "") or ""
    qline = (
        f"Determine whether the hypothesis: {str(hyp).strip()} "
        f"is entailment, contradiction, neutral, or self_contradiction"
    )
    ctx = f"Premise: {str(premise).strip()}"
    return wrap_input_with_context(qline, ctx)

def get_logicnli_gold(ex: Dict[str, Any]) -> Optional[str]:
    g = ex.get("label", ex.get("gold", ex.get("answer", None)))
    if g is None:
        return None
    s = str(g).strip().lower()
    valid = {"entailment", "contradiction", "neutral", "self_contradiction"}
    return s if s in valid else None


# ============================================================
# SQuAD EM/F1
# ============================================================

_ARTICLES = {"a", "an", "the"}

def _normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    tokens = s.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens)

def _f1_score(pred: str, gold: str) -> float:
    pred_toks = _normalize_answer(pred).split()
    gold_toks = _normalize_answer(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0

def extract_qa_final_answer(text: str) -> str:
    """
    Robust SQuAD answer extraction:
      1) <ANSWER>...</ANSWER>
      2) "Final answer: ..." / "Answer: ..."
      3) last non-empty line
    """
    if not text:
        return ""
    t = text.strip()

    xml = list(re.finditer(r"(?is)<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t))
    if xml:
        return xml[-1].group(1).strip()

    labeled = list(re.finditer(r"(final\s*answer|answer)\s*[:：]\s*(.*)", t, re.I))
    if labeled:
        val = labeled[-1].group(2).strip()
        val = re.sub(r"(?is)</?\s*answer\s*>", "", val).strip()
        if val:
            return val.splitlines()[0].strip()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ============================================================
# GSM8K parsing (your old logic)
# ============================================================

_GSM_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
_GSM_LAST_NUM_RE = re.compile(r"([-+]?\d[\d,]*\.?\d*)")

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
    m = _GSM_ANSWER_RE.search(answer_text)
    if not m:
        return None
    return normalize_number_str(m.group(1))

def extract_pred_number(text: str) -> Optional[str]:
    if not text:
        return None
    nums = _GSM_LAST_NUM_RE.findall(text)
    if not nums:
        return None
    return normalize_number_str(nums[-1])

def is_correct_num(gold: Optional[str], pred: Optional[str]) -> bool:
    return (gold is not None) and (pred is not None) and (gold == pred)


# ============================================================
# Medusa Inference
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

        self.system_message = "You are a helpfulassistant."

        print("[Init] model_path:", model_path)
        print("[Init] device:", self.device)
        print("[Init] dtype:", self.dtype)

        self.model, _ = Qwen3ForMedusa.from_pretrained(
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

    def generate_1head(
        self,
        question_text: str,
        final_prefix: str = "",
        max_length_per_head: int = 256,
        max_steps: int = 2048,
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


# ============================================================
# Dataset loading per task (HF or JSONL)
# ============================================================

def load_examples(task: str, split: Optional[str], n: int, seed: int, in_jsonl: Optional[str]) -> List[Dict[str, Any]]:
    task = task.lower()
    if task == "squad":
        if in_jsonl:
            exs = load_squad_jsonl(in_jsonl)
            return sample_examples_list(exs, n, seed)
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad")
        use_split = split or ("validation" if "validation" in ds else pick_split(ds))
        return sample_examples_hfds(ds[use_split], n, seed)

    if task == "proofwriter":
        if in_jsonl:
            exs = load_proofwriter_jsonl(in_jsonl)
            return sample_examples_list(exs, n, seed)
        from datasets import load_dataset
        ds = load_dataset("tasksource/proofwriter")
        use_split = split or pick_split(ds)
        return sample_examples_hfds(ds[use_split], n, seed)

    if task == "logicnli":
        from datasets import load_dataset
        ds = load_dataset("tasksource/LogicNLI")
        use_split = split or pick_split(ds)
        return sample_examples_hfds(ds[use_split], n, seed)

    if task == "pubmedqa":
        from datasets import load_dataset
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        use_split = split or ("train" if "train" in ds else pick_split(ds))
        return sample_examples_hfds(ds[use_split], n, seed)

    if task == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")
        use_split = split or pick_split(ds)
        return sample_examples_hfds(ds[use_split], n, seed)

    raise ValueError(f"Unsupported task: {task}")


# ============================================================
# Evaluation per task (NO reflection)
# ============================================================

def eval_task(
    infer: MedusaInference,
    task: str,
    examples: List[Dict[str, Any]],
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
):
    task = task.lower()
    ensure_dir(os.path.dirname(output_jsonl) or ".")
    ensure_dir(os.path.dirname(output_summary_json) or ".")

    t0 = time.time()
    n = 0

    # metrics
    acc = 0
    em_sum = 0.0
    f1_sum = 0.0

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(examples):
            if task == "gsm8k":
                q = ex.get("question", "")
                gold = extract_gsm8k_gold(ex.get("answer", ""))
                prompt = q  # gsm8k 原始就是 question
                out = infer.generate_1head(
                    question_text=prompt,
                    final_prefix=final_prefix,
                    max_length_per_head=max_length_per_head,
                    max_steps=max_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    stop_on_im_end=stop_on_im_end,
                )
                pred = extract_pred_number(out)
                ok = is_correct_num(gold, pred)
                acc += int(ok)

                rec = {
                    "task": "gsm8k",
                    "index": i,
                    "prompt": prompt,
                    "gold_final": gold,
                    "pred_final": pred,
                    "correct": bool(ok),
                    "raw_full": out,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue

            if task == "squad":
                prompt = build_squad_input(ex)
                golds = get_squad_gold_answers(ex)

                out = infer.generate_1head(
                    question_text=prompt,
                    final_prefix=final_prefix,
                    max_length_per_head=max_length_per_head,
                    max_steps=max_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    stop_on_im_end=stop_on_im_end,
                )
                pred = extract_qa_final_answer(out)

                em_best = max((_exact_match(pred, g) for g in golds), default=0.0)
                f1_best = max((_f1_score(pred, g) for g in golds), default=0.0)
                em_sum += em_best
                f1_sum += f1_best

                rec = {
                    "task": "squad",
                    "index": i,
                    "id": ex.get("id", None),
                    "title": ex.get("title", None),
                    "prompt": prompt,
                    "gold": golds,
                    "pred": pred,
                    "em": em_best,
                    "f1": f1_best,
                    "raw_full": out,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue

            if task == "pubmedqa":
                prompt = build_pubmedqa_input(ex)
                gold = get_pubmedqa_gold(ex)

                out = infer.generate_1head(
                    question_text=prompt,
                    final_prefix=final_prefix,
                    max_length_per_head=max_length_per_head,
                    max_steps=max_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    stop_on_im_end=stop_on_im_end,
                )

                # you did NOT ask for strict format; we still need to extract a label.
                # Take last occurrence of yes/no/maybe in the output (robust).
                tail = (out or "").lower()[-1200:]
                m = re.findall(r"\b(yes|no|maybe)\b", tail)
                pred = m[-1] if m else None
                ok = (pred == gold) if (pred is not None and gold is not None) else False
                acc += int(ok)

                rec = {
                    "task": "pubmedqa",
                    "index": i,
                    "prompt": prompt,
                    "gold": gold,
                    "pred": pred,
                    "correct": bool(ok),
                    "raw_full": out,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue

            if task == "proofwriter":
                prompt = build_proofwriter_input(ex)
                gold = get_proofwriter_gold(ex)

                out = infer.generate_1head(
                    question_text=prompt,
                    final_prefix=final_prefix,
                    max_length_per_head=max_length_per_head,
                    max_steps=max_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    stop_on_im_end=stop_on_im_end,
                )

                tail = (out or "").lower()[-1200:]
                m = re.findall(r"\b(true|false|unknown)\b", tail)
                pred = m[-1] if m else None
                ok = (pred == gold) if (pred is not None and gold is not None) else False
                acc += int(ok)

                rec = {
                    "task": "proofwriter",
                    "index": i,
                    "id": ex.get("id", None),
                    "prompt": prompt,
                    "gold": gold,
                    "pred": pred,
                    "correct": bool(ok),
                    "raw_full": out,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue

            if task == "logicnli":
                prompt = build_logicnli_input(ex)
                gold = get_logicnli_gold(ex)

                out = infer.generate_1head(
                    question_text=prompt,
                    final_prefix=final_prefix,
                    max_length_per_head=max_length_per_head,
                    max_steps=max_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    stop_on_im_end=stop_on_im_end,
                )

                tail = (out or "").lower()[-1500:]
                # longest-first to avoid partial matches
                labels = ["self_contradiction", "contradiction", "entailment", "neutral"]
                pred = None
                for lab in labels:
                    if re.search(rf"\b{re.escape(lab)}\b", tail):
                        pred = lab
                ok = (pred == gold) if (pred is not None and gold is not None) else False
                acc += int(ok)

                rec = {
                    "task": "logicnli",
                    "index": i,
                    "prompt": prompt,
                    "gold": gold,
                    "pred": pred,
                    "correct": bool(ok),
                    "raw_full": out,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue

            raise ValueError(f"Unsupported task in eval loop: {task}")

    elapsed = time.time() - t0
    summary: Dict[str, Any] = {
        "task": task,
        "num_samples": n,
        "elapsed_sec": elapsed,
        "throughput_samples_per_sec": n / max(elapsed, 1e-9),
        "output_jsonl": output_jsonl,
    }
    if task == "squad":
        summary.update({
            "em": (em_sum / n) if n else 0.0,
            "f1": (f1_sum / n) if n else 0.0,
        })
    else:
        summary.update({
            "accuracy": acc / max(n, 1),
        })

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)

    ap.add_argument("--task", type=str, required=True,
                    choices=["gsm8k", "squad", "pubmedqa", "proofwriter", "logicnli"])
    ap.add_argument("--split", type=str, default=None, help="HF split (ignored for jsonl input)")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--in_jsonl", type=str, default=None,
                    help="optional local jsonl (for squad/proofwriter). "
                         "For proofwriter can be wrapper lines containing {'data': {...}}")

    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--output_jsonl", type=str, default=None)
    ap.add_argument("--output_summary_json", type=str, default=None)

    ap.add_argument("--final_prefix", type=str, default="")
    ap.add_argument("--max_length_per_head", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=2048)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--stop_on_im_end", action="store_true")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    args = ap.parse_args()
    seed_everything(args.seed)

    ensure_dir(args.out_dir)
    if args.output_jsonl is None:
        args.output_jsonl = os.path.join(args.out_dir, f"{args.task}.jsonl")
    if args.output_summary_json is None:
        args.output_summary_json = os.path.join(args.out_dir, f"{args.task}_summary.json")

    # Load data
    examples = load_examples(
        task=args.task,
        split=args.split,
        n=args.n,
        seed=args.seed,
        in_jsonl=args.in_jsonl,
    )
    print(f"[Load] task={args.task} n={len(examples)} source={'jsonl' if args.in_jsonl else 'huggingface'}")

    # Init infer
    infer = MedusaInference(args.model_path, device=args.device, torch_dtype=args.torch_dtype)

    # Run eval
    eval_task(
        infer=infer,
        task=args.task,
        examples=examples,
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
    )


if __name__ == "__main__":
    main()
