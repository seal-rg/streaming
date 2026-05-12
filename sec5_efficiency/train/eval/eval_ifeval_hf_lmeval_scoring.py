# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# IFEval evaluation with:
#   - Generation: HuggingFace Transformers (chat template if available)
#   - Scoring: lm_eval's IFEval task (doc_to_text, process_results, aggregation)

# This mirrors your Medusa loop logic:
#   prompt = task.doc_to_text(doc)
#   pred   = generate(system_text + user prompt)
#   doc_metric_dict = task.process_results(doc, [pred])
#   agg_fns = task.aggregation()

# Install:
#   pip install -U "lm_eval" "transformers" "datasets" "torch" "accelerate" "tqdm" \
#                  "nltk" "langdetect" "immutabledict" "absl-py"

# Example:
#   python3 eval_ifeval_hf_lmeval_scoring.py \
#     --model ${RESULTS_ROOT}/p_20260217_230739_605921948 \
#     --out_jsonl ${DATA_ROOT}/para_out/ifeval_hf.jsonl \
#     --max_new_tokens 512 \
#     --temperature 0.0 \
#     --top_p 1.0
# """

# import os
# import json
# import math
# import argparse
# from typing import Any, Dict, List, Optional

# import numpy as np
# from tqdm import tqdm

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM


# # ----------------------------
# # Utils
# # ----------------------------
# def ensure_dir(p: str):
#     if p and not os.path.isdir(p):
#         os.makedirs(p, exist_ok=True)


# def strip_special_tokens_text(s: Any) -> str:
#     """
#     Minimal cleanup. Do NOT aggressively strip, or you may harm strict checks.
#     """
#     if s is None:
#         return ""
#     s = str(s)
#     # remove NUL, normalize trailing spaces
#     s = s.replace("\u0000", "")
#     return s.strip()


# def apply_until(pred: str, task_until) -> str:
#     if not task_until:
#         return pred
#     until_list = [task_until] if isinstance(task_until, str) else list(task_until)
#     cut = len(pred)
#     for u in until_list:
#         if not u:
#             continue
#         j = pred.find(u)
#         if j != -1:
#             cut = min(cut, j)
#     return pred[:cut]


# def binom_stderr(xs: List[float]) -> float:
#     if not xs:
#         return 0.0
#     n = len(xs)
#     p = float(np.mean(xs))
#     return math.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))


# # ----------------------------
# # HF generation
# # ----------------------------
# @torch.inference_mode()
# def hf_generate_system_user(
#     model,
#     tokenizer,
#     system_text: str,
#     user_text: str,
#     max_new_tokens: int = 512,
#     temperature: float = 0.0,
#     top_p: float = 1.0,
# ) -> str:
#     """
#     Chat-style generation. IMPORTANT:
#       - IFEval prompt goes to USER, not SYSTEM.
#       - system_text can be a fixed assistant instruction.

#     Uses tokenizer.apply_chat_template if present; else plain concat.
#     """
#     system_text = str(system_text)
#     user_text = str(user_text)

#     # messages = [
#     #     {"role": "system", "content": system_text},
#     #     {"role": "user", "content": user_text},
#     # ]
#     messages = [
#         {"role": "system", "content": user_text},
#         {"role": "user", "content": ""},
#     ]

#     if hasattr(tokenizer, "apply_chat_template"):
#         input_ids = tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt",
#         ).to(model.device)
#         attention_mask = torch.ones_like(input_ids, device=model.device)
#     else:
#         text = system_text.strip() + "\n\n" + user_text.strip()
#         enc = tokenizer(text, return_tensors="pt").to(model.device)
#         input_ids = enc["input_ids"]
#         attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

#     do_sample = float(temperature) > 0.0

#     out = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         max_new_tokens=int(max_new_tokens),
#         do_sample=do_sample,
#         temperature=float(temperature) if do_sample else None,
#         top_p=float(top_p) if do_sample else None,
#         pad_token_id=tokenizer.eos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     gen_ids = out[0, input_ids.shape[-1] :]
#     return tokenizer.decode(gen_ids, skip_special_tokens=True)


# # ----------------------------
# # Core eval (Medusa-like loop)
# # ----------------------------
# def eval_ifeval_with_hf_using_lmeval_scoring(
#     model_name_or_path: str,
#     out_jsonl: str,
#     gen_kwargs: Dict[str, Any],
#     system_text: str = "You are a helpful assistant.",
#     max_samples: int = 0,
#     subsample_seed: int = 0,
#     trust_remote_code: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Mirrors your Medusa eval loop but:
#       - generation: HF
#       - scoring: lm_eval's IFEval task
#     """
#     ensure_dir(os.path.dirname(out_jsonl) or ".")

#     # -----------------------
#     # Load lm_eval IFEval task (scoring)
#     # -----------------------
#     try:
#         from lm_eval.tasks import get_task_dict
#     except Exception as e:
#         raise RuntimeError(
#             "Missing lm_eval. Install with:\n"
#             "  pip install -U lm_eval\n"
#             f"Import error: {e}"
#         )

#     task_dict = get_task_dict(["ifeval"])
#     if "ifeval" not in task_dict:
#         raise RuntimeError("lm_eval did not return 'ifeval' task. Your lm_eval version may not include it.")
#     task = task_dict["ifeval"]

#     # -----------------------
#     # Get docs (version differences)
#     # -----------------------
#     docs = None
#     for attr in ("test_docs", "validation_docs", "eval_docs"):
#         if hasattr(task, attr):
#             try:
#                 docs = list(getattr(task, attr)())
#                 break
#             except Exception:
#                 docs = None
#     if docs is None:
#         raise RuntimeError("Cannot obtain IFEval docs from lm_eval task (test_docs/validation_docs/eval_docs all failed).")

#     # Optional subsample
#     if max_samples and 0 < max_samples < len(docs):
#         rng = np.random.RandomState(int(subsample_seed))
#         idx = rng.choice(len(docs), size=int(max_samples), replace=False)
#         docs = [docs[i] for i in idx]

#     # -----------------------
#     # Load HF model
#     # -----------------------
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path,
#         torch_dtype="auto",
#         device_map="auto",
#         trust_remote_code=trust_remote_code,
#     )
#     model.eval()

#     # -----------------------
#     # task "until" (stop strings)
#     # -----------------------
#     task_until = None
#     try:
#         if hasattr(task, "generation_kwargs"):
#             gk = task.generation_kwargs()
#             task_until = gk.get("until", None)
#     except Exception:
#         task_until = None

#     # -----------------------
#     # Generate + score per doc
#     # -----------------------
#     metrics_accum: Dict[str, List[float]] = {}

#     def _add_metric(metric_name: str, value: Any):
#         try:
#             v = float(value)
#         except Exception:
#             return
#         metrics_accum.setdefault(metric_name, []).append(v)

#     with open(out_jsonl, "w", encoding="utf-8") as f:
#         for i, doc in enumerate(tqdm(docs, desc="IFEval: HF gen + lm_eval scoring", leave=False)):
#             # Prompt text defined by task
#             try:
#                 prompt = task.doc_to_text(doc)
#             except Exception as e:
#                 raise RuntimeError(f"task.doc_to_text failed at i={i}: {e}")

#             # IMPORTANT: prompt in USER, not SYSTEM
#             raw = hf_generate_system_user(
#                 model=model,
#                 tokenizer=tokenizer,
#                 system_text=str(system_text),
#                 user_text=str(prompt),
#                 **gen_kwargs,
#             )
#             pred = strip_special_tokens_text(raw)

#             # Apply stop strings if provided
#             pred = apply_until(pred, task_until)

#             # Score via lm_eval task
#             try:
#                 doc_metric_dict = task.process_results(doc, [pred])
#             except Exception as e:
#                 raise RuntimeError(f"task.process_results failed at i={i}: {e}")

#             for k, v in doc_metric_dict.items():
#                 _add_metric(k, v)

#             rec = {
#                 "dataset": "ifeval",
#                 "i": i,
#                 "prompt": prompt,
#                 "pred": pred,
#                 "raw_full": raw,
#                 "doc_metrics": doc_metric_dict,
#             }
#             f.write(json.dumps(rec, ensure_ascii=False) + "\n")

#     # -----------------------
#     # Aggregate to final metrics (same as lm_eval)
#     # -----------------------
#     try:
#         agg_fns = task.aggregation()
#     except Exception as e:
#         raise RuntimeError(f"task.aggregation() failed: {e}")

#     final_results: Dict[str, float] = {}
#     for metric_name, values in metrics_accum.items():
#         if metric_name in agg_fns:
#             try:
#                 final_results[metric_name] = float(agg_fns[metric_name](values))
#             except Exception:
#                 final_results[metric_name] = float(np.mean(values)) if values else 0.0
#         else:
#             final_results[metric_name] = float(np.mean(values)) if values else 0.0

#     out = {
#         "dataset": "ifeval",
#         "num_samples": len(docs),
#         "inst_level_loose_acc": float(final_results.get("inst_level_loose_acc", 0.0)),
#         "inst_level_strict_acc": float(final_results.get("inst_level_strict_acc", 0.0)),
#         "prompt_level_loose_acc": float(final_results.get("prompt_level_loose_acc", 0.0)),
#         "prompt_level_strict_acc": float(final_results.get("prompt_level_strict_acc", 0.0)),
#     }

#     # stderr for prompt-level metrics (binomial)
#     if "prompt_level_loose_acc" in metrics_accum:
#         out["prompt_level_loose_acc_stderr"] = float(binom_stderr(metrics_accum["prompt_level_loose_acc"]))
#     if "prompt_level_strict_acc" in metrics_accum:
#         out["prompt_level_strict_acc_stderr"] = float(binom_stderr(metrics_accum["prompt_level_strict_acc"]))

#     # dump compact metrics json next to jsonl
#     details_path = os.path.splitext(out_jsonl)[0] + ".ifeval_metrics.json"
#     with open(details_path, "w", encoding="utf-8") as f:
#         json.dump(
#             {"final": out, "all_metrics_raw_counts": {k: len(v) for k, v in metrics_accum.items()}},
#             f,
#             ensure_ascii=False,
#             indent=2,
#         )
#     out["details_json"] = details_path
#     return out


# # ----------------------------
# # CLI
# # ----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", required=True, help="HF model name or local path")
#     ap.add_argument("--out_jsonl", required=True, help="Where to write per-sample records")
#     ap.add_argument("--system_text", default="You are a helpful assistant.")
#     ap.add_argument("--max_samples", type=int, default=0)
#     ap.add_argument("--subsample_seed", type=int, default=0)

#     # generation params
#     ap.add_argument("--max_new_tokens", type=int, default=512)
#     ap.add_argument("--temperature", type=float, default=0.0)
#     ap.add_argument("--top_p", type=float, default=1.0)

#     ap.add_argument("--no_trust_remote_code", action="store_true")
#     args = ap.parse_args()

#     gen_kwargs = dict(
#         max_new_tokens=int(args.max_new_tokens),
#         temperature=float(args.temperature),
#         top_p=float(args.top_p),
#     )

#     res = eval_ifeval_with_hf_using_lmeval_scoring(
#         model_name_or_path=args.model,
#         out_jsonl=args.out_jsonl,
#         gen_kwargs=gen_kwargs,
#         system_text=args.system_text,
#         max_samples=args.max_samples,
#         subsample_seed=args.subsample_seed,
#         trust_remote_code=(not args.no_trust_remote_code),
#     )
#     print(json.dumps(res, ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IFEval evaluation with:
  - Generation: HuggingFace Transformers (chat template if available)
  - Scoring: lm_eval's IFEval task (doc_to_text, process_results, aggregation)

This mirrors your Medusa loop logic:
  prompt = task.doc_to_text(doc)
  pred   = generate(system_text + user prompt)  (BUT you requested: prompt in SYSTEM)

IMPORTANT (per your request):
  - Keep current role logic:
      messages = [
        {"role": "system", "content": ifeval_prompt},
        {"role": "user", "content": ""},
      ]

Adds robust stopping for Qwen3 + Llama-2 style markers:
  - generation-time stop via StoppingCriteria (subsequence match)
  - decode-time truncation as safety net

Install:
  pip install -U "lm_eval" "transformers" "datasets" "torch" "accelerate" "tqdm" \
                 "nltk" "langdetect" "immutabledict" "absl-py"
"""

import os
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def strip_special_tokens_text(s: Any) -> str:
    """
    Minimal cleanup. Do NOT aggressively strip, or you may harm strict checks.
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u0000", "")
    return s.strip()


def apply_until(pred: str, task_until) -> str:
    """
    Cut at lm_eval task's stop strings (until) if present.
    """
    if not task_until:
        return pred
    until_list = [task_until] if isinstance(task_until, str) else list(task_until)
    cut = len(pred)
    for u in until_list:
        if not u:
            continue
        j = pred.find(u)
        if j != -1:
            cut = min(cut, j)
    return pred[:cut]


def binom_stderr(xs: List[float]) -> float:
    if not xs:
        return 0.0
    n = len(xs)
    p = float(np.mean(xs))
    return math.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))


def truncate_on_stop_strings(text: str, stop_strings: Sequence[str]) -> str:
    """
    String-level truncation (decode-time safety net).
    """
    if not text:
        return text
    cut = len(text)
    for s in stop_strings:
        if not s:
            continue
        j = text.find(s)
        if j != -1:
            cut = min(cut, j)
    return text[:cut]


def _safe_encode(tokenizer, s: str) -> List[int]:
    """
    Encode stop string into token ids.
    If it maps to a single UNK token, treat as missing.
    """
    try:
        ids = tokenizer.encode(s, add_special_tokens=False)
    except Exception:
        return []
    if ids and tokenizer.unk_token_id is not None and len(ids) == 1 and ids[0] == tokenizer.unk_token_id:
        return []
    return ids


# ----------------------------
# Stopping Criteria (subsequence)
# ----------------------------
class StopOnSubsequence(StoppingCriteria):
    """
    Stop when the sequence ends with any of the provided token-id subsequences.
    """
    def __init__(self, stop_seqs: List[List[int]]):
        super().__init__()
        self.stop_seqs = [s for s in stop_seqs if s]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_seqs:
            return False
        cur_len = input_ids.shape[-1]
        for seq in self.stop_seqs:
            L = len(seq)
            if L == 0 or cur_len < L:
                continue
            if input_ids[0, -L:].tolist() == seq:
                return True
        return False


def build_stop_markers_for_qwen3_llama2(tokenizer):
    """
    Markers we want to stop on.
    Qwen3 (ChatML): <|im_end|> is the key.
    Llama-2 sometimes emits </assistant> (depends on template/tooling), so include it as string stop.
    """
    stop_strings = [
        "<|im_end|>",        # Qwen/ChatML
        "<|eot_id|>",        # some templates
        "</assistant>",      # generic
        "<|assistant_end|>", # some templates
        "<|end|>",           # generic
    ]

    stop_seqs = []
    for s in stop_strings:
        ids = _safe_encode(tokenizer, s)
        if ids:
            stop_seqs.append(ids)

    # decode-time truncation strings (keep the same set + any tokenizer specials)
    decode_stops = list(stop_strings)
    if getattr(tokenizer, "eos_token", None):
        decode_stops.append(tokenizer.eos_token)
    for t in getattr(tokenizer, "additional_special_tokens", []) or []:
        if isinstance(t, str) and t:
            decode_stops.append(t)

    # de-dup while preserving order
    seen = set()
    decode_stops_dedup = []
    for s in decode_stops:
        if s not in seen:
            seen.add(s)
            decode_stops_dedup.append(s)

    return stop_seqs, decode_stops_dedup


# ----------------------------
# HF generation (KEEP YOUR ROLE LOGIC)
# ----------------------------
@torch.inference_mode()
def hf_generate_prompt_in_system(
    model,
    tokenizer,
    system_text: str,  # kept for signature compatibility; not used in chat_template branch
    ifeval_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
) -> str:
    """
    KEEP CURRENT ROLE LOGIC (as you requested):
      - put IFEval prompt in SYSTEM
      - user content is empty

    If tokenizer has apply_chat_template, we build messages accordingly.
    Else we fallback to concatenation (system_text + prompt), but your desired role logic
    is only meaningful for chat_template models.
    """
    system_text = str(system_text)
    ifeval_prompt = str(ifeval_prompt)

    messages = [
        {"role": "system", "content": ifeval_prompt},
        {"role": "user", "content": ""},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
    else:
        # Fallback (non-chat models): best-effort concat
        text = system_text.strip() + "\n\n" + ifeval_prompt.strip()
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    do_sample = float(temperature) > 0.0

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        top_p=float(top_p) if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,  # keep EOS as hard stop too
        stopping_criteria=stopping_criteria,
    )

    gen_ids = out[0, input_ids.shape[-1]:]
    # keep special tokens in decoded text so we can truncate on markers
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


# ----------------------------
# Core eval (Medusa-like loop)
# ----------------------------
def eval_ifeval_with_hf_using_lmeval_scoring(
    model_name_or_path: str,
    out_jsonl: str,
    gen_kwargs: Dict[str, Any],
    system_text: str = "You are a helpful assistant.",
    max_samples: int = 0,
    subsample_seed: int = 0,
    trust_remote_code: bool = True,
) -> Dict[str, Any]:
    ensure_dir(os.path.dirname(out_jsonl) or ".")

    # -----------------------
    # Load lm_eval IFEval task (scoring)
    # -----------------------
    try:
        from lm_eval.tasks import get_task_dict
    except Exception as e:
        raise RuntimeError(
            "Missing lm_eval. Install with:\n"
            "  pip install -U lm_eval\n"
            f"Import error: {e}"
        )

    task_dict = get_task_dict(["ifeval"])
    if "ifeval" not in task_dict:
        raise RuntimeError("lm_eval did not return 'ifeval' task. Your lm_eval version may not include it.")
    task = task_dict["ifeval"]

    # -----------------------
    # Get docs (version differences)
    # -----------------------
    docs = None
    for attr in ("test_docs", "validation_docs", "eval_docs"):
        if hasattr(task, attr):
            try:
                docs = list(getattr(task, attr)())
                break
            except Exception:
                docs = None
    if docs is None:
        raise RuntimeError("Cannot obtain IFEval docs from lm_eval task (test_docs/validation_docs/eval_docs all failed).")

    # Optional subsample
    if max_samples and 0 < max_samples < len(docs):
        rng = np.random.RandomState(int(subsample_seed))
        idx = rng.choice(len(docs), size=int(max_samples), replace=False)
        docs = [docs[i] for i in idx]

    # -----------------------
    # Load HF model
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # -----------------------
    # task "until" (stop strings)
    # -----------------------
    task_until = None
    try:
        if hasattr(task, "generation_kwargs"):
            gk = task.generation_kwargs()
            task_until = gk.get("until", None)
    except Exception:
        task_until = None

    # -----------------------
    # Build stopping (Qwen3/Llama-2)
    # -----------------------
    stop_seqs, decode_stop_strings = build_stop_markers_for_qwen3_llama2(tokenizer)
    stopping_criteria = StoppingCriteriaList([StopOnSubsequence(stop_seqs)]) if stop_seqs else None

    # -----------------------
    # Generate + score per doc
    # -----------------------
    metrics_accum: Dict[str, List[float]] = {}

    def _add_metric(metric_name: str, value: Any):
        try:
            v = float(value)
        except Exception:
            return
        metrics_accum.setdefault(metric_name, []).append(v)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, doc in enumerate(tqdm(docs, desc="IFEval: HF gen + lm_eval scoring", leave=False)):
            # Prompt text defined by task
            try:
                prompt = task.doc_to_text(doc)
            except Exception as e:
                raise RuntimeError(f"task.doc_to_text failed at i={i}: {e}")

            raw = hf_generate_prompt_in_system(
                model=model,
                tokenizer=tokenizer,
                system_text=str(system_text),
                ifeval_prompt=str(prompt),
                stopping_criteria=stopping_criteria,
                **gen_kwargs,
            )

            # Decode-time truncation first (remove chat markers tail)
            pred = truncate_on_stop_strings(raw, decode_stop_strings)

            # Minimal cleanup
            pred = strip_special_tokens_text(pred)

            # Apply lm_eval until (task-defined stop strings) if provided
            pred = apply_until(pred, task_until)

            # Score via lm_eval task
            try:
                doc_metric_dict = task.process_results(doc, [pred])
            except Exception as e:
                raise RuntimeError(f"task.process_results failed at i={i}: {e}")

            for k, v in doc_metric_dict.items():
                _add_metric(k, v)

            rec = {
                "dataset": "ifeval",
                "i": i,
                "prompt": prompt,
                "pred": pred,
                "raw_full": raw,
                "doc_metrics": doc_metric_dict,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -----------------------
    # Aggregate to final metrics (same as lm_eval)
    # -----------------------
    try:
        agg_fns = task.aggregation()
    except Exception as e:
        raise RuntimeError(f"task.aggregation() failed: {e}")

    final_results: Dict[str, float] = {}
    for metric_name, values in metrics_accum.items():
        if metric_name in agg_fns:
            try:
                final_results[metric_name] = float(agg_fns[metric_name](values))
            except Exception:
                final_results[metric_name] = float(np.mean(values)) if values else 0.0
        else:
            final_results[metric_name] = float(np.mean(values)) if values else 0.0

    out = {
        "dataset": "ifeval",
        "num_samples": len(docs),
        "inst_level_loose_acc": float(final_results.get("inst_level_loose_acc", 0.0)),
        "inst_level_strict_acc": float(final_results.get("inst_level_strict_acc", 0.0)),
        "prompt_level_loose_acc": float(final_results.get("prompt_level_loose_acc", 0.0)),
        "prompt_level_strict_acc": float(final_results.get("prompt_level_strict_acc", 0.0)),
    }

    # stderr for prompt-level metrics (binomial)
    if "prompt_level_loose_acc" in metrics_accum:
        out["prompt_level_loose_acc_stderr"] = float(binom_stderr(metrics_accum["prompt_level_loose_acc"]))
    if "prompt_level_strict_acc" in metrics_accum:
        out["prompt_level_strict_acc_stderr"] = float(binom_stderr(metrics_accum["prompt_level_strict_acc"]))

    # dump compact metrics json next to jsonl
    details_path = os.path.splitext(out_jsonl)[0] + ".ifeval_metrics.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(
            {"final": out, "all_metrics_raw_counts": {k: len(v) for k, v in metrics_accum.items()}},
            f,
            ensure_ascii=False,
            indent=2,
        )
    out["details_json"] = details_path
    return out


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--out_jsonl", required=True, help="Where to write per-sample records")
    ap.add_argument("--system_text", default="You are a helpful assistant.")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--subsample_seed", type=int, default=0)

    # generation params
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--no_trust_remote_code", action="store_true")
    args = ap.parse_args()

    gen_kwargs = dict(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    res = eval_ifeval_with_hf_using_lmeval_scoring(
        model_name_or_path=args.model,
        out_jsonl=args.out_jsonl,
        gen_kwargs=gen_kwargs,
        system_text=args.system_text,
        max_samples=args.max_samples,
        subsample_seed=args.subsample_seed,
        trust_remote_code=(not args.no_trust_remote_code),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()