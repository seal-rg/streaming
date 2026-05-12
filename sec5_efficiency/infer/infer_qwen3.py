#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import string
import time
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]


# ============================================================
# Utilities
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


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def pick_split(ds_dict, preferred=("test", "validation", "dev", "train")) -> str:
    for s in preferred:
        if s in ds_dict:
            return s
    return list(ds_dict.keys())[0]


def sample_examples(ds, n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(ds):
        return [ds[i] for i in range(len(ds))]
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), n)
    return [ds[i] for i in idxs]


def head_tail(text: str, head: int = 400, tail: int = 400) -> tuple[str, str]:
    t = text or ""
    return (t[:head], t[-tail:] if len(t) > tail else t)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ============================================================
# Gold label normalization
# ============================================================


def normalize_mcq_gold(gold: Any) -> str | None:
    """Convert MCQ gold to A/B/C/D/E/F."""
    if gold is None:
        return None
    if isinstance(gold, int):
        return CHOICE_LETTERS[gold] if 0 <= gold < len(CHOICE_LETTERS) else None

    s = str(gold).strip()
    if s.isdigit():
        idx = int(s)
        return CHOICE_LETTERS[idx] if 0 <= idx < len(CHOICE_LETTERS) else None

    m = re.search(r"\b([A-Fa-f])\b", s)
    if m:
        return m.group(1).upper()

    upper = s.upper()
    return upper if upper in CHOICE_LETTERS else None


def normalize_nli_gold(gold: Any) -> str | None:
    """Convert NLI gold to lowercase standard."""
    if gold is None:
        return None

    if isinstance(gold, bool):
        return "true" if gold else "false"
    if isinstance(gold, int) and gold in (0, 1):
        return "true" if gold == 1 else "false"

    s = str(gold).strip().lower()
    mapping = {
        "true": "true",
        "false": "false",
        "uncertain": "uncertain",
        "unknown": "unknown",
        # keep existing behavior for some tasks
        "yes": "true",
        "no": "false",
        "0": "false",
        "1": "true",
        "entailment": "entailment",
        "contradiction": "contradiction",
        "neutral": "neutral",
        "self_contradiction": "self_contradiction",
        "entailed": "true",
        "contradicted": "false",
        # ---- PubMedQA (pqa_labeled) ----
        # allow yes/no/maybe as their own labels too
        "maybe": "maybe",
        "yes.": "yes",
        "no.": "no",
        "maybe.": "maybe",
        "yes ": "yes",
        "no ": "no",
        "maybe ": "maybe",
    }
    if s in mapping:
        return mapping[s]

    valid = {
        "true",
        "false",
        "unknown",
        "uncertain",
        "entailment",
        "contradiction",
        "neutral",
        "self_contradiction",
        "yes",
        "no",
        "maybe",
    }
    return s if s in valid else None


# ============================================================
# Robust extraction (SF + fallback)
# ============================================================


def allowed_letters_from_max(max_opts: int) -> str:
    max_opts = max(1, min(max_opts, len(CHOICE_LETTERS)))
    return "".join(CHOICE_LETTERS[:max_opts])


# ---- MCQ ----


def extract_mcq_sf(text: str, allowed: str) -> str | None:
    """Strict-format extraction: any 'Final answer: X' anywhere; take last valid."""
    if not text:
        return None
    allowed_set = set(allowed.upper())
    vals = []
    for m in re.finditer(r"final\s*answer\s*[:：]\s*([A-F])\b", text, re.I):
        v = m.group(1).upper()
        if v in allowed_set:
            vals.append(v)
    return vals[-1] if vals else None


def extract_mcq_fallback(text: str, allowed: str) -> str | None:
    if not text:
        return None
    allowed_set = set(allowed.upper())
    t = text
    tail = t[-900:]

    m = re.search(r"(?:answer|final|correct|option|choice|select|choose)\s*(?:is)?\s*[:：]?\s*\(?\s*([A-F])\s*\)?\b", tail, re.I)
    if m:
        v = m.group(1).upper()
        if v in allowed_set:
            return v

    last = t.splitlines()[-1].strip() if t else ""
    m = re.fullmatch(r"\(?\s*([A-F])\s*\)?\.?", last, re.I)
    if m:
        v = m.group(1).upper()
        if v in allowed_set:
            return v

    matches = [mm.group(1).upper() for mm in re.finditer(r"\b([A-F])\b", tail, re.I)]
    for v in reversed(matches):
        if v in allowed_set:
            return v
    return None


# ---- NLI (Final label) ----


def extract_nli_sf(text: str, valid_labels: list[str]) -> str | None:
    if not text:
        return None
    valid = {x.lower() for x in valid_labels}
    t = text.lower()
    labels_pat = "|".join(map(re.escape, sorted(valid, key=len, reverse=True)))
    vals = []
    for m in re.finditer(rf"final\s*label\s*[:：]\s*({labels_pat})\b", t, re.I):
        v = m.group(1).lower()
        if v in valid:
            vals.append(v)
    return vals[-1] if vals else None


def extract_nli_fallback(text: str, valid_labels: list[str]) -> str | None:
    if not text:
        return None
    valid = {x.lower() for x in valid_labels}
    t = text.lower()
    tail = t[-1200:]
    labels_pat = "|".join(map(re.escape, sorted(valid, key=len, reverse=True)))

    m = re.search(rf"(?:answer|final|label|conclusion|result)\s*[:：]?\s*({labels_pat})\b", tail, re.I)
    if m:
        v = m.group(1).lower()
        if v in valid:
            return v

    last = t.splitlines()[-1].strip() if t else ""
    if last in valid:
        return last

    allm = []
    for mm in re.finditer(rf"\b({labels_pat})\b", tail, re.I):
        allm.append((mm.start(), mm.group(1).lower()))
    return sorted(allm, key=lambda x: x[0])[-1][1] if allm else None


# ---- ProofWriter strict + fallback ----
_PW_ANSWER_RE = re.compile(r"(?is)\banswer\b\s*(?::|=|-|is)\s*(true|false|unknown)\b")
_PW_LAST_LABEL_RE = re.compile(r"(?is)\b(true|false|unknown)\b")


def extract_pw_strict_answer(text: str) -> str | None:
    if not text:
        return None
    m = _PW_ANSWER_RE.search(text)
    return m.group(1).lower() if m else None


def extract_pw_lastlabel(text: str) -> str | None:
    if not text:
        return None
    allm = _PW_LAST_LABEL_RE.findall(text)
    return allm[-1].lower() if allm else None


# ---- StrategyQA strict + fallback (true/false) ----
_SQA_STRICT_RE = re.compile(r"(?is)\bfinal\s*answer\b\s*(?::|=|-|is)\s*(true|false)\b")
_SQA_LAST_RE = re.compile(r"(?is)\b(true|false)\b")


def extract_strategyqa_strict(text: str) -> str | None:
    if not text:
        return None
    m = _SQA_STRICT_RE.search(text)
    return m.group(1).lower() if m else None


def extract_strategyqa_last(text: str) -> str | None:
    if not text:
        return None
    allm = _SQA_LAST_RE.findall(text)
    return allm[-1].lower() if allm else None


# ---- Boxed SF helpers ----


def _find_all_boxed_payloads(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    lower = text.lower()

    for m in re.finditer(r"final\s*answer\s*[:：]", lower):
        i = m.end()
        boxed_idx = lower.find(r"\boxed", i)
        if boxed_idx == -1:
            continue

        j = boxed_idx + len(r"\boxed")
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "{":
            k = text.find("{", j)
            if k == -1:
                continue
            j = k

        payload_start = j + 1
        depth = 1
        k = payload_start
        while k < len(text) and depth > 0:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth != 0:
            continue

        payload_end = k - 1
        payload = text[payload_start:payload_end].strip()
        if payload:
            out.append(payload)
    return out


_BOXED_ANY_RE = re.compile(r"\\boxed\s*\{", re.I)


def _extract_all_boxed_payloads_anywhere(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    s = text

    for m in _BOXED_ANY_RE.finditer(s):
        j = m.end()
        # skip spaces
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            # our regex ends at "{", so this should rarely happen, but be defensive
            k = s.find("{", j)
            if k == -1:
                continue
            j = k

        payload_start = j + 1
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


def extract_final_boxed(text: str) -> str | None:
    """
    Return the LAST boxed payload that is not a placeholder like <VALUE>.
    Works even if model doesn't include 'Final answer:'.
    """
    vals = _extract_all_boxed_payloads_anywhere(text)
    if not vals:
        return None

    # Drop common placeholders the model may copy from prompt
    bad = {"<value>", "value", "<answer>", "answer", "..."}
    for v in reversed(vals):
        v0 = v.strip()
        if v0.lower() in bad:
            continue
        # if prompt got copied exactly: <VALUE> or \text{<VALUE>} etc.
        if "<value>" in v0.lower():
            continue
        return v0
    return None


# def extract_final_boxed(text: str) -> Optional[str]:
#     vals = _find_all_boxed_payloads(text)
#     return vals[-1] if vals else None


def extract_gsm8k_sf_number(text: str) -> str | None:
    payload = extract_final_boxed(text)
    if payload is None:
        return None
    num = payload.replace(",", "").strip()
    return num if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", num) else None


# ---- SQuAD strict/fallback extraction ----
def extract_qa_final_answer(text: str) -> str | None:
    if not text:
        return None

    t = text.strip()

    # 1) XML-style: <ANSWER>xxx</ANSWER>  (take LAST)
    xml = list(re.finditer(r"(?is)<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t))
    if xml:
        return xml[-1].group(1).strip()

    # 2) Labeled lines: Final answer: / Answer:
    label_pat = r"(final\s*answer|answer)\s*[:：]\s*(.*)"
    labeled = list(re.finditer(label_pat, t, re.I))
    if labeled:
        val = labeled[-1].group(2).strip()
        val = re.sub(r"(?is)</?\s*answer\s*>", "", val).strip()
        if val:
            return val.splitlines()[0].strip()

    # 3) Weak fallback: last non-empty line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return None


# ============================================================
# Scoring (fmt_ok + acc)
# ============================================================


def mcq_score(pred_text: str, gold: Any, max_opts: int) -> tuple[bool, bool, str | None, str | None, str | None]:
    gold_norm = normalize_mcq_gold(gold)
    if gold_norm is None:
        return (False, False, None, None, None)

    allowed = allowed_letters_from_max(max_opts)
    pred_sf = extract_mcq_sf(pred_text, allowed=allowed)
    fmt_ok = pred_sf is not None

    if pred_sf is not None:
        pred_used = pred_sf
        return (pred_used == gold_norm, fmt_ok, pred_used, pred_sf, None)

    pred_fb = extract_mcq_fallback(pred_text, allowed=allowed)
    pred_used = pred_fb
    acc_ok = (pred_fb == gold_norm) if pred_fb is not None else False
    return (acc_ok, fmt_ok, pred_used, None, pred_fb)


def nli_valid_labels(task: str) -> list[str]:
    task = task.lower()
    if task == "logicnli":
        return ["entailment", "contradiction", "neutral", "self_contradiction"]
    if task == "folio":
        return ["true", "false", "uncertain"]
    if task == "proofwriter":
        return ["true", "false", "unknown"]
    if task == "strategyqa":
        return ["true", "false"]
    if task == "pubmedqa":
        return ["yes", "no", "maybe"]
    raise ValueError(f"Unknown NLI task: {task}")


def normalize_nli_gold_task(gold: Any, task: str) -> str | None:
    """
    Task-aware gold normalization.

    Key fix:
      - For pubmedqa: keep yes/no/maybe as-is (DO NOT map yes->true).
      - For other NLI tasks: keep your existing behavior (yes->true, no->false, etc.).
    """
    task = (task or "").lower()

    if gold is None:
        return None

    # --- PubMedQA: keep yes/no/maybe ---
    if task == "pubmedqa":
        s = str(gold).strip().lower()
        # common variants
        s = s.rstrip(".")
        if s in {"yes", "no", "maybe"}:
            return s
        return None

    # --- otherwise: fall back to your old normalize_nli_gold behavior ---
    return normalize_nli_gold(gold)


def nli_score(pred_text: str, gold: Any, task: str) -> tuple[bool, bool, str | None, str | None, str | None]:
    gold_norm = normalize_nli_gold_task(gold, task=task)
    if gold_norm is None:
        return (False, False, None, None, None)

    task = task.lower()
    valid = set(nli_valid_labels(task))

    if task == "proofwriter":
        pred_strict = extract_pw_strict_answer(pred_text)
        fmt_ok = (pred_strict in valid) if pred_strict is not None else False
        if fmt_ok:
            return (pred_strict == gold_norm, True, pred_strict, pred_strict, None)

        pred_fb = extract_pw_lastlabel(pred_text)
        pred_used = pred_fb if (pred_fb in valid) else None
        acc_ok = (pred_used == gold_norm) if pred_used is not None else False
        return (acc_ok, False, pred_used, None, pred_used)

    if task == "strategyqa":
        pred_strict = extract_strategyqa_strict(pred_text)
        fmt_ok = (pred_strict in valid) if pred_strict is not None else False
        if fmt_ok:
            return (pred_strict == gold_norm, True, pred_strict, pred_strict, None)

        pred_fb = extract_strategyqa_last(pred_text)
        pred_used = pred_fb if (pred_fb in valid) else None
        acc_ok = (pred_used == gold_norm) if pred_used is not None else False
        return (acc_ok, False, pred_used, None, pred_used)

    pred_sf = extract_nli_sf(pred_text, valid_labels=list(valid))
    fmt_ok = pred_sf is not None
    if pred_sf is not None:
        return (pred_sf == gold_norm, True, pred_sf, pred_sf, None)

    pred_fb = extract_nli_fallback(pred_text, valid_labels=list(valid))
    pred_used = pred_fb
    acc_ok = (pred_fb == gold_norm) if pred_fb is not None else False
    return (acc_ok, False, pred_used, None, pred_fb)


# ============================================================
# Dataset parsing helpers
# ============================================================


def parse_mathqa_options(options_str: str) -> list[str]:
    if not options_str:
        return []
    s = str(options_str).strip()
    matches = re.findall(r"[a-e]\s*\)\s*([^,]+)", s, re.I)
    if matches and len(matches) >= 4:
        return [normalize_space(m) for m in matches[:5]]
    parts = [normalize_space(p) for p in s.split(",") if p.strip()]
    return parts[:5]


def arc_choices_to_options(choices: dict[str, Any]) -> list[str]:
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    pairs = list(zip(labels, texts))
    pairs.sort(key=lambda x: x[0])
    return [normalize_space(str(p[1])) for p in pairs]


# ---- PubMedQA helpers ----

# def pubmedqa_build_abstract(ex: Dict[str, Any]) -> str:
#     """
#     PubMedQA pqa_labeled fields:
#       - question: str
#       - context: Sequence[dict] (each dict often has "contexts": str)
#       - final_decision: "yes"/"no"/"maybe"
#       - long_answer: str
#       - pubid: int
#     We join all context[i]["contexts"] into one abstract.
#     """
#     ctx = ex.get("context", None)

#     if isinstance(ctx, list):
#         parts: List[str] = []
#         for item in ctx:
#             if isinstance(item, dict):
#                 s = item.get("contexts", "")
#                 if s:
#                     parts.append(str(s).strip())
#             elif isinstance(item, str):
#                 parts.append(item.strip())
#         return "\n".join([p for p in parts if p])

#     if isinstance(ctx, dict):
#         v = ctx.get("contexts", "")
#         return str(v).strip() if v else ""

#     return str(ctx or "").strip()
from typing import Any


def pubmedqa_build_abstract(ex: dict[str, Any]) -> str:
    """
    Robust PubMedQA abstract builder.

    Handles:
      - ex["context"] can be list[dict] | dict | list[str] | str | None
      - each dict may have "contexts" as str | list[str]
    Returns:
      abstract string, joined by '\n' for readability.
    """
    ctx = ex.get("context", None)
    parts: list[str] = []

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
            # ignore other types
            return

    add(ctx)

    # Fallback: use long_answer if context empty (common in some configs)
    if not parts:
        la = ex.get("long_answer", None)
        add(la)

    # Deduplicate empty and join
    parts = [p for p in parts if p]
    return "\n".join(parts)


def pubmedqa_get_gold(ex: dict[str, Any]) -> str | None:
    # gold label in final_decision (yes/no/maybe)
    g = ex.get("final_decision", None)
    if g is None:
        return None
    s = str(g).strip().lower()
    if s in {"yes", "no", "maybe"}:
        return s
    # defensive normalization
    ng = normalize_nli_gold(s)
    return ng if ng in {"yes", "no", "maybe"} else None


# ============================================================
# GSM8K + MATH verify
# ============================================================


def extract_gsm8k_gold(answer_field: str) -> str | None:
    if not answer_field:
        return None
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_field.replace(",", ""))
    if m:
        return m.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer_field.replace(",", ""))
    return nums[-1] if nums else None


def mv_verify_pred(pred_text: str, gold_text: str) -> tuple[bool, str | None, str | None]:
    try:
        from math_verify import parse, verify
    except Exception:
        raise RuntimeError('Install: pip install "math-verify[antlr4_13_2]"')

    pred_lines = (pred_text or "").splitlines()
    pred_tail = "\n".join(pred_lines[-12:]) if pred_lines else (pred_text or "")

    parsed_pred = parse(pred_tail)
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
        return (False, str(pred_cand) if pred_cand is not None else None, str(gold_cand) if gold_cand is not None else None)

    ok = bool(verify(parsed_gold, parsed_pred))
    return (ok, str(pred_cand), str(gold_cand))


# ============================================================
# Prompts (minimal) + think/non-think
# ============================================================


def _maybe_step_by_step(think_mode: str) -> str:
    return "Think step by step.\n\n" if think_mode == "think" else ""


def build_mc_prompt(question: str, options: list[str], max_opts: int, think_mode: str) -> str:
    opts = []
    for i in range(min(len(options), max_opts)):
        opts.append(f"{CHOICE_LETTERS[i]}. {options[i]}")
    value_spec = "one of " + ", ".join(CHOICE_LETTERS[:max_opts])

    return (
        f"Question: {question}\n\n"
        f"Options:\n" + "\n".join(opts) + "\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer as:\n"
        "Final answer: <VALUE>\n\n"
        f"<VALUE> must be {value_spec}.\n"
    )


def build_logiqa_prompt(context: str, question: str, options: list[str], max_opts: int, think_mode: str) -> str:
    opts = []
    for i in range(min(len(options), max_opts)):
        opts.append(f"{CHOICE_LETTERS[i]}. {options[i]}")
    value_spec = "one of " + ", ".join(CHOICE_LETTERS[:max_opts])

    ctx = (context or "").strip()
    q = (question or "").strip()

    return (
        f"Context: {ctx}\n\n"
        f"Question: {q}\n\n"
        f"Options:\n" + "\n".join(opts) + "\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer as:\n"
        "Final answer: <VALUE>\n\n"
        f"<VALUE> must be {value_spec}.\n"
    )


def build_proofwriter_problem(theory: str, query: str) -> str:
    premises = []
    if isinstance(theory, list):
        premises = [str(x).strip() for x in theory if str(x).strip()]
    else:
        premises = [ln.strip() for ln in str(theory or "").splitlines() if ln.strip()]
    conclusion = str(query or "").strip()
    return (
        f'Determine whether the conclusion "{conclusion}" is true, false, or unknown '
        f"based on the given premises below.\n\n"
        f"Premises:\n" + "\n".join(premises)
    )


def build_nli_prompt(premise: str, hypothesis: str, task: str, think_mode: str) -> str:
    task = task.lower()

    if task == "proofwriter":
        problem = build_proofwriter_problem(premise, hypothesis)
        return (
            f"{problem}\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer EXACTLY in this format:\n"
            "Answer:\n"
            "<VALUE>\n\n"
            "<VALUE> must be one of true, false, unknown.\n"
        )

    if task == "pubmedqa":
        valid = nli_valid_labels(task)
        value_spec = "one of " + ", ".join(valid)
        abstract = (premise or "").strip()
        question = (hypothesis or "").strip()
        return (
            f"Abstract:\n{abstract}\n\n"
            f"Question: {question}\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer as:\n"
            "Final label: <VALUE>\n\n"
            f"<VALUE> must be {value_spec}.\n"
        )

    valid = nli_valid_labels(task)
    value_spec = "one of " + ", ".join(valid)

    if task == "logicnli":
        head1, head2 = "Premise", "Hypothesis"
    elif task == "folio":
        head1, head2 = "Facts", "Conclusion"
    else:
        raise ValueError(f"Unknown NLI task: {task}")

    return (
        f"{head1}: {premise}\n\n"
        f"{head2}: {hypothesis}\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer as:\n"
        "Final label: <VALUE>\n\n"
        f"<VALUE> must be {value_spec}.\n"
    )


def build_strategyqa_prompt(question: str, facts: str | None, think_mode: str) -> str:
    q = (question or "").strip()
    fx = (facts or "").strip()
    return (
        f"Facts:\n{fx}\n\n"
        f"Question: {q}\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer as:\n"
        "Final answer: <VALUE>\n\n"
        "<VALUE> must be one of true, false.\n"
    )


def build_gsm8k_prompt(question: str, think_mode: str) -> str:
    return (
        f"Problem: {question}\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer as:\n"
        r"Final answer: \boxed{<VALUE>}"
        "\n\n"
        "<VALUE> must be a single integer or decimal number.\n"
    )


def build_math500_prompt(problem: str, think_mode: str) -> str:
    return (
        f"Problem: {problem}\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer as:\n"
        r"Final answer: \boxed{<VALUE>}"
        "\n\n"
        "<VALUE> must be the final simplified answer (may be an expression).\n"
    )


def build_squad_prompt(context: str, question: str, think_mode: str) -> str:
    ctx = (context or "").strip()
    q = (question or "").strip()
    return (
        f"Context:\n{ctx}\n\n"
        f"Question: {q}\n\n" + _maybe_step_by_step(think_mode) + "Output your final answer as:\n"
        "Final answer: <ANSWER>\n\n"
        "<ANSWER> must be the answer span from the context that best answers the question.\n"
    )


def build_reflection_prompt(task: str, original_prompt: str, first_response: str, think_mode: str, mcq_max_opts: int | None = None) -> str:
    task = task.lower()

    if task in {"mathqa", "logiqa", "logicqa", "mmlu_redux", "arc_c"}:
        assert mcq_max_opts is not None
        value_spec = "one of " + ", ".join(CHOICE_LETTERS[:mcq_max_opts])
        sf_block = f"Output your corrected final answer STRICTLY in this format:\nFinal answer: <VALUE>\n<VALUE> must be {value_spec}.\n"
    elif task in {"logicnli", "folio", "pubmedqa"}:
        valid = nli_valid_labels(task)
        value_spec = "one of " + ", ".join(valid)
        sf_block = f"Output your corrected final label STRICTLY in this format:\nFinal label: <VALUE>\n<VALUE> must be {value_spec}.\n"
    elif task == "proofwriter":
        sf_block = (
            "Output your corrected final answer STRICTLY in this format:\nAnswer:\n<VALUE>\n<VALUE> must be one of true, false, unknown.\n"
        )
    elif task in {"strategyqa", "strageqa"}:
        sf_block = (
            "Output your corrected final answer STRICTLY in this format:\nFinal answer: <VALUE>\n<VALUE> must be one of true, false.\n"
        )
    elif task == "gsm8k":
        sf_block = (
            "Output your corrected final answer STRICTLY in this format:\n"
            r"Final answer: \boxed{<VALUE>}"
            "\n"
            "<VALUE> must be a single integer or decimal number.\n"
        )
    elif task == "math500":
        sf_block = (
            "Output your corrected final answer STRICTLY in this format:\n"
            r"Final answer: \boxed{<VALUE>}"
            "\n"
            "<VALUE> must be the final simplified answer (may be an expression).\n"
        )
    elif task == "squad":
        sf_block = "Output your corrected final answer STRICTLY in this format:\nFinal answer: <ANSWER>\n"
    else:
        sf_block = "Output your final answer.\n"

    return (
        original_prompt.rstrip()
        + "\n\nPrevious response:\n"
        + (first_response or "")
        + "\n\nVerify your solution. If it is wrong, correct it.\n"
        + sf_block
    )


# ============================================================
# Qwen recommended sampling defaults
# ============================================================


def qwen_recommended_sampling(think_mode: str) -> dict[str, Any]:
    think_mode = think_mode.lower()
    if think_mode == "think":
        return dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)
    return dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)


def _model_forces_non_think(model_name: str) -> bool:
    name = (model_name or "").lower()
    return ("instruct-2507" in name) or ("-instruct-2507" in name)


# ============================================================
# Streaming helper: capture token ids and TTFT token count
# ============================================================


class CapturingStreamer(TextIteratorStreamer):
    """
    Safe wrapper for TextIteratorStreamer:
      - cache generated token ids via put() (robust, authoritative under streamer)
      - compute ttft_tokens at first NON-empty chunk
      - skip empty chunks by looping in __next__()
    """

    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self._generated_token_count = 0
        self._ttft_tokens = None
        self._gen_token_ids: list[int] = []

    def put(self, value):
        # value: token ids tensor (can be shape [1] or [1, k] depending on backend)
        try:
            if isinstance(value, torch.Tensor):
                flat = value.detach().view(-1).to("cpu").tolist()
                flat_int = []
                for x in flat:
                    try:
                        flat_int.append(int(x))
                    except Exception:
                        pass
                self._gen_token_ids.extend(flat_int)
                self._generated_token_count += len(flat_int)
        except Exception:
            pass
        return super().put(value)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            chunk = super().__next__()  # may raise StopIteration
            if chunk is None or chunk == "":
                continue
            if self._ttft_tokens is None:
                self._ttft_tokens = int(self._generated_token_count)
            return chunk

    @property
    def ttft_tokens(self):
        return self._ttft_tokens

    @property
    def gen_token_ids(self) -> list[int]:
        return self._gen_token_ids


# ============================================================
# Model
# ============================================================


@dataclass
class GenResult:
    text: str
    text_streamed: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None
    ttft_tokens: int | None
    latency_s: float
    toks_per_s: float | None
    prompt_raw: str
    prompt_formatted: str


class HFGenerator:
    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int = 42,
        think_mode: str = "auto",
        measure_ttft: bool = True,
        inject_think_tokens: bool = False,
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self.model_name = model
        self.think_mode = think_mode.lower()
        self.measure_ttft = bool(measure_ttft)
        self.inject_think_tokens = bool(inject_think_tokens)

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = "auto" if dtype == "auto" else torch.float16 if dtype == "fp16" else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_map,
            torch_dtype=torch_dtype,
        ).eval()

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.repetition_penalty = float(repetition_penalty)
        self.presence_penalty = float(presence_penalty)

        try:
            import inspect

            self._gen_sig = inspect.signature(self.model.generate)
        except Exception:
            self._gen_sig = None

        try:
            import inspect

            self._tmpl_sig = inspect.signature(self.tokenizer.apply_chat_template)
        except Exception:
            self._tmpl_sig = None

    def _supports_gen_kw(self, k: str) -> bool:
        if self._gen_sig is None:
            return True
        return k in self._gen_sig.parameters

    def _supports_template_kw(self, k: str) -> bool:
        if self._tmpl_sig is None:
            return False
        return k in self._tmpl_sig.parameters

    def _resolve_think_mode(self) -> str:
        if _model_forces_non_think(self.model_name):
            return "no_think"
        if self.think_mode in {"think", "no_think"}:
            return self.think_mode
        return "no_think"

    def _format_prompt(self, prompt: str) -> str:
        think_mode = self._resolve_think_mode()

        user_content = prompt
        if self.inject_think_tokens:
            if think_mode == "think":
                user_content = user_content.rstrip() + "\n/think"
            elif think_mode == "no_think":
                user_content = user_content.rstrip() + "\n/no_think"

        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ]
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            if self._supports_template_kw("enable_thinking") and not _model_forces_non_think(self.model_name):
                kwargs["enable_thinking"] = think_mode == "think"
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return self.tokenizer.apply_chat_template(messages, **kwargs)

        return user_content

    def _build_gen_kwargs(self) -> dict[str, Any]:
        do_sample = self.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.repetition_penalty != 1.0 and self._supports_gen_kw("repetition_penalty"):
            gen_kwargs["repetition_penalty"] = float(self.repetition_penalty)
        if self.presence_penalty != 0.0 and self._supports_gen_kw("presence_penalty"):
            gen_kwargs["presence_penalty"] = float(self.presence_penalty)

        if do_sample:
            if self._supports_gen_kw("temperature"):
                gen_kwargs["temperature"] = float(self.temperature)
            if self._supports_gen_kw("top_p"):
                gen_kwargs["top_p"] = float(self.top_p)
            if self.top_k > 0 and self._supports_gen_kw("top_k"):
                gen_kwargs["top_k"] = int(self.top_k)
            if self.min_p > 0.0 and self._supports_gen_kw("min_p"):
                gen_kwargs["min_p"] = float(self.min_p)
        return gen_kwargs

    @torch.inference_mode()
    def generate(self, prompt: str) -> GenResult:
        formatted = self._format_prompt(prompt)

        emb_dev = self.model.get_input_embeddings().weight.device
        toks = self.tokenizer(formatted, return_tensors="pt").to(emb_dev)
        prompt_len = int(toks["input_ids"].shape[1])

        gen_kwargs = self._build_gen_kwargs()

        if self._supports_gen_kw("return_dict_in_generate"):
            gen_kwargs["return_dict_in_generate"] = True
        if self._supports_gen_kw("output_scores"):
            gen_kwargs["output_scores"] = False

        if self.measure_ttft:
            streamer = CapturingStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs_stream = dict(gen_kwargs)
            gen_kwargs_stream["streamer"] = streamer

            import threading

            start_t = time.perf_counter()
            first_t = None
            pieces: list[str] = []
            holder: dict[str, Any] = {"out": None, "err": None}

            def _run():
                try:
                    holder["out"] = self.model.generate(**toks, **gen_kwargs_stream)
                except Exception as e:
                    holder["err"] = e

            th = threading.Thread(target=_run, daemon=True)
            th.start()

            for chunk in streamer:
                if first_t is None:
                    first_t = time.perf_counter()
                pieces.append(chunk)

            th.join()
            end_t = time.perf_counter()

            if holder["err"] is not None:
                raise holder["err"]

            gen_token_ids = streamer.gen_token_ids
            completion_tokens = int(len(gen_token_ids))

            decoded = self.tokenizer.decode(
                gen_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            streamed = "".join(pieces)

            ttft_s = (first_t - start_t) if first_t is not None else None
            latency_s = float(end_t - start_t)

            toks_per_s = None
            if ttft_s is not None:
                denom = max(1e-6, latency_s - ttft_s)
                toks_per_s = float(completion_tokens / denom)
            else:
                toks_per_s = float(completion_tokens / max(1e-6, latency_s))

            ttft_tokens = streamer.ttft_tokens
            if ttft_tokens is not None:
                ttft_tokens = int(min(ttft_tokens, completion_tokens))

            return GenResult(
                text=decoded,
                text_streamed=streamed,
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                ttft_s=ttft_s,
                ttft_tokens=ttft_tokens,
                latency_s=latency_s,
                toks_per_s=toks_per_s,
                prompt_raw=prompt,
                prompt_formatted=formatted,
            )

        start_t = time.perf_counter()
        out = self.model.generate(**toks, **gen_kwargs)
        end_t = time.perf_counter()

        try:
            seq = out.sequences[0]
        except Exception:
            seq = out[0] if isinstance(out, (list, tuple)) else out

        completion_tokens = int(max(0, int(seq.shape[0]) - prompt_len))
        gen_ids = seq[prompt_len:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        latency_s = float(end_t - start_t)
        toks_per_s = float(completion_tokens / max(1e-6, latency_s))

        return GenResult(
            text=decoded,
            text_streamed="",
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            ttft_s=None,
            ttft_tokens=None,
            latency_s=latency_s,
            toks_per_s=toks_per_s,
            prompt_raw=prompt,
            prompt_formatted=formatted,
        )


# ============================================================
# Evaluators
# ============================================================


def _accum_ttft(sum_ttft: float, cnt_ttft: int, r: GenResult) -> tuple[float, int]:
    if r.ttft_s is None:
        return sum_ttft, cnt_ttft
    return sum_ttft + float(r.ttft_s), cnt_ttft + 1


def _accum_ttft_tokens(sum_ttft_tok: int, cnt_ttft_tok: int, r: GenResult) -> tuple[int, int]:
    if r.ttft_tokens is None:
        return sum_ttft_tok, cnt_ttft_tok
    return sum_ttft_tok + int(r.ttft_tokens), cnt_ttft_tok + 1


def eval_mcq(
    gen: HFGenerator,
    examples: list[dict[str, Any]],
    task: str,
    get_data_fn,
    reflection: bool,
    out_path: str,
    max_opts: int,
    think_mode: str,
) -> dict[str, Any]:
    correct_1 = correct_2 = 0
    fmt_1 = fmt_2 = 0
    skipped = 0

    sum_prompt_1 = sum_comp_1 = 0
    sum_prompt_2 = sum_comp_2 = 0
    sum_ttft_1 = sum_lat_1 = 0.0
    sum_ttft_2 = sum_lat_2 = 0.0
    cnt_ttft_1 = cnt_ttft_2 = 0

    sum_ttft_tok_1 = sum_ttft_tok_2 = 0
    cnt_ttft_tok_1 = cnt_ttft_tok_2 = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            item = get_data_fn(ex)

            if not isinstance(item, (list, tuple)):
                raise ValueError("get_data_fn must return tuple/list")

            if len(item) == 3:
                q, opts, gold = item
                if normalize_mcq_gold(gold) is None:
                    skipped += 1
                    continue
                prompt1 = build_mc_prompt(q, opts, max_opts=max_opts, think_mode=think_mode)

            elif len(item) == 4:
                ctx, question, opts, gold = item
                if normalize_mcq_gold(gold) is None:
                    skipped += 1
                    continue
                prompt1 = build_logiqa_prompt(ctx, question, opts, max_opts=max_opts, think_mode=think_mode)

            else:
                raise ValueError("get_data_fn must return (q, opts, gold) or (ctx, question, opts, gold)")

            r1 = gen.generate(prompt1)

            sum_prompt_1 += r1.prompt_tokens
            sum_comp_1 += r1.completion_tokens
            sum_lat_1 += r1.latency_s
            sum_ttft_1, cnt_ttft_1 = _accum_ttft(sum_ttft_1, cnt_ttft_1, r1)
            sum_ttft_tok_1, cnt_ttft_tok_1 = _accum_ttft_tokens(sum_ttft_tok_1, cnt_ttft_tok_1, r1)

            ok1, fmtok1, pred_used1, pred_sf1, pred_fb1 = mcq_score(r1.text, gold, max_opts=max_opts)
            correct_1 += int(ok1)
            fmt_1 += int(fmtok1)

            h1, t1 = head_tail(r1.text)
            rec = {
                "task": task,
                "gold": str(gold),
                "ok_1": ok1,
                "fmt_1": fmtok1,
                "pred_1": pred_used1,
                "pred_sf_1": pred_sf1,
                "pred_fb_1": pred_fb1,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
                "total_tokens_1": r1.prompt_tokens + r1.completion_tokens,
                "ttft_s_1": r1.ttft_s,
                "ttft_tokens_1": r1.ttft_tokens,
                "latency_s_1": r1.latency_s,
                "toks_per_s_1": r1.toks_per_s,
                "prompt_1_raw": r1.prompt_raw,
                "prompt_1_formatted": r1.prompt_formatted,
                "prompt_chars_1_raw": len(r1.prompt_raw),
                "prompt_chars_1_formatted": len(r1.prompt_formatted),
                "raw_1_head": h1,
                "raw_1_tail": t1,
                "raw_1_full": r1.text,
                "raw_1_streamed": r1.text_streamed,
            }

            if reflection:
                prompt2 = build_reflection_prompt(task, prompt1, r1.text, think_mode=think_mode, mcq_max_opts=max_opts)
                r2 = gen.generate(prompt2)

                sum_prompt_2 += r2.prompt_tokens
                sum_comp_2 += r2.completion_tokens
                sum_lat_2 += r2.latency_s
                sum_ttft_2, cnt_ttft_2 = _accum_ttft(sum_ttft_2, cnt_ttft_2, r2)
                sum_ttft_tok_2, cnt_ttft_tok_2 = _accum_ttft_tokens(sum_ttft_tok_2, cnt_ttft_tok_2, r2)

                ok2, fmtok2, pred_used2, pred_sf2, pred_fb2 = mcq_score(r2.text, gold, max_opts=max_opts)
                correct_2 += int(ok2)
                fmt_2 += int(fmtok2)

                h2, t2 = head_tail(r2.text)
                rec.update(
                    {
                        "ok_2": ok2,
                        "fmt_2": fmtok2,
                        "pred_2": pred_used2,
                        "pred_sf_2": pred_sf2,
                        "pred_fb_2": pred_fb2,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                        "total_tokens_2": r2.prompt_tokens + r2.completion_tokens,
                        "ttft_s_2": r2.ttft_s,
                        "ttft_tokens_2": r2.ttft_tokens,
                        "latency_s_2": r2.latency_s,
                        "toks_per_s_2": r2.toks_per_s,
                        "prompt_2_raw": r2.prompt_raw,
                        "prompt_2_formatted": r2.prompt_formatted,
                        "prompt_chars_2_raw": len(r2.prompt_raw),
                        "prompt_chars_2_formatted": len(r2.prompt_formatted),
                        "raw_2_head": h2,
                        "raw_2_tail": t2,
                        "raw_2_full": r2.text,
                        "raw_2_streamed": r2.text_streamed,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples) - skipped
    res: dict[str, Any] = {
        "task": task,
        "total": len(examples),
        "valid": n,
        "skipped": skipped,
        "acc_1": correct_1 / n if n else 0.0,
        "fmt_1": fmt_1 / n if n else 0.0,
        "avg_prompt_tokens_1": (sum_prompt_1 / n) if n else 0.0,
        "avg_completion_tokens_1": (sum_comp_1 / n) if n else 0.0,
        "avg_total_tokens_1": ((sum_prompt_1 + sum_comp_1) / n) if n else 0.0,
        "avg_latency_s_1": (sum_lat_1 / n) if n else 0.0,
        "avg_ttft_s_1": (sum_ttft_1 / cnt_ttft_1) if cnt_ttft_1 else None,
        "avg_ttft_tokens_1": (sum_ttft_tok_1 / cnt_ttft_tok_1) if cnt_ttft_tok_1 else None,
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "fmt_2": fmt_2 / n if n else 0.0,
                "avg_prompt_tokens_2": (sum_prompt_2 / n) if n else 0.0,
                "avg_completion_tokens_2": (sum_comp_2 / n) if n else 0.0,
                "avg_total_tokens_2": ((sum_prompt_2 + sum_comp_2) / n) if n else 0.0,
                "avg_latency_s_2": (sum_lat_2 / n) if n else 0.0,
                "avg_ttft_s_2": (sum_ttft_2 / cnt_ttft_2) if cnt_ttft_2 else None,
                "avg_ttft_tokens_2": (sum_ttft_tok_2 / cnt_ttft_tok_2) if cnt_ttft_tok_2 else None,
            }
        )
    return res


def eval_nli(
    gen: HFGenerator, examples: list[dict[str, Any]], task: str, get_data_fn, reflection: bool, out_path: str, think_mode: str
) -> dict[str, Any]:
    correct_1 = correct_2 = 0
    fmt_1 = fmt_2 = 0
    skipped = 0

    sum_prompt_1 = sum_comp_1 = 0
    sum_prompt_2 = sum_comp_2 = 0
    sum_ttft_1 = sum_lat_1 = 0.0
    sum_ttft_2 = sum_lat_2 = 0.0
    cnt_ttft_1 = cnt_ttft_2 = 0

    sum_ttft_tok_1 = sum_ttft_tok_2 = 0
    cnt_ttft_tok_1 = cnt_ttft_tok_2 = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            premise, hyp, gold = get_data_fn(ex)
            if normalize_nli_gold_task(gold, task=task) is None:
                skipped += 1
                continue
            # if normalize_nli_gold(gold) is None:
            #     skipped += 1
            #     continue

            prompt1 = build_nli_prompt(premise, hyp, task, think_mode=think_mode)
            r1 = gen.generate(prompt1)

            sum_prompt_1 += r1.prompt_tokens
            sum_comp_1 += r1.completion_tokens
            sum_lat_1 += r1.latency_s
            sum_ttft_1, cnt_ttft_1 = _accum_ttft(sum_ttft_1, cnt_ttft_1, r1)
            sum_ttft_tok_1, cnt_ttft_tok_1 = _accum_ttft_tokens(sum_ttft_tok_1, cnt_ttft_tok_1, r1)

            ok1, fmtok1, pred_used1, pred_sf1, pred_fb1 = nli_score(r1.text, gold, task=task)
            correct_1 += int(ok1)
            fmt_1 += int(fmtok1)

            h1, t1 = head_tail(r1.text)
            rec = {
                "task": task,
                "gold": str(gold),
                "example_id": ex.get("pubid", ex.get("id", None)) if isinstance(ex, dict) else None,
                "ok_1": ok1,
                "fmt_1": fmtok1,
                "pred_1": pred_used1,
                "pred_sf_1": pred_sf1,
                "pred_fb_1": pred_fb1,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
                "total_tokens_1": r1.prompt_tokens + r1.completion_tokens,
                "ttft_s_1": r1.ttft_s,
                "ttft_tokens_1": r1.ttft_tokens,
                "latency_s_1": r1.latency_s,
                "toks_per_s_1": r1.toks_per_s,
                "prompt_1_raw": r1.prompt_raw,
                "prompt_1_formatted": r1.prompt_formatted,
                "prompt_chars_1_raw": len(r1.prompt_raw),
                "prompt_chars_1_formatted": len(r1.prompt_formatted),
                "raw_1_head": h1,
                "raw_1_tail": t1,
                "raw_1_full": r1.text,
                "raw_1_streamed": r1.text_streamed,
            }

            if reflection:
                prompt2 = build_reflection_prompt(task, prompt1, r1.text, think_mode=think_mode)
                r2 = gen.generate(prompt2)

                sum_prompt_2 += r2.prompt_tokens
                sum_comp_2 += r2.completion_tokens
                sum_lat_2 += r2.latency_s
                sum_ttft_2, cnt_ttft_2 = _accum_ttft(sum_ttft_2, cnt_ttft_2, r2)
                sum_ttft_tok_2, cnt_ttft_tok_2 = _accum_ttft_tokens(sum_ttft_tok_2, cnt_ttft_tok_2, r2)

                ok2, fmtok2, pred_used2, pred_sf2, pred_fb2 = nli_score(r2.text, gold, task=task)
                correct_2 += int(ok2)
                fmt_2 += int(fmtok2)

                h2, t2 = head_tail(r2.text)
                rec.update(
                    {
                        "ok_2": ok2,
                        "fmt_2": fmtok2,
                        "pred_2": pred_used2,
                        "pred_sf_2": pred_sf2,
                        "pred_fb_2": pred_fb2,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                        "total_tokens_2": r2.prompt_tokens + r2.completion_tokens,
                        "ttft_s_2": r2.ttft_s,
                        "ttft_tokens_2": r2.ttft_tokens,
                        "latency_s_2": r2.latency_s,
                        "toks_per_s_2": r2.toks_per_s,
                        "prompt_2_raw": r2.prompt_raw,
                        "prompt_2_formatted": r2.prompt_formatted,
                        "prompt_chars_2_raw": len(r2.prompt_raw),
                        "prompt_chars_2_formatted": len(r2.prompt_formatted),
                        "raw_2_head": h2,
                        "raw_2_tail": t2,
                        "raw_2_full": r2.text,
                        "raw_2_streamed": r2.text_streamed,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples) - skipped
    res: dict[str, Any] = {
        "task": task,
        "total": len(examples),
        "valid": n,
        "skipped": skipped,
        "acc_1": correct_1 / n if n else 0.0,
        "fmt_1": fmt_1 / n if n else 0.0,
        "avg_prompt_tokens_1": (sum_prompt_1 / n) if n else 0.0,
        "avg_completion_tokens_1": (sum_comp_1 / n) if n else 0.0,
        "avg_total_tokens_1": ((sum_prompt_1 + sum_comp_1) / n) if n else 0.0,
        "avg_latency_s_1": (sum_lat_1 / n) if n else 0.0,
        "avg_ttft_s_1": (sum_ttft_1 / cnt_ttft_1) if cnt_ttft_1 else None,
        "avg_ttft_tokens_1": (sum_ttft_tok_1 / cnt_ttft_tok_1) if cnt_ttft_tok_1 else None,
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "fmt_2": fmt_2 / n if n else 0.0,
                "avg_prompt_tokens_2": (sum_prompt_2 / n) if n else 0.0,
                "avg_completion_tokens_2": (sum_comp_2 / n) if n else 0.0,
                "avg_total_tokens_2": ((sum_prompt_2 + sum_comp_2) / n) if n else 0.0,
                "avg_latency_s_2": (sum_lat_2 / n) if n else 0.0,
                "avg_ttft_s_2": (sum_ttft_2 / cnt_ttft_2) if cnt_ttft_2 else None,
                "avg_ttft_tokens_2": (sum_ttft_tok_2 / cnt_ttft_tok_2) if cnt_ttft_tok_2 else None,
            }
        )
    return res


def eval_gsm8k(gen: HFGenerator, examples: list[dict[str, Any]], reflection: bool, out_path: str, think_mode: str) -> dict[str, Any]:
    correct_1 = correct_2 = 0
    fmt_1 = fmt_2 = 0

    sum_prompt_1 = sum_comp_1 = 0
    sum_prompt_2 = sum_comp_2 = 0
    sum_ttft_1 = sum_lat_1 = 0.0
    sum_ttft_2 = sum_lat_2 = 0.0
    cnt_ttft_1 = cnt_ttft_2 = 0

    sum_ttft_tok_1 = sum_ttft_tok_2 = 0
    cnt_ttft_tok_1 = cnt_ttft_tok_2 = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            q = ex["question"]
            gold_num = extract_gsm8k_gold(ex["answer"])
            gold_text = gold_num or ""

            prompt1 = build_gsm8k_prompt(q, think_mode=think_mode)
            r1 = gen.generate(prompt1)

            sum_prompt_1 += r1.prompt_tokens
            sum_comp_1 += r1.completion_tokens
            sum_lat_1 += r1.latency_s
            sum_ttft_1, cnt_ttft_1 = _accum_ttft(sum_ttft_1, cnt_ttft_1, r1)
            sum_ttft_tok_1, cnt_ttft_tok_1 = _accum_ttft_tokens(sum_ttft_tok_1, cnt_ttft_tok_1, r1)

            pred_sf1 = extract_gsm8k_sf_number(r1.text)
            fmtok1 = pred_sf1 is not None
            fmt_1 += int(fmtok1)

            verify_text_1 = pred_sf1 if pred_sf1 is not None else r1.text
            ok1, pred1_best, _ = mv_verify_pred(verify_text_1, gold_text)
            correct_1 += int(ok1)

            h1, t1 = head_tail(r1.text)
            rec = {
                "task": "gsm8k",
                "gold": gold_num,
                "ok_1": ok1,
                "fmt_1": fmtok1,
                "pred_1": pred_sf1 if pred_sf1 is not None else pred1_best,
                "pred_sf_1": pred_sf1,
                "pred_fb_1": None if pred_sf1 is not None else pred1_best,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
                "total_tokens_1": r1.prompt_tokens + r1.completion_tokens,
                "ttft_s_1": r1.ttft_s,
                "ttft_tokens_1": r1.ttft_tokens,
                "latency_s_1": r1.latency_s,
                "toks_per_s_1": r1.toks_per_s,
                "prompt_1_raw": r1.prompt_raw,
                "prompt_1_formatted": r1.prompt_formatted,
                "prompt_chars_1_raw": len(r1.prompt_raw),
                "prompt_chars_1_formatted": len(r1.prompt_formatted),
                "raw_1_head": h1,
                "raw_1_tail": t1,
                "raw_1_full": r1.text,
                "raw_1_streamed": r1.text_streamed,
            }

            if reflection:
                prompt2 = build_reflection_prompt("gsm8k", prompt1, r1.text, think_mode=think_mode)
                r2 = gen.generate(prompt2)

                sum_prompt_2 += r2.prompt_tokens
                sum_comp_2 += r2.completion_tokens
                sum_lat_2 += r2.latency_s
                sum_ttft_2, cnt_ttft_2 = _accum_ttft(sum_ttft_2, cnt_ttft_2, r2)
                sum_ttft_tok_2, cnt_ttft_tok_2 = _accum_ttft_tokens(sum_ttft_tok_2, cnt_ttft_tok_2, r2)

                pred_sf2 = extract_gsm8k_sf_number(r2.text)
                fmtok2 = pred_sf2 is not None
                fmt_2 += int(fmtok2)

                verify_text_2 = pred_sf2 if pred_sf2 is not None else r2.text
                ok2, pred2_best, _ = mv_verify_pred(verify_text_2, gold_text)
                correct_2 += int(ok2)

                h2, t2 = head_tail(r2.text)
                rec.update(
                    {
                        "ok_2": ok2,
                        "fmt_2": fmtok2,
                        "pred_2": pred_sf2 if pred_sf2 is not None else pred2_best,
                        "pred_sf_2": pred_sf2,
                        "pred_fb_2": None if pred_sf2 is not None else pred2_best,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                        "total_tokens_2": r2.prompt_tokens + r2.completion_tokens,
                        "ttft_s_2": r2.ttft_s,
                        "ttft_tokens_2": r2.ttft_tokens,
                        "latency_s_2": r2.latency_s,
                        "toks_per_s_2": r2.toks_per_s,
                        "prompt_2_raw": r2.prompt_raw,
                        "prompt_2_formatted": r2.prompt_formatted,
                        "prompt_chars_2_raw": len(r2.prompt_raw),
                        "prompt_chars_2_formatted": len(r2.prompt_formatted),
                        "raw_2_head": h2,
                        "raw_2_tail": t2,
                        "raw_2_full": r2.text,
                        "raw_2_streamed": r2.text_streamed,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples)
    res: dict[str, Any] = {
        "task": "gsm8k",
        "total": n,
        "valid": n,
        "acc_1": correct_1 / n if n else 0.0,
        "fmt_1": fmt_1 / n if n else 0.0,
        "avg_prompt_tokens_1": (sum_prompt_1 / n) if n else 0.0,
        "avg_completion_tokens_1": (sum_comp_1 / n) if n else 0.0,
        "avg_total_tokens_1": ((sum_prompt_1 + sum_comp_1) / n) if n else 0.0,
        "avg_latency_s_1": (sum_lat_1 / n) if n else 0.0,
        "avg_ttft_s_1": (sum_ttft_1 / cnt_ttft_1) if cnt_ttft_1 else None,
        "avg_ttft_tokens_1": (sum_ttft_tok_1 / cnt_ttft_tok_1) if cnt_ttft_tok_1 else None,
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "fmt_2": fmt_2 / n if n else 0.0,
                "avg_prompt_tokens_2": (sum_prompt_2 / n) if n else 0.0,
                "avg_completion_tokens_2": (sum_comp_2 / n) if n else 0.0,
                "avg_total_tokens_2": ((sum_prompt_2 + sum_comp_2) / n) if n else 0.0,
                "avg_latency_s_2": (sum_lat_2 / n) if n else 0.0,
                "avg_ttft_s_2": (sum_ttft_2 / cnt_ttft_2) if cnt_ttft_2 else None,
                "avg_ttft_tokens_2": (sum_ttft_tok_2 / cnt_ttft_tok_2) if cnt_ttft_tok_2 else None,
            }
        )
    return res


def eval_math500(gen: HFGenerator, examples: list[dict[str, Any]], reflection: bool, out_path: str, think_mode: str) -> dict[str, Any]:
    correct_1 = correct_2 = 0
    fmt_1 = fmt_2 = 0
    skipped = 0

    sum_prompt_1 = sum_comp_1 = 0
    sum_prompt_2 = sum_comp_2 = 0
    sum_ttft_1 = sum_lat_1 = 0.0
    sum_ttft_2 = sum_lat_2 = 0.0
    cnt_ttft_1 = cnt_ttft_2 = 0

    sum_ttft_tok_1 = sum_ttft_tok_2 = 0
    cnt_ttft_tok_1 = cnt_ttft_tok_2 = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            problem = ex.get("problem", "")
            gold = ex.get("answer", None)
            if gold is None or str(gold).strip() == "":
                skipped += 1
                continue

            prompt1 = build_math500_prompt(problem, think_mode=think_mode)
            r1 = gen.generate(prompt1)

            sum_prompt_1 += r1.prompt_tokens
            sum_comp_1 += r1.completion_tokens
            sum_lat_1 += r1.latency_s
            sum_ttft_1, cnt_ttft_1 = _accum_ttft(sum_ttft_1, cnt_ttft_1, r1)
            sum_ttft_tok_1, cnt_ttft_tok_1 = _accum_ttft_tokens(sum_ttft_tok_1, cnt_ttft_tok_1, r1)

            pred_sf1 = extract_final_boxed(r1.text)
            fmtok1 = pred_sf1 is not None
            fmt_1 += int(fmtok1)

            verify_text_1 = pred_sf1 if pred_sf1 is not None else r1.text
            ok1, pred1_best, _ = mv_verify_pred(verify_text_1, str(gold))
            correct_1 += int(ok1)

            rec = {
                "task": "math500",
                "gold": str(gold),
                "ok_1": ok1,
                "fmt_1": fmtok1,
                "pred_1": pred_sf1 if pred_sf1 is not None else pred1_best,
                "pred_sf_1": pred_sf1,
                "pred_fb_1": None if pred_sf1 is not None else pred1_best,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
                "total_tokens_1": r1.prompt_tokens + r1.completion_tokens,
                "ttft_s_1": r1.ttft_s,
                "ttft_tokens_1": r1.ttft_tokens,
                "latency_s_1": r1.latency_s,
                "toks_per_s_1": r1.toks_per_s,
                "prompt_1_raw": r1.prompt_raw,
                "prompt_1_formatted": r1.prompt_formatted,
                "prompt_chars_1_raw": len(r1.prompt_raw),
                "prompt_chars_1_formatted": len(r1.prompt_formatted),
                "raw_1_full": r1.text,
                "raw_1_streamed": r1.text_streamed,
            }

            if reflection:
                prompt2 = build_reflection_prompt("math500", prompt1, r1.text, think_mode=think_mode)
                r2 = gen.generate(prompt2)

                sum_prompt_2 += r2.prompt_tokens
                sum_comp_2 += r2.completion_tokens
                sum_lat_2 += r2.latency_s
                sum_ttft_2, cnt_ttft_2 = _accum_ttft(sum_ttft_2, cnt_ttft_2, r2)
                sum_ttft_tok_2, cnt_ttft_tok_2 = _accum_ttft_tokens(sum_ttft_tok_2, cnt_ttft_tok_2, r2)

                pred_sf2 = extract_final_boxed(r2.text)
                fmtok2 = pred_sf2 is not None
                fmt_2 += int(fmtok2)

                verify_text_2 = pred_sf2 if pred_sf2 is not None else r2.text
                ok2, pred2_best, _ = mv_verify_pred(verify_text_2, str(gold))
                correct_2 += int(ok2)

                rec.update(
                    {
                        "ok_2": ok2,
                        "fmt_2": fmtok2,
                        "pred_2": pred_sf2 if pred_sf2 is not None else pred2_best,
                        "pred_sf_2": pred_sf2,
                        "pred_fb_2": None if pred_sf2 is not None else pred2_best,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                        "total_tokens_2": r2.prompt_tokens + r2.completion_tokens,
                        "ttft_s_2": r2.ttft_s,
                        "ttft_tokens_2": r2.ttft_tokens,
                        "latency_s_2": r2.latency_s,
                        "toks_per_s_2": r2.toks_per_s,
                        "prompt_2_raw": r2.prompt_raw,
                        "prompt_2_formatted": r2.prompt_formatted,
                        "prompt_chars_2_raw": len(r2.prompt_raw),
                        "prompt_chars_2_formatted": len(r2.prompt_formatted),
                        "raw_2_full": r2.text,
                        "raw_2_streamed": r2.text_streamed,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples) - skipped
    res: dict[str, Any] = {
        "task": "math500",
        "total": len(examples),
        "valid": n,
        "skipped": skipped,
        "acc_1": correct_1 / n if n else 0.0,
        "fmt_1": fmt_1 / n if n else 0.0,
        "avg_prompt_tokens_1": (sum_prompt_1 / n) if n else 0.0,
        "avg_completion_tokens_1": (sum_comp_1 / n) if n else 0.0,
        "avg_total_tokens_1": ((sum_prompt_1 + sum_comp_1) / n) if n else 0.0,
        "avg_latency_s_1": (sum_lat_1 / n) if n else 0.0,
        "avg_ttft_s_1": (sum_ttft_1 / cnt_ttft_1) if cnt_ttft_1 else None,
        "avg_ttft_tokens_1": (sum_ttft_tok_1 / cnt_ttft_tok_1) if cnt_ttft_tok_1 else None,
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "fmt_2": fmt_2 / n if n else 0.0,
                "avg_prompt_tokens_2": (sum_prompt_2 / n) if n else 0.0,
                "avg_completion_tokens_2": (sum_comp_2 / n) if n else 0.0,
                "avg_total_tokens_2": ((sum_prompt_2 + sum_comp_2) / n) if n else 0.0,
                "avg_latency_s_2": (sum_lat_2 / n) if n else 0.0,
                "avg_ttft_s_2": (sum_ttft_2 / cnt_ttft_2) if cnt_ttft_2 else None,
                "avg_ttft_tokens_2": (sum_ttft_tok_2 / cnt_ttft_tok_2) if cnt_ttft_tok_2 else None,
            }
        )
    return res


def _strategyqa_get_question(ex: dict[str, Any]) -> str:
    for k in ("question", "query", "input", "prompt", "q"):
        if k in ex and ex[k]:
            return str(ex[k])
    return ""


def _strategyqa_get_facts(ex: dict[str, Any]) -> str | None:
    for k in ("facts", "fact", "context", "evidence", "explanation"):
        if k in ex and ex[k]:
            v = ex[k]
            if isinstance(v, list):
                return "\n".join([str(x) for x in v])
            return str(v)
    return None


def _strategyqa_get_gold(ex: dict[str, Any]) -> str | None:
    for k in ("answer", "label", "gold", "target"):
        if k in ex:
            return normalize_nli_gold(ex[k])
    return None


def eval_strategyqa(gen: HFGenerator, examples: list[dict[str, Any]], reflection: bool, out_path: str, think_mode: str) -> dict[str, Any]:
    correct_1 = correct_2 = 0
    fmt_1 = fmt_2 = 0
    skipped = 0

    sum_prompt_1 = sum_comp_1 = 0
    sum_prompt_2 = sum_comp_2 = 0
    sum_ttft_1 = sum_lat_1 = 0.0
    sum_ttft_2 = sum_lat_2 = 0.0
    cnt_ttft_1 = cnt_ttft_2 = 0

    sum_ttft_tok_1 = sum_ttft_tok_2 = 0
    cnt_ttft_tok_1 = cnt_ttft_tok_2 = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            q = _strategyqa_get_question(ex)
            facts = _strategyqa_get_facts(ex)
            gold_norm = _strategyqa_get_gold(ex)
            if gold_norm not in {"true", "false"}:
                skipped += 1
                continue

            prompt1 = build_strategyqa_prompt(q, facts=facts, think_mode=think_mode)
            r1 = gen.generate(prompt1)

            sum_prompt_1 += r1.prompt_tokens
            sum_comp_1 += r1.completion_tokens
            sum_lat_1 += r1.latency_s
            sum_ttft_1, cnt_ttft_1 = _accum_ttft(sum_ttft_1, cnt_ttft_1, r1)
            sum_ttft_tok_1, cnt_ttft_tok_1 = _accum_ttft_tokens(sum_ttft_tok_1, cnt_ttft_tok_1, r1)

            pred_strict = extract_strategyqa_strict(r1.text)
            fmtok1 = pred_strict in {"true", "false"} if pred_strict is not None else False
            pred_used1 = pred_strict if fmtok1 else extract_strategyqa_last(r1.text)
            ok1 = (pred_used1 == gold_norm) if pred_used1 in {"true", "false"} else False

            correct_1 += int(ok1)
            fmt_1 += int(fmtok1)

            h1, t1 = head_tail(r1.text)
            rec = {
                "task": "strategyqa",
                "gold": gold_norm,
                "ok_1": ok1,
                "fmt_1": fmtok1,
                "pred_1": pred_used1,
                "pred_sf_1": pred_strict if fmtok1 else None,
                "pred_fb_1": None if fmtok1 else pred_used1,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
                "total_tokens_1": r1.prompt_tokens + r1.completion_tokens,
                "ttft_s_1": r1.ttft_s,
                "ttft_tokens_1": r1.ttft_tokens,
                "latency_s_1": r1.latency_s,
                "toks_per_s_1": r1.toks_per_s,
                "prompt_1_raw": r1.prompt_raw,
                "prompt_1_formatted": r1.prompt_formatted,
                "prompt_chars_1_raw": len(r1.prompt_raw),
                "prompt_chars_1_formatted": len(r1.prompt_formatted),
                "raw_1_head": h1,
                "raw_1_tail": t1,
                "raw_1_full": r1.text,
                "raw_1_streamed": r1.text_streamed,
            }

            if reflection:
                prompt2 = build_reflection_prompt("strategyqa", prompt1, r1.text, think_mode=think_mode)
                r2 = gen.generate(prompt2)

                sum_prompt_2 += r2.prompt_tokens
                sum_comp_2 += r2.completion_tokens
                sum_lat_2 += r2.latency_s
                sum_ttft_2, cnt_ttft_2 = _accum_ttft(sum_ttft_2, cnt_ttft_2, r2)
                sum_ttft_tok_2, cnt_ttft_tok_2 = _accum_ttft_tokens(sum_ttft_tok_2, cnt_ttft_tok_2, r2)

                pred_strict2 = extract_strategyqa_strict(r2.text)
                fmtok2 = pred_strict2 in {"true", "false"} if pred_strict2 is not None else False
                pred_used2 = pred_strict2 if fmtok2 else extract_strategyqa_last(r2.text)
                ok2 = (pred_used2 == gold_norm) if pred_used2 in {"true", "false"} else False

                correct_2 += int(ok2)
                fmt_2 += int(fmtok2)

                h2, t2 = head_tail(r2.text)
                rec.update(
                    {
                        "ok_2": ok2,
                        "fmt_2": fmtok2,
                        "pred_2": pred_used2,
                        "pred_sf_2": pred_strict2 if fmtok2 else None,
                        "pred_fb_2": None if fmtok2 else pred_used2,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                        "total_tokens_2": r2.prompt_tokens + r2.completion_tokens,
                        "ttft_s_2": r2.ttft_s,
                        "ttft_tokens_2": r2.ttft_tokens,
                        "latency_s_2": r2.latency_s,
                        "toks_per_s_2": r2.toks_per_s,
                        "prompt_2_raw": r2.prompt_raw,
                        "prompt_2_formatted": r2.prompt_formatted,
                        "prompt_chars_2_raw": len(r2.prompt_raw),
                        "prompt_chars_2_formatted": len(r2.prompt_formatted),
                        "raw_2_head": h2,
                        "raw_2_tail": t2,
                        "raw_2_full": r2.text,
                        "raw_2_streamed": r2.text_streamed,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples) - skipped
    res: dict[str, Any] = {
        "task": "strategyqa",
        "total": len(examples),
        "valid": n,
        "skipped": skipped,
        "acc_1": correct_1 / n if n else 0.0,
        "fmt_1": fmt_1 / n if n else 0.0,
        "avg_prompt_tokens_1": (sum_prompt_1 / n) if n else 0.0,
        "avg_completion_tokens_1": (sum_comp_1 / n) if n else 0.0,
        "avg_total_tokens_1": ((sum_prompt_1 + sum_comp_1) / n) if n else 0.0,
        "avg_latency_s_1": (sum_lat_1 / n) if n else 0.0,
        "avg_ttft_s_1": (sum_ttft_1 / cnt_ttft_1) if cnt_ttft_1 else None,
        "avg_ttft_tokens_1": (sum_ttft_tok_1 / cnt_ttft_tok_1) if cnt_ttft_tok_1 else None,
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "fmt_2": fmt_2 / n if n else 0.0,
                "avg_prompt_tokens_2": (sum_prompt_2 / n) if n else 0.0,
                "avg_completion_tokens_2": (sum_comp_2 / n) if n else 0.0,
                "avg_total_tokens_2": ((sum_prompt_2 + sum_comp_2) / n) if n else 0.0,
                "avg_latency_s_2": (sum_lat_2 / n) if n else 0.0,
                "avg_ttft_s_2": (sum_ttft_2 / cnt_ttft_2) if cnt_ttft_2 else None,
                "avg_ttft_tokens_2": (sum_ttft_tok_2 / cnt_ttft_tok_2) if cnt_ttft_tok_2 else None,
            }
        )
    return res


# ============================================================
# SQuAD (v1) EM/F1
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


def load_or_make_squad_cache(data_dir: str, seed: int = 42, n: int = 2000) -> str:
    import json
    import os
    import random

    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"squad_v1_val{n}_seed{seed}.jsonl")
    if os.path.exists(cache_path):
        return cache_path

    last_err = None
    ds = None
    for name in ("squad", "rajpurkar/squad"):
        try:
            ds = load_dataset(name)
            break
        except Exception as e:
            last_err = e
            ds = None

    if ds is None:
        raise RuntimeError(
            "Failed to load SQuAD dataset. Try upgrading `datasets` (recommended), or ensure network access. Last error:\n" + str(last_err)
        )

    split = "validation" if "validation" in ds else (list(ds.keys())[0])
    dsv = ds[split]

    rng = random.Random(seed)
    idxs = rng.sample(range(len(dsv)), min(n, len(dsv)))

    with open(cache_path, "w", encoding="utf-8") as f:
        for i in idxs:
            ex = dsv[i]
            obj = {
                "id": ex.get("id"),
                "title": ex.get("title"),
                "context": ex.get("context", ""),
                "question": ex.get("question", ""),
                "answers": ex.get("answers", {}),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return cache_path


def load_squad_cache(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def get_squad_gold_answers(ex: dict[str, Any]) -> list[str]:
    ans = ex.get("answers", {}) or {}
    texts = ans.get("text", [])
    if isinstance(texts, str):
        texts = [texts]
    return [str(x) for x in texts if str(x).strip()]


def eval_squad(gen: HFGenerator, examples: list[dict[str, Any]], reflection: bool, out_path: str, think_mode: str) -> dict[str, Any]:
    em_1 = f1_1 = 0.0
    em_2 = f1_2 = 0.0

    sum_prompt_1 = sum_comp_1 = 0
    sum_prompt_2 = sum_comp_2 = 0
    sum_ttft_1 = sum_lat_1 = 0.0
    sum_ttft_2 = sum_lat_2 = 0.0
    cnt_ttft_1 = cnt_ttft_2 = 0

    sum_ttft_tok_1 = sum_ttft_tok_2 = 0
    cnt_ttft_tok_1 = cnt_ttft_tok_2 = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            ctx = ex.get("context", "")
            q = ex.get("question", "")
            golds = get_squad_gold_answers(ex)

            prompt1 = build_squad_prompt(ctx, q, think_mode=think_mode)
            r1 = gen.generate(prompt1)

            sum_prompt_1 += r1.prompt_tokens
            sum_comp_1 += r1.completion_tokens
            sum_lat_1 += r1.latency_s
            sum_ttft_1, cnt_ttft_1 = _accum_ttft(sum_ttft_1, cnt_ttft_1, r1)
            sum_ttft_tok_1, cnt_ttft_tok_1 = _accum_ttft_tokens(sum_ttft_tok_1, cnt_ttft_tok_1, r1)

            pred1 = extract_qa_final_answer(r1.text) or ""
            if golds:
                em_best = max(_exact_match(pred1, g) for g in golds)
                f1_best = max(_f1_score(pred1, g) for g in golds)
            else:
                em_best = 0.0
                f1_best = 0.0

            em_1 += em_best
            f1_1 += f1_best

            h1, t1 = head_tail(r1.text)
            rec = {
                "task": "squad",
                "id": ex.get("id", None),
                "gold": golds,
                "em_1": em_best,
                "f1_1": f1_best,
                "pred_1": pred1,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
                "total_tokens_1": r1.prompt_tokens + r1.completion_tokens,
                "ttft_s_1": r1.ttft_s,
                "ttft_tokens_1": r1.ttft_tokens,
                "latency_s_1": r1.latency_s,
                "toks_per_s_1": r1.toks_per_s,
                "prompt_1_raw": r1.prompt_raw,
                "prompt_1_formatted": r1.prompt_formatted,
                "raw_1_head": h1,
                "raw_1_tail": t1,
                "raw_1_full": r1.text,
                "raw_1_streamed": r1.text_streamed,
            }

            if reflection:
                prompt2 = build_reflection_prompt("squad", prompt1, r1.text, think_mode=think_mode)
                r2 = gen.generate(prompt2)

                sum_prompt_2 += r2.prompt_tokens
                sum_comp_2 += r2.completion_tokens
                sum_lat_2 += r2.latency_s
                sum_ttft_2, cnt_ttft_2 = _accum_ttft(sum_ttft_2, cnt_ttft_2, r2)
                sum_ttft_tok_2, cnt_ttft_tok_2 = _accum_ttft_tokens(sum_ttft_tok_2, cnt_ttft_tok_2, r2)

                pred2 = extract_qa_final_answer(r2.text) or ""
                if golds:
                    em_best2 = max(_exact_match(pred2, g) for g in golds)
                    f1_best2 = max(_f1_score(pred2, g) for g in golds)
                else:
                    em_best2 = 0.0
                    f1_best2 = 0.0

                em_2 += em_best2
                f1_2 += f1_best2

                h2, t2 = head_tail(r2.text)
                rec.update(
                    {
                        "em_2": em_best2,
                        "f1_2": f1_best2,
                        "pred_2": pred2,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                        "total_tokens_2": r2.prompt_tokens + r2.completion_tokens,
                        "ttft_s_2": r2.ttft_s,
                        "ttft_tokens_2": r2.ttft_tokens,
                        "latency_s_2": r2.latency_s,
                        "toks_per_s_2": r2.toks_per_s,
                        "prompt_2_raw": r2.prompt_raw,
                        "prompt_2_formatted": r2.prompt_formatted,
                        "raw_2_head": h2,
                        "raw_2_tail": t2,
                        "raw_2_full": r2.text,
                        "raw_2_streamed": r2.text_streamed,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples)
    res: dict[str, Any] = {
        "task": "squad",
        "total": n,
        "valid": n,
        "em_1": (em_1 / n) if n else 0.0,
        "f1_1": (f1_1 / n) if n else 0.0,
        "avg_prompt_tokens_1": (sum_prompt_1 / n) if n else 0.0,
        "avg_completion_tokens_1": (sum_comp_1 / n) if n else 0.0,
        "avg_total_tokens_1": ((sum_prompt_1 + sum_comp_1) / n) if n else 0.0,
        "avg_latency_s_1": (sum_lat_1 / n) if n else 0.0,
        "avg_ttft_s_1": (sum_ttft_1 / cnt_ttft_1) if cnt_ttft_1 else None,
        "avg_ttft_tokens_1": (sum_ttft_tok_1 / cnt_ttft_tok_1) if cnt_ttft_tok_1 else None,
    }
    if reflection:
        res.update(
            {
                "em_2": (em_2 / n) if n else 0.0,
                "f1_2": (f1_2 / n) if n else 0.0,
                "avg_prompt_tokens_2": (sum_prompt_2 / n) if n else 0.0,
                "avg_completion_tokens_2": (sum_comp_2 / n) if n else 0.0,
                "avg_total_tokens_2": ((sum_prompt_2 + sum_comp_2) / n) if n else 0.0,
                "avg_latency_s_2": (sum_lat_2 / n) if n else 0.0,
                "avg_ttft_s_2": (sum_ttft_2 / cnt_ttft_2) if cnt_ttft_2 else None,
                "avg_ttft_tokens_2": (sum_ttft_tok_2 / cnt_ttft_tok_2) if cnt_ttft_tok_2 else None,
            }
        )
    return res


# ============================================================
# Local ProofWriter JSONL loader
# ============================================================


def load_local_proofwriter_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
                out.append(obj["data"])
            else:
                out.append(obj)
    return out


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "gsm8k",
            "proofwriter",
            "logicnli",
            "mathqa",
            "logiqa",
            "logicqa",
            "arc_c",
            "math500",
            "strategyqa",
            "strageqa",
            "squad",
            "pubmedqa",
        ],
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--n", type=int, default=0, help="Num samples (0=all) (ignored for squad default cache)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--data_dir", type=str, default="${SEC5_ROOT}/infer/dataset")

    parser.add_argument(
        "--in_jsonl",
        type=str,
        default="${WORKSPACE_ROOT}/open-router/process_data_new/data/proofwriter_test_1k.jsonl",
        help="(proofwriter only) local sampled jsonl path",
    )

    parser.add_argument("--think_mode", type=str, default="auto", choices=["auto", "think", "no_think"])
    parser.add_argument("--inject_think_tokens", action="store_true")
    parser.add_argument("--no_ttft", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=None)

    parser.add_argument("--use_qwen_best_practices", action="store_true")
    parser.add_argument("--reflection", action="store_true")

    parser.add_argument(
        "--squad_cache",
        type=str,
        default=None,
        help="optional path to cached squad sample jsonl (default: data/squad_v1_val{n}_seed{seed}.jsonl)",
    )
    parser.add_argument("--squad_n", type=int, default=1000, help="squad cache sample size (default 2000)")

    args = parser.parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.data_dir)
    seed_everything(args.seed)

    task = args.task.lower()
    if task == "logicqa":
        task = "logiqa"
    if task == "strageqa":
        task = "strategyqa"

    if args.use_qwen_best_practices:
        resolved_mode = args.think_mode
        if _model_forces_non_think(args.model):
            resolved_mode = "no_think"
        rec = qwen_recommended_sampling(resolved_mode)
        if args.temperature is None:
            args.temperature = rec["temperature"]
        if args.top_p is None:
            args.top_p = rec["top_p"]
        if args.top_k is None:
            args.top_k = rec["top_k"]
        if args.min_p is None:
            args.min_p = rec["min_p"]
        if args.presence_penalty is None:
            args.presence_penalty = rec["presence_penalty"]

    if args.temperature is None:
        args.temperature = 0.0
    if args.top_p is None:
        args.top_p = 1.0
    if args.top_k is None:
        args.top_k = 0
    if args.min_p is None:
        args.min_p = 0.0
    if args.presence_penalty is None:
        args.presence_penalty = 0.0

    gen = HFGenerator(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        think_mode=args.think_mode,
        measure_ttft=(not args.no_ttft),
        inject_think_tokens=args.inject_think_tokens,
    )

    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        split = args.split or pick_split(ds)
        examples = sample_examples(ds[split], args.n, args.seed)
        out_path = os.path.join(args.out_dir, "gsm8k.jsonl")
        res = eval_gsm8k(gen, examples, args.reflection, out_path, think_mode=args.think_mode)

    elif task == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500")
        split = args.split or pick_split(ds)
        examples = sample_examples(ds[split], args.n, args.seed)
        out_path = os.path.join(args.out_dir, "math500.jsonl")
        res = eval_math500(gen, examples, args.reflection, out_path, think_mode=args.think_mode)

    elif task == "proofwriter":
        if args.in_jsonl:
            examples = load_local_proofwriter_jsonl(args.in_jsonl)
        else:
            ds = load_dataset("tasksource/proofwriter")
            split = args.split or pick_split(ds)
            examples = sample_examples(ds[split], args.n, args.seed)

        if args.in_jsonl and args.n > 0 and args.n < len(examples):
            rng = random.Random(args.seed)
            idxs = rng.sample(range(len(examples)), args.n)
            examples = [examples[i] for i in idxs]

        out_path = os.path.join(args.out_dir, "proofwriter.jsonl")
        res = eval_nli(
            gen,
            examples,
            "proofwriter",
            lambda ex: (ex["theory"], ex["question"], ex["answer"]),
            args.reflection,
            out_path,
            think_mode=args.think_mode,
        )

    elif task == "logicnli":
        ds = load_dataset("tasksource/LogicNLI")
        split = args.split or pick_split(ds)
        examples = sample_examples(ds[split], args.n, args.seed)
        out_path = os.path.join(args.out_dir, "logicnli.jsonl")
        res = eval_nli(
            gen,
            examples,
            "logicnli",
            lambda ex: (ex["premise"], ex["hypothesis"], ex["label"]),
            args.reflection,
            out_path,
            think_mode=args.think_mode,
        )

    elif task == "strategyqa":
        ds = load_dataset("ChilleD/StrategyQA")
        split = args.split or pick_split(ds)
        examples = sample_examples(ds[split], args.n, args.seed)
        out_path = os.path.join(args.out_dir, "strategyqa.jsonl")
        res = eval_strategyqa(gen, examples, args.reflection, out_path, think_mode=args.think_mode)

    elif task == "mathqa":
        ds = load_dataset("allenai/math_qa")
        split = args.split or pick_split(ds)
        examples = sample_examples(ds[split], args.n, args.seed)
        out_path = os.path.join(args.out_dir, "mathqa.jsonl")

        def get_data(ex):
            q = ex.get("Problem", "")
            opts = parse_mathqa_options(ex.get("options", ""))
            gold = ex.get("correct", "")
            return q, opts, gold

        res = eval_mcq(gen, examples, "mathqa", get_data, args.reflection, out_path, max_opts=5, think_mode=args.think_mode)

    elif task == "logiqa":
        ds = load_dataset("lucasmccabe/logiqa")
        split = args.split or pick_split(ds)
        examples = sample_examples(ds[split], args.n, args.seed)
        out_path = os.path.join(args.out_dir, "logiqa.jsonl")

        def get_data(ex):
            ctx = ex.get("context", "")
            question = ex.get("query", "")
            opts = ex.get("options", [])
            if not isinstance(opts, list):
                opts = [x.strip() for x in str(opts).split("\n") if x.strip()]
            opts = [normalize_space(str(x)) for x in opts]
            gold = ex.get("correct_option", 0)
            return ctx, question, opts, gold

        res = eval_mcq(gen, examples, "logiqa", get_data, args.reflection, out_path, max_opts=4, think_mode=args.think_mode)

    elif task == "arc_c":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        split = args.split or pick_split(ds)
        examples = sample_examples(ds[split], args.n, args.seed)
        out_path = os.path.join(args.out_dir, "arc_c.jsonl")

        def get_data(ex):
            q = ex["question"]
            opts = arc_choices_to_options(ex["choices"])
            gold = ex["answerKey"]
            return q, opts, gold

        res = eval_mcq(gen, examples, "arc_c", get_data, args.reflection, out_path, max_opts=5, think_mode=args.think_mode)

    elif task == "squad":
        cache_path = args.squad_cache
        if cache_path is None:
            cache_path = load_or_make_squad_cache(args.data_dir, seed=args.seed, n=args.squad_n)
        else:
            if not os.path.exists(cache_path):
                ensure_dir(os.path.dirname(cache_path) or ".")
                tmp = load_or_make_squad_cache(args.data_dir, seed=args.seed, n=args.squad_n)
                with open(tmp, encoding="utf-8") as src, open(cache_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())

        examples = load_squad_cache(cache_path)
        examples = sample_examples(examples, args.n, args.seed)
        out_path = os.path.join(args.out_dir, "squad.jsonl")
        res = eval_squad(gen, examples, args.reflection, out_path, think_mode=args.think_mode)
        res["cache_path"] = cache_path

    elif task == "pubmedqa":
        # PubMedQA pqa_labeled
        # Fields:
        #   - question: str
        #   - context: Sequence[dict] where dict has key "contexts" etc.
        #   - final_decision: yes/no/maybe
        #   - long_answer: str
        #   - pubid: int
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        split = args.split or pick_split(ds)  # typically only "train"
        examples = sample_examples(ds["train"], args.n, args.seed)

        out_path = os.path.join(args.out_dir, "pubmedqa_pqa_labeled.jsonl")
        res = eval_nli(
            gen,
            examples,
            "pubmedqa",
            lambda ex: (pubmedqa_build_abstract(ex), ex.get("question", ""), pubmedqa_get_gold(ex)),
            args.reflection,
            out_path,
            think_mode=args.think_mode,
        )

    else:
        raise ValueError(f"Unsupported task: {task}")

    summary_path = os.path.join(args.out_dir, f"{task}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"[Saved summary] {summary_path}")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n[NOTE] avg_latency_s_1 = average wall-clock seconds per example for the FIRST pass (prompt->generate done), including TTFT.\n")


if __name__ == "__main__":
    main()
