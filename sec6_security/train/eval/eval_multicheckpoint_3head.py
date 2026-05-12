#!/usr/bin/env python3

import argparse
import ast
import json
import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import transformers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3 import Qwen3ForMedusa  # noqa


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        transformers.set_seed(seed)
    except Exception:
        pass


def pick_split(ds_dict, preferred=("test", "validation", "dev", "train")) -> str:
    for s in preferred:
        if s in ds_dict:
            return s
    return list(ds_dict.keys())[0]


def sample_examples_hfds(ds_split, n: int, seed: int) -> list[dict[str, Any]]:
    L = len(ds_split)
    if n <= 0 or n >= L:
        return [ds_split[i] for i in range(L)]
    rng = random.Random(seed)
    idxs = rng.sample(range(L), n)
    return [ds_split[i] for i in idxs]


def sample_examples_list(examples: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(examples):
        return examples
    rng = random.Random(seed)
    idxs = rng.sample(range(len(examples)), n)
    return [examples[i] for i in idxs]


def load_jsonl_generic(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"jsonl line {line_no} is not a dict")
            out.append(obj)
    return out


def load_squad_jsonl(path: str) -> list[dict[str, Any]]:
    exs = load_jsonl_generic(path)
    for i, ex in enumerate(exs):
        if "context" not in ex or "question" not in ex or "answers" not in ex:
            raise ValueError(f"Invalid SQuAD jsonl line {i + 1}: need context/question/answers")
    return exs


def load_proofwriter_jsonl(path: str) -> list[dict[str, Any]]:
    raw = load_jsonl_generic(path)
    out: list[dict[str, Any]] = []
    for obj in raw:
        if isinstance(obj.get("data"), dict):
            out.append(obj["data"])
        else:
            out.append(obj)
    for i, ex in enumerate(out):
        if "theory" not in ex or "question" not in ex or "answer" not in ex:
            raise ValueError(f"Invalid ProofWriter jsonl item {i + 1}: need theory/question/answer")
    return out


def load_dual_paraphrase_json(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("dual_paraphrase file must be a JSON list")
    out: list[dict[str, Any]] = []
    for i, ex in enumerate(obj):
        if not isinstance(ex, dict):
            raise ValueError(f"dual_paraphrase item {i + 1} is not dict")
        if "sentence_a" not in ex or "sentence_b" not in ex or "rewrite_a" not in ex or "rewrite_b" not in ex:
            raise ValueError(f"dual_paraphrase item {i + 1} missing required keys")
        out.append(ex)
    return out


def wrap_input_with_context(question_line: str, context_block: str) -> str:
    q = (question_line or "").strip()
    ctx = (context_block or "").strip()
    return f"{q} based on the provided context below.\n\nContext:\n{ctx}"


def build_squad_input(ex: dict[str, Any]) -> str:
    return wrap_input_with_context(ex.get("question", ""), ex.get("context", ""))


def get_squad_gold_answers(ex: dict[str, Any]) -> list[str]:
    ans = ex.get("answers", {}) or {}
    texts = ans.get("text", [])
    if isinstance(texts, str):
        texts = [texts]
    return [str(x) for x in texts if str(x).strip()]


def _pw_premises_to_text(theory: Any) -> str:
    if isinstance(theory, list):
        return "\n".join([str(x).strip() for x in theory if str(x).strip()])
    return "\n".join([ln.strip() for ln in str(theory or "").splitlines() if ln.strip()])


def build_proofwriter_input(ex: dict[str, Any]) -> str:
    query = ex.get("question", ex.get("query", ex.get("conclusion", ""))) or ""
    theory = ex.get("theory", ex.get("premises", ex.get("context", "")))
    qline = f'Determine whether the conclusion "{str(query).strip()}" is true, false, or unknown.'
    return wrap_input_with_context(qline, _pw_premises_to_text(theory))


def get_proofwriter_gold(ex: dict[str, Any]) -> str | None:
    g = ex.get("answer", ex.get("label", ex.get("gold", None)))
    if g is None:
        return None
    s = str(g).strip().lower().rstrip(".")
    return s if s in {"true", "false", "unknown"} else None


def pubmedqa_build_abstract(ex: dict[str, Any]) -> str:
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

    add(ctx)
    if not parts:
        add(ex.get("long_answer", None))
    return "\n".join([p for p in parts if p])


def build_pubmedqa_input(ex: dict[str, Any]) -> str:
    q = (ex.get("question", "") or "").strip()
    qline = f"{q} Answer with one of: Yes, No, or Maybe."
    return wrap_input_with_context(qline, pubmedqa_build_abstract(ex))


def get_pubmedqa_gold(ex: dict[str, Any]) -> str | None:
    g = ex.get("final_decision", None)
    if g is None:
        return None
    s = str(g).strip().lower().rstrip(".")
    return s if s in {"yes", "no", "maybe"} else None


def build_logicnli_input(ex: dict[str, Any]) -> str:
    premise = ex.get("premise", "") or ""
    hyp = ex.get("hypothesis", "") or ""
    qline = f"Determine whether the hypothesis: {str(hyp).strip()} is entailment, contradiction, neutral, or self_contradiction"
    return wrap_input_with_context(qline, f"Premise: {str(premise).strip()}")


def get_logicnli_gold(ex: dict[str, Any]) -> str | None:
    g = ex.get("label", ex.get("gold", ex.get("answer", None)))
    if g is None:
        return None
    s = str(g).strip().lower()
    valid = {"entailment", "contradiction", "neutral", "self_contradiction"}
    return s if s in valid else None


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


def extract_gsm8k_gold(answer_text: str) -> str | None:
    if not answer_text:
        return None
    m = _GSM_ANSWER_RE.search(answer_text)
    return normalize_number_str(m.group(1)) if m else None


def extract_pred_number(text: str) -> str | None:
    nums = _GSM_LAST_NUM_RE.findall(text or "")
    return normalize_number_str(nums[-1]) if nums else None


_ARTICLES = {"a", "an", "the"}


def _normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _f1_score(pred: str, gold: str) -> float:
    pred_toks = _normalize_answer(pred).split()
    gold_toks = _normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common: dict[str, int] = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0


def extract_qa_final_answer(text: str) -> str:
    t = (text or "").strip()
    m = list(re.finditer(r"(?is)<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", t))
    if m:
        return m[-1].group(1).strip()
    m2 = list(re.finditer(r"(final\s*answer|answer)\s*[:：]\s*(.*)", t, re.I))
    if m2:
        return m2[-1].group(2).splitlines()[0].strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def parse_rewrite_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [s]


def normalize_text_loose(s: str) -> str:
    return " ".join((s or "").lower().split())


def token_f1_loose(pred: str, gold: str) -> float:
    p = normalize_text_loose(pred).split()
    g = normalize_text_loose(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    cnt: dict[str, int] = {}
    for t in p:
        cnt[t] = cnt.get(t, 0) + 1
    same = 0
    for t in g:
        if cnt.get(t, 0) > 0:
            same += 1
            cnt[t] -= 1
    if same == 0:
        return 0.0
    pr = same / len(p)
    rc = same / len(g)
    return 2 * pr * rc / (pr + rc)


def build_dual_paraphrase_system_prompt(sentence_a: str, sentence_b: str) -> str:
    return (
        "You are a multi channel rewriting and copying machine. "
        "Roles: channel 0 rewrites Sentence A only. "
        "channel 2 rewrites Sentence B only and must copy the FULL rewrite from channel 0. "
        "channel 1 must copy from both channel 0 and channel 2 outputs. "
        "Do not add extra text. "
        f"Sentence A: {sentence_a} "
        f"Sentence B: {sentence_b}"
    )


def build_dual_paraphrase_prompt_and_gold(ex: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    sentence_a = " ".join(str(ex.get("sentence_a", "")).split())
    sentence_b = " ".join(str(ex.get("sentence_b", "")).split())
    ra = parse_rewrite_list(ex.get("rewrite_a", ""))
    rb = parse_rewrite_list(ex.get("rewrite_b", ""))
    a_gold = " ".join((ra[0] if ra else "").split())
    b_gold = " ".join((rb[0] if rb else "").split())
    system_text = build_dual_paraphrase_system_prompt(sentence_a, sentence_b)
    return system_text, "", {"a_gold": a_gold, "b_gold": b_gold}


def extract_after_marker(text: str, marker: str) -> str:
    if not text:
        return ""
    idx = text.lower().find(marker.lower())
    if idx < 0:
        return ""
    return text[idx + len(marker) :].strip()


def extract_between_markers(text: str, left: str, right: str) -> str:
    if not text:
        return ""
    tl = text.lower()
    l = tl.find(left.lower())
    if l < 0:
        return ""
    r = tl.find(right.lower(), l + len(left))
    if r < 0:
        return text[l + len(left) :].strip()
    return text[l + len(left) : r].strip()


@dataclass
class TaskSpec:
    head: int
    task: str
    split: str | None = None
    n: int = 0
    seed: int = 42
    in_jsonl: str | None = None
    final_prefix: str = ""
    max_new_tokens: int = 256
    max_steps: int = 2048


def load_examples_for_task(spec: TaskSpec) -> list[dict[str, Any]]:
    task = spec.task.lower()
    if task == "squad":
        if spec.in_jsonl:
            return sample_examples_list(load_squad_jsonl(spec.in_jsonl), spec.n, spec.seed)
        from datasets import load_dataset

        ds = load_dataset("rajpurkar/squad")
        use_split = spec.split or ("validation" if "validation" in ds else pick_split(ds))
        return sample_examples_hfds(ds[use_split], spec.n, spec.seed)
    if task == "proofwriter":
        if spec.in_jsonl:
            return sample_examples_list(load_proofwriter_jsonl(spec.in_jsonl), spec.n, spec.seed)
        from datasets import load_dataset

        ds = load_dataset("tasksource/proofwriter")
        use_split = spec.split or pick_split(ds)
        return sample_examples_hfds(ds[use_split], spec.n, spec.seed)
    if task == "logicnli":
        from datasets import load_dataset

        ds = load_dataset("tasksource/LogicNLI")
        use_split = spec.split or pick_split(ds)
        return sample_examples_hfds(ds[use_split], spec.n, spec.seed)
    if task == "pubmedqa":
        from datasets import load_dataset

        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        use_split = spec.split or ("train" if "train" in ds else pick_split(ds))
        return sample_examples_hfds(ds[use_split], spec.n, spec.seed)
    if task == "gsm8k":
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main")
        use_split = spec.split or ("test" if "test" in ds else pick_split(ds))
        return sample_examples_hfds(ds[use_split], spec.n, spec.seed)
    if task == "dual_paraphrase":
        if not spec.in_jsonl:
            raise ValueError("dual_paraphrase requires in_jsonl path to test.json")
        return sample_examples_list(load_dual_paraphrase_json(spec.in_jsonl), spec.n, spec.seed)
    raise ValueError(f"Unsupported task: {spec.task}")


def build_prompt_and_gold(task: str, ex: dict[str, Any]) -> tuple[str, str, Any]:
    t = task.lower()
    if t == "squad":
        return "You are a helpful assistant.", build_squad_input(ex), get_squad_gold_answers(ex)
    if t == "proofwriter":
        return "You are a helpful assistant.", build_proofwriter_input(ex), get_proofwriter_gold(ex)
    if t == "logicnli":
        return "You are a helpful assistant.", build_logicnli_input(ex), get_logicnli_gold(ex)
    if t == "pubmedqa":
        return "You are a helpful assistant.", build_pubmedqa_input(ex), get_pubmedqa_gold(ex)
    if t == "gsm8k":
        q = (ex.get("question", "") or "").strip()
        return "You are a helpful assistant.", q, extract_gsm8k_gold(ex.get("answer", ""))
    if t == "dual_paraphrase":
        return build_dual_paraphrase_prompt_and_gold(ex)
    raise ValueError(f"Unsupported task: {task}")


class MultiHeadMedusaInference:
    def __init__(self, model_path: str, assistant_heads: int, device: str = "auto", torch_dtype: str = "auto"):
        self.assistant_heads = int(assistant_heads)
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        if self.device == "auto":
            self.device = "cpu"
        if torch_dtype == "auto":
            self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        elif torch_dtype == "float16":
            self.dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

        self._gen_fn = getattr(self.model, "medusa_generate_interleaved_multihead_stream_user_same_y0", None)
        if self._gen_fn is None:
            raise RuntimeError("Model is missing medusa_generate_interleaved_multihead_stream_user_same_y0")

    @torch.no_grad()
    def generate_all_heads(
        self,
        system_text: str,
        question_text: str,
        final_prefix_by_head: list[str],
        max_new_tokens: int,
        max_steps: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        stop_on_im_end: bool,
    ) -> str:
        self.model.system_message = system_text or "You are a helpful assistant."
        out = self._gen_fn(
            question_text=question_text,
            assistant_heads=self.assistant_heads,
            assistant_prefix_texts=final_prefix_by_head,
            assistant_prefill_texts=[""] * self.assistant_heads,
            max_new_tokens=max_new_tokens,
            max_steps=max_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            stop_on_im_end=stop_on_im_end,
            allow_same_step_visible=False,
        )
        text_by_head: dict[int, str] = {}
        for h in range(self.assistant_heads):
            toks = out.get(h, None)
            if toks is None or toks.numel() == 0:
                text_by_head[h] = ""
            else:
                text_by_head[h] = self.tokenizer.decode(
                    toks.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
        return text_by_head

    @torch.no_grad()
    def generate_head(
        self,
        system_text: str,
        question_text: str,
        head_idx: int,
        final_prefix: str,
        max_new_tokens: int,
        max_steps: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        stop_on_im_end: bool,
    ) -> str:
        prefixes = [""] * self.assistant_heads
        if 0 <= head_idx < self.assistant_heads and final_prefix:
            prefixes[head_idx] = final_prefix
        text_by_head = self.generate_all_heads(
            system_text=system_text,
            question_text=question_text,
            final_prefix_by_head=prefixes,
            max_new_tokens=max_new_tokens,
            max_steps=max_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            stop_on_im_end=stop_on_im_end,
        )
        return text_by_head.get(head_idx, "")


def evaluate_task_on_head(
    infer: MultiHeadMedusaInference,
    checkpoint_name: str,
    spec: TaskSpec,
    examples: list[dict[str, Any]],
    out_dir: str,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    stop_on_im_end: bool,
) -> dict[str, Any]:
    task = spec.task.lower()
    ensure_dir(out_dir)
    out_jsonl = os.path.join(out_dir, f"{checkpoint_name}_head{spec.head}_{task}.jsonl")

    n = 0
    acc = 0.0
    em_sum = 0.0
    f1_sum = 0.0
    h1_copy_a_gold_sum = 0.0
    h1_copy_b_gold_sum = 0.0
    h1_copy_both_gold_sum = 0.0
    h1_copy_a_ref_sum = 0.0
    h1_copy_b_ref_sum = 0.0
    h1_copy_both_ref_sum = 0.0
    h2_b_f1_gold_sum = 0.0
    h2_copy_a_gold_sum = 0.0
    h2_mixed_gold_sum = 0.0
    h2_copy_a_ref_sum = 0.0
    h2_mixed_ref_sum = 0.0
    start = time.time()

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(examples):
            system_text, prompt, gold = build_prompt_and_gold(task, ex)
            if task == "dual_paraphrase":
                prefix_by_head = [""] * infer.assistant_heads
                if 0 <= spec.head < infer.assistant_heads and spec.final_prefix:
                    prefix_by_head[spec.head] = spec.final_prefix
                pred_all = infer.generate_all_heads(
                    system_text=system_text,
                    question_text=prompt,
                    final_prefix_by_head=prefix_by_head,
                    max_new_tokens=spec.max_new_tokens,
                    max_steps=spec.max_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    stop_on_im_end=stop_on_im_end,
                )
                pred_text = pred_all.get(spec.head, "")
                pred_h0 = pred_all.get(0, "")
                pred_h2 = pred_all.get(2, "")
            else:
                pred_text = infer.generate_head(
                    system_text=system_text,
                    question_text=prompt,
                    head_idx=spec.head,
                    final_prefix=spec.final_prefix,
                    max_new_tokens=spec.max_new_tokens,
                    max_steps=spec.max_steps,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    stop_on_im_end=stop_on_im_end,
                )
                pred_h0 = ""
                pred_h2 = ""

            rec: dict[str, Any] = {
                "checkpoint": checkpoint_name,
                "task": task,
                "head": spec.head,
                "index": i,
                "system": system_text,
                "prompt": prompt,
                "gold": gold,
                "pred_text": pred_text,
            }

            if task == "squad":
                pred = extract_qa_final_answer(pred_text)
                golds = gold if isinstance(gold, list) else []
                em_best = max((_exact_match(pred, g) for g in golds), default=0.0)
                f1_best = max((_f1_score(pred, g) for g in golds), default=0.0)
                em_sum += em_best
                f1_sum += f1_best
                rec["pred"] = pred
                rec["em"] = em_best
                rec["f1"] = f1_best
            elif task == "gsm8k":
                pred = extract_pred_number(pred_text)
                ok = bool(gold is not None and pred is not None and gold == pred)
                acc += 1.0 if ok else 0.0
                rec["pred"] = pred
                rec["correct"] = ok
            elif task == "proofwriter":
                tail = pred_text.lower()[-1200:]
                m = re.findall(r"\b(true|false|unknown)\b", tail)
                pred = m[-1] if m else None
                ok = bool(gold is not None and pred == gold)
                acc += 1.0 if ok else 0.0
                rec["pred"] = pred
                rec["correct"] = ok
            elif task == "pubmedqa":
                tail = pred_text.lower()[-1200:]
                m = re.findall(r"\b(yes|no|maybe)\b", tail)
                pred = m[-1] if m else None
                ok = bool(gold is not None and pred == gold)
                acc += 1.0 if ok else 0.0
                rec["pred"] = pred
                rec["correct"] = ok
            elif task == "logicnli":
                tail = pred_text.lower()[-1500:]
                labels = ["self_contradiction", "contradiction", "entailment", "neutral"]
                pred = None
                for lab in labels:
                    if re.search(rf"\b{re.escape(lab)}\b", tail):
                        pred = lab
                ok = bool(gold is not None and pred == gold)
                acc += 1.0 if ok else 0.0
                rec["pred"] = pred
                rec["correct"] = ok
            elif task == "dual_paraphrase":
                a_gold = gold.get("a_gold", "")
                b_gold = gold.get("b_gold", "")
                pred_norm = normalize_text_loose(pred_text)
                a_norm = normalize_text_loose(a_gold)
                b_norm = normalize_text_loose(b_gold)
                h0_ref_norm = normalize_text_loose(pred_h0)
                h2_ref_norm = normalize_text_loose(pred_h2)

                if spec.head == 0:
                    a_part = extract_after_marker(pred_text, "Sentence A rewrite:")
                    if not a_part:
                        a_part = pred_text
                    a_f1 = token_f1_loose(a_part, a_gold)
                    a_exact = 1.0 if normalize_text_loose(a_part) == a_norm else 0.0
                    acc += a_f1
                    rec["a_f1"] = a_f1
                    rec["a_exact"] = a_exact
                elif spec.head == 2:
                    b_part = extract_between_markers(
                        pred_text,
                        "Sentence B rewrite:",
                        "Copied full rewrite from channel 0:",
                    )
                    if not b_part:
                        b_part = extract_after_marker(pred_text, "Sentence B rewrite:")
                    if not b_part:
                        b_part = pred_text
                    b_f1 = token_f1_loose(b_part, b_gold)
                    copy_a_hit_gold = 1.0 if a_norm and a_norm in pred_norm else 0.0
                    mixed_gold = 0.5 * b_f1 + 0.5 * copy_a_hit_gold
                    copy_a_hit_ref = 1.0 if h0_ref_norm and h0_ref_norm in pred_norm else 0.0
                    mixed_ref = 0.5 * b_f1 + 0.5 * copy_a_hit_ref
                    acc += mixed_gold
                    h2_b_f1_gold_sum += b_f1
                    h2_copy_a_gold_sum += copy_a_hit_gold
                    h2_mixed_gold_sum += mixed_gold
                    h2_copy_a_ref_sum += copy_a_hit_ref
                    h2_mixed_ref_sum += mixed_ref
                    rec["b_f1"] = b_f1
                    rec["copy_a_hit_gold"] = copy_a_hit_gold
                    rec["mixed_score_gold"] = mixed_gold
                    rec["copy_a_hit_ref"] = copy_a_hit_ref
                    rec["mixed_score_ref"] = mixed_ref
                    rec["ref_head0_text"] = pred_h0
                elif spec.head == 1:
                    copy_a_hit_gold = 1.0 if a_norm and a_norm in pred_norm else 0.0
                    copy_b_hit_gold = 1.0 if b_norm and b_norm in pred_norm else 0.0
                    copy_both_hit_gold = 1.0 if (copy_a_hit_gold > 0 and copy_b_hit_gold > 0) else 0.0
                    copy_a_hit_ref = 1.0 if h0_ref_norm and h0_ref_norm in pred_norm else 0.0
                    copy_b_hit_ref = 1.0 if h2_ref_norm and h2_ref_norm in pred_norm else 0.0
                    copy_both_hit_ref = 1.0 if (copy_a_hit_ref > 0 and copy_b_hit_ref > 0) else 0.0
                    acc += copy_both_hit_gold
                    h1_copy_a_gold_sum += copy_a_hit_gold
                    h1_copy_b_gold_sum += copy_b_hit_gold
                    h1_copy_both_gold_sum += copy_both_hit_gold
                    h1_copy_a_ref_sum += copy_a_hit_ref
                    h1_copy_b_ref_sum += copy_b_hit_ref
                    h1_copy_both_ref_sum += copy_both_hit_ref
                    rec["copy_a_hit_gold"] = copy_a_hit_gold
                    rec["copy_b_hit_gold"] = copy_b_hit_gold
                    rec["copy_both_hit_gold"] = copy_both_hit_gold
                    rec["copy_a_hit_ref"] = copy_a_hit_ref
                    rec["copy_b_hit_ref"] = copy_b_hit_ref
                    rec["copy_both_hit_ref"] = copy_both_hit_ref
                    rec["ref_head0_text"] = pred_h0
                    rec["ref_head2_text"] = pred_h2
                else:
                    raise ValueError("dual_paraphrase only supports head 0/1/2")
            else:
                raise ValueError(f"Unsupported task: {task}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    elapsed = time.time() - start
    summary: dict[str, Any] = {
        "checkpoint": checkpoint_name,
        "task": task,
        "head": spec.head,
        "num_samples": n,
        "elapsed_sec": elapsed,
        "throughput_samples_per_sec": n / max(elapsed, 1e-9),
        "output_jsonl": out_jsonl,
    }
    if task == "squad":
        summary["em"] = em_sum / max(n, 1)
        summary["f1"] = f1_sum / max(n, 1)
        summary["primary_metric"] = "f1"
        summary["primary_value"] = summary["f1"]
    elif task == "dual_paraphrase":
        if spec.head == 0:
            summary["score"] = acc / max(n, 1)
            summary["primary_metric"] = "a_f1"
            summary["primary_value"] = summary["score"]
        elif spec.head == 2:
            summary["score"] = h2_mixed_gold_sum / max(n, 1)
            summary["primary_metric"] = "mixed_score_gold"
            summary["primary_value"] = summary["score"]
            summary["b_f1_gold"] = h2_b_f1_gold_sum / max(n, 1)
            summary["copy_a_hit_gold"] = h2_copy_a_gold_sum / max(n, 1)
            summary["mixed_score_gold"] = h2_mixed_gold_sum / max(n, 1)
            summary["copy_a_hit_ref"] = h2_copy_a_ref_sum / max(n, 1)
            summary["mixed_score_ref"] = h2_mixed_ref_sum / max(n, 1)
        else:
            summary["score"] = h1_copy_both_gold_sum / max(n, 1)
            summary["primary_metric"] = "copy_both_hit_gold"
            summary["primary_value"] = summary["score"]
            summary["copy_a_hit_gold"] = h1_copy_a_gold_sum / max(n, 1)
            summary["copy_b_hit_gold"] = h1_copy_b_gold_sum / max(n, 1)
            summary["copy_both_hit_gold"] = h1_copy_both_gold_sum / max(n, 1)
            summary["copy_a_hit_ref"] = h1_copy_a_ref_sum / max(n, 1)
            summary["copy_b_hit_ref"] = h1_copy_b_ref_sum / max(n, 1)
            summary["copy_both_hit_ref"] = h1_copy_both_ref_sum / max(n, 1)
    else:
        summary["accuracy"] = acc / max(n, 1)
        summary["primary_metric"] = "accuracy"
        summary["primary_value"] = summary["accuracy"]
    return summary


def evaluate_dual_paraphrase_joint(
    infer: MultiHeadMedusaInference,
    checkpoint_name: str,
    specs: list[TaskSpec],
    examples: list[dict[str, Any]],
    out_dir: str,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    stop_on_im_end: bool,
) -> list[dict[str, Any]]:
    """
    One forward/generation per sample -> get head0/1/2 together, then compute all 3 head metrics.
    """
    ensure_dir(out_dir)
    spec_by_head = {s.head: s for s in specs}
    required = {0, 1, 2}
    if set(spec_by_head.keys()) != required:
        raise ValueError("dual_paraphrase joint eval requires heads exactly {0,1,2}")

    out_h0 = os.path.join(out_dir, f"{checkpoint_name}_head0_dual_paraphrase.jsonl")
    out_h1 = os.path.join(out_dir, f"{checkpoint_name}_head1_dual_paraphrase.jsonl")
    out_h2 = os.path.join(out_dir, f"{checkpoint_name}_head2_dual_paraphrase.jsonl")

    n = 0
    h0_a_f1_sum = 0.0
    h0_a_exact_sum = 0.0
    h1_copy_a_gold_sum = 0.0
    h1_copy_b_gold_sum = 0.0
    h1_copy_both_gold_sum = 0.0
    h1_copy_a_ref_sum = 0.0
    h1_copy_b_ref_sum = 0.0
    h1_copy_both_ref_sum = 0.0
    h2_b_f1_gold_sum = 0.0
    h2_copy_a_gold_sum = 0.0
    h2_mixed_gold_sum = 0.0
    h2_copy_a_ref_sum = 0.0
    h2_mixed_ref_sum = 0.0
    start = time.time()

    with open(out_h0, "w", encoding="utf-8") as f0, open(out_h1, "w", encoding="utf-8") as f1, open(out_h2, "w", encoding="utf-8") as f2:
        for i, ex in enumerate(examples):
            system_text, prompt, gold = build_prompt_and_gold("dual_paraphrase", ex)

            prefix_by_head = [""] * infer.assistant_heads
            for h in [0, 1, 2]:
                s = spec_by_head[h]
                if s.final_prefix:
                    prefix_by_head[h] = s.final_prefix

            max_new_tokens = max(spec_by_head[h].max_new_tokens for h in [0, 1, 2])
            max_steps = max(spec_by_head[h].max_steps for h in [0, 1, 2])

            pred_all = infer.generate_all_heads(
                system_text=system_text,
                question_text=prompt,
                final_prefix_by_head=prefix_by_head,
                max_new_tokens=max_new_tokens,
                max_steps=max_steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                stop_on_im_end=stop_on_im_end,
            )
            pred_h0 = pred_all.get(0, "")
            pred_h1 = pred_all.get(1, "")
            pred_h2 = pred_all.get(2, "")

            a_gold = gold.get("a_gold", "")
            b_gold = gold.get("b_gold", "")
            a_norm = normalize_text_loose(a_gold)
            b_norm = normalize_text_loose(b_gold)
            h0_norm = normalize_text_loose(pred_h0)
            h1_norm = normalize_text_loose(pred_h1)
            h2_norm = normalize_text_loose(pred_h2)

            # head0
            a_part = extract_after_marker(pred_h0, "Sentence A rewrite:")
            if not a_part:
                a_part = pred_h0
            a_f1 = token_f1_loose(a_part, a_gold)
            a_exact = 1.0 if normalize_text_loose(a_part) == a_norm else 0.0
            h0_a_f1_sum += a_f1
            h0_a_exact_sum += a_exact
            f0.write(
                json.dumps(
                    {
                        "checkpoint": checkpoint_name,
                        "task": "dual_paraphrase",
                        "head": 0,
                        "index": i,
                        "system": system_text,
                        "prompt": prompt,
                        "gold": gold,
                        "pred_text": pred_h0,
                        "a_f1": a_f1,
                        "a_exact": a_exact,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            # head1
            copy_a_hit_gold = 1.0 if a_norm and a_norm in h1_norm else 0.0
            copy_b_hit_gold = 1.0 if b_norm and b_norm in h1_norm else 0.0
            copy_both_hit_gold = 1.0 if (copy_a_hit_gold > 0 and copy_b_hit_gold > 0) else 0.0
            copy_a_hit_ref = 1.0 if h0_norm and h0_norm in h1_norm else 0.0
            copy_b_hit_ref = 1.0 if h2_norm and h2_norm in h1_norm else 0.0
            copy_both_hit_ref = 1.0 if (copy_a_hit_ref > 0 and copy_b_hit_ref > 0) else 0.0
            h1_copy_a_gold_sum += copy_a_hit_gold
            h1_copy_b_gold_sum += copy_b_hit_gold
            h1_copy_both_gold_sum += copy_both_hit_gold
            h1_copy_a_ref_sum += copy_a_hit_ref
            h1_copy_b_ref_sum += copy_b_hit_ref
            h1_copy_both_ref_sum += copy_both_hit_ref
            f1.write(
                json.dumps(
                    {
                        "checkpoint": checkpoint_name,
                        "task": "dual_paraphrase",
                        "head": 1,
                        "index": i,
                        "system": system_text,
                        "prompt": prompt,
                        "gold": gold,
                        "pred_text": pred_h1,
                        "copy_a_hit_gold": copy_a_hit_gold,
                        "copy_b_hit_gold": copy_b_hit_gold,
                        "copy_both_hit_gold": copy_both_hit_gold,
                        "copy_a_hit_ref": copy_a_hit_ref,
                        "copy_b_hit_ref": copy_b_hit_ref,
                        "copy_both_hit_ref": copy_both_hit_ref,
                        "ref_head0_text": pred_h0,
                        "ref_head2_text": pred_h2,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            # head2
            b_part = extract_between_markers(
                pred_h2,
                "Sentence B rewrite:",
                "Copied full rewrite from channel 0:",
            )
            if not b_part:
                b_part = extract_after_marker(pred_h2, "Sentence B rewrite:")
            if not b_part:
                b_part = pred_h2
            b_f1 = token_f1_loose(b_part, b_gold)
            copy_a_hit_gold_h2 = 1.0 if a_norm and a_norm in h2_norm else 0.0
            mixed_gold = 0.5 * b_f1 + 0.5 * copy_a_hit_gold_h2
            copy_a_hit_ref_h2 = 1.0 if h0_norm and h0_norm in h2_norm else 0.0
            mixed_ref = 0.5 * b_f1 + 0.5 * copy_a_hit_ref_h2
            h2_b_f1_gold_sum += b_f1
            h2_copy_a_gold_sum += copy_a_hit_gold_h2
            h2_mixed_gold_sum += mixed_gold
            h2_copy_a_ref_sum += copy_a_hit_ref_h2
            h2_mixed_ref_sum += mixed_ref
            f2.write(
                json.dumps(
                    {
                        "checkpoint": checkpoint_name,
                        "task": "dual_paraphrase",
                        "head": 2,
                        "index": i,
                        "system": system_text,
                        "prompt": prompt,
                        "gold": gold,
                        "pred_text": pred_h2,
                        "b_f1": b_f1,
                        "copy_a_hit_gold": copy_a_hit_gold_h2,
                        "mixed_score_gold": mixed_gold,
                        "copy_a_hit_ref": copy_a_hit_ref_h2,
                        "mixed_score_ref": mixed_ref,
                        "ref_head0_text": pred_h0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            n += 1

    elapsed = time.time() - start
    tps = n / max(elapsed, 1e-9)
    row0 = {
        "checkpoint": checkpoint_name,
        "task": "dual_paraphrase",
        "head": 0,
        "num_samples": n,
        "elapsed_sec": elapsed,
        "throughput_samples_per_sec": tps,
        "output_jsonl": out_h0,
        "score": h0_a_f1_sum / max(n, 1),
        "a_exact": h0_a_exact_sum / max(n, 1),
        "primary_metric": "a_f1",
        "primary_value": h0_a_f1_sum / max(n, 1),
    }
    row1 = {
        "checkpoint": checkpoint_name,
        "task": "dual_paraphrase",
        "head": 1,
        "num_samples": n,
        "elapsed_sec": elapsed,
        "throughput_samples_per_sec": tps,
        "output_jsonl": out_h1,
        "score": h1_copy_both_gold_sum / max(n, 1),
        "copy_a_hit_gold": h1_copy_a_gold_sum / max(n, 1),
        "copy_b_hit_gold": h1_copy_b_gold_sum / max(n, 1),
        "copy_both_hit_gold": h1_copy_both_gold_sum / max(n, 1),
        "copy_a_hit_ref": h1_copy_a_ref_sum / max(n, 1),
        "copy_b_hit_ref": h1_copy_b_ref_sum / max(n, 1),
        "copy_both_hit_ref": h1_copy_both_ref_sum / max(n, 1),
        "primary_metric": "copy_both_hit_gold",
        "primary_value": h1_copy_both_gold_sum / max(n, 1),
    }
    row2 = {
        "checkpoint": checkpoint_name,
        "task": "dual_paraphrase",
        "head": 2,
        "num_samples": n,
        "elapsed_sec": elapsed,
        "throughput_samples_per_sec": tps,
        "output_jsonl": out_h2,
        "score": h2_mixed_gold_sum / max(n, 1),
        "b_f1_gold": h2_b_f1_gold_sum / max(n, 1),
        "copy_a_hit_gold": h2_copy_a_gold_sum / max(n, 1),
        "mixed_score_gold": h2_mixed_gold_sum / max(n, 1),
        "copy_a_hit_ref": h2_copy_a_ref_sum / max(n, 1),
        "mixed_score_ref": h2_mixed_ref_sum / max(n, 1),
        "primary_metric": "mixed_score_gold",
        "primary_value": h2_mixed_gold_sum / max(n, 1),
    }
    return [row0, row1, row2]


def discover_checkpoints(run_dir: str) -> list[str]:
    if not os.path.isdir(run_dir):
        raise ValueError(f"run_dir not found: {run_dir}")
    cands = []
    for name in os.listdir(run_dir):
        p = os.path.join(run_dir, name)
        if not os.path.isdir(p):
            continue
        if re.match(r"^checkpoint-\d+$", name):
            cands.append(p)
    cands.sort(key=lambda x: int(re.findall(r"checkpoint-(\d+)$", x)[0]))
    if not cands:
        raise ValueError(f"No checkpoint-* dirs found in {run_dir}")
    return cands


def parse_specs(spec_arg: str) -> list[TaskSpec]:
    text = ""
    if os.path.isfile(spec_arg):
        with open(spec_arg, encoding="utf-8") as f:
            text = f.read()
    else:
        text = spec_arg
    raw = json.loads(text)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("task specs must be a non-empty JSON list")
    specs: list[TaskSpec] = []
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            raise ValueError(f"task spec #{i} must be dict")
        specs.append(
            TaskSpec(
                head=int(obj["head"]),
                task=str(obj["task"]).lower(),
                split=obj.get("split"),
                n=int(obj.get("n", 0)),
                seed=int(obj.get("seed", 42)),
                in_jsonl=obj.get("in_jsonl"),
                final_prefix=str(obj.get("final_prefix", "")),
                max_new_tokens=int(obj.get("max_new_tokens", 256)),
                max_steps=int(obj.get("max_steps", 2048)),
            )
        )
    return specs


def checkpoint_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", checkpoint_name(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description="Evaluate multiple checkpoints on multiple heads/tasks with task-specific metrics.")
    ap.add_argument("--run_dir", required=True, help="Directory containing checkpoint-*")
    ap.add_argument(
        "--task_specs_json",
        required=True,
        help="JSON string or path to JSON file. Example: "
        '[{"head":0,"task":"gsm8k","split":"test","n":200},'
        '{"head":1,"task":"squad","split":"validation","n":200},'
        '{"head":2,"task":"proofwriter","split":"test","n":200}]',
    )
    ap.add_argument("--assistant_heads", type=int, default=3)
    ap.add_argument("--max_checkpoints", type=int, default=0, help="0 means all")
    ap.add_argument(
        "--checkpoint_names",
        type=str,
        default="",
        help="Optional comma-separated checkpoint dir names to run, e.g. checkpoint-100,checkpoint-200",
    )
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--stop_on_im_end", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="eval_ckpt_3head_out")
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    specs = parse_specs(args.task_specs_json)
    for s in specs:
        if s.head < 0 or s.head >= args.assistant_heads:
            raise ValueError(f"task spec head {s.head} out of range [0,{args.assistant_heads - 1}]")

    ckpts = discover_checkpoints(args.run_dir)
    if args.checkpoint_names.strip():
        wanted = set([x.strip() for x in args.checkpoint_names.split(",") if x.strip()])
        ckpts = [p for p in ckpts if checkpoint_name(p) in wanted]
        if not ckpts:
            raise ValueError(f"No checkpoints matched --checkpoint_names={args.checkpoint_names}")
    if args.max_checkpoints > 0:
        ckpts = ckpts[: args.max_checkpoints]

    examples_cache: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        key = json.dumps(
            {
                "task": spec.task,
                "split": spec.split,
                "n": spec.n,
                "seed": spec.seed,
                "in_jsonl": spec.in_jsonl,
            },
            sort_keys=True,
        )
        if key not in examples_cache:
            examples_cache[key] = load_examples_for_task(spec)

    all_rows: list[dict[str, Any]] = []
    for ckpt in ckpts:
        ckpt_name = checkpoint_name(ckpt)
        print(f"\n[Checkpoint] {ckpt_name}")
        infer = MultiHeadMedusaInference(
            model_path=ckpt,
            assistant_heads=args.assistant_heads,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )

        all_dual = all(s.task == "dual_paraphrase" for s in specs)
        heads = sorted([s.head for s in specs])
        if all_dual and heads == [0, 1, 2]:
            key0 = json.dumps(
                {
                    "task": specs[0].task,
                    "split": specs[0].split,
                    "n": specs[0].n,
                    "seed": specs[0].seed,
                    "in_jsonl": specs[0].in_jsonl,
                },
                sort_keys=True,
            )
            exs = examples_cache[key0]
            print(f"[Eval-Joint] {ckpt_name} | heads=0,1,2 task=dual_paraphrase n={len(exs)}")
            rows = evaluate_dual_paraphrase_joint(
                infer=infer,
                checkpoint_name=ckpt_name,
                specs=specs,
                examples=exs,
                out_dir=args.out_dir,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
                stop_on_im_end=args.stop_on_im_end,
            )
            for row in rows:
                row["checkpoint_step"] = checkpoint_step(ckpt)
                all_rows.append(row)
        else:
            for spec in specs:
                key = json.dumps(
                    {
                        "task": spec.task,
                        "split": spec.split,
                        "n": spec.n,
                        "seed": spec.seed,
                        "in_jsonl": spec.in_jsonl,
                    },
                    sort_keys=True,
                )
                exs = examples_cache[key]
                print(f"[Eval] {ckpt_name} | head={spec.head} task={spec.task} n={len(exs)}")
                row = evaluate_task_on_head(
                    infer=infer,
                    checkpoint_name=ckpt_name,
                    spec=spec,
                    examples=exs,
                    out_dir=args.out_dir,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                    stop_on_im_end=args.stop_on_im_end,
                )
                row["checkpoint_step"] = checkpoint_step(ckpt)
                all_rows.append(row)

        del infer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_rows.sort(key=lambda x: (int(x.get("checkpoint_step", -1)), int(x.get("head", -1))))

    summary_json = os.path.join(args.out_dir, "checkpoint_head_task_metrics.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(args.out_dir, "checkpoint_head_task_metrics.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(
            "checkpoint,checkpoint_step,head,task,num_samples,primary_metric,primary_value,accuracy,em,f1,elapsed_sec,throughput_samples_per_sec\n"
        )
        for r in all_rows:
            f.write(
                f"{r.get('checkpoint', '')},"
                f"{r.get('checkpoint_step', '')},"
                f"{r.get('head', '')},"
                f"{r.get('task', '')},"
                f"{r.get('num_samples', '')},"
                f"{r.get('primary_metric', '')},"
                f"{r.get('primary_value', '')},"
                f"{r.get('accuracy', '')},"
                f"{r.get('em', '')},"
                f"{r.get('f1', '')},"
                f"{r.get('elapsed_sec', '')},"
                f"{r.get('throughput_samples_per_sec', '')}\n"
            )

    print("\n[Done] Wrote:")
    print(summary_json)
    print(summary_csv)


if __name__ == "__main__":
    main()
