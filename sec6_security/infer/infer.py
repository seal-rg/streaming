"""
Unified HF evaluation script with reflection + token counting.

Tasks supported (all from HuggingFace Datasets):
- proofwriter      (tasksource/proofwriter)
- gsm8k            (madrylab/gsm8k-platinum, config: main)
- mathqa           (allenai/math_qa)
- logicnli         (tasksource/LogicNLI)
- logiqa           (lucasmccabe/logiqa)
- mmlu_redux       (edinburgh-dawg/mmlu-redux)
- arc_c            (allenai/ai2_arc, config: ARC-Challenge)

NEW:
- Supports sampling params:
  --temperature, --top_p, --top_k, --min_p
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]


# ============================================================
# Utilities
# ============================================================


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def extract_last_number(text: str) -> str | None:
    if not text:
        return None
    t = text.replace(",", " ")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    return nums[-1] if nums else None


def extract_gsm8k_gold(answer_field: str) -> str | None:
    if not answer_field:
        return None
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_field.replace(",", ""))
    return m.group(1) if m else extract_last_number(answer_field)


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


# ============================================================
# FINAL extractors
# ============================================================


def extract_final_choice(text: str) -> str | None:
    m = re.findall(r"^FINAL:\s*([A-F])\s*$", text.strip(), flags=re.I | re.M)
    return m[-1].upper() if m else None


def extract_final_bool3(text: str) -> str | None:
    m = re.findall(r"^FINAL:\s*(True|False|Unknown)\s*$", text.strip(), flags=re.I | re.M)
    return m[-1].capitalize() if m else None


def extract_final_nli(text: str) -> str | None:
    m = re.findall(r"^FINAL:\s*(entailment|contradiction|neutral|self_contradiction)\s*$", text.strip(), flags=re.I | re.M)
    return m[-1].lower() if m else None


def extract_final_number(text: str) -> str | None:
    m = re.findall(r"^FINAL ANSWER:\s*\\boxed\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*$", text.strip(), flags=re.I | re.M)
    return m[-1] if m else None


def parse_mathqa_options(options_field: Any) -> list[str]:
    if options_field is None:
        return []
    if isinstance(options_field, list):
        return [normalize_space(str(x)) for x in options_field]

    s = str(options_field).strip()

    if (s.startswith("[") and s.endswith("]")) or (s.startswith('["') and s.endswith('"]')):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [normalize_space(str(x)) for x in arr]
        except Exception:
            pass

    s2 = s.replace("\n", " ").strip()
    matches = re.findall(r"([a-eA-E])\s*\)?\s*([^\n]*?)(?=(?:\b[a-eA-E]\s*\)?\s*)|$)", s2)
    if matches:
        opt_map = {}
        for letter, txt in matches:
            letter = letter.lower()
            txt = txt.strip(" ,;")
            if txt:
                opt_map[letter] = normalize_space(txt)
        opts = []
        for letter in ["a", "b", "c", "d", "e"]:
            if letter in opt_map:
                opts.append(opt_map[letter])
        if len(opts) >= 4:
            return opts

    chunks = [normalize_space(x) for x in re.split(r"\s*,\s*", s2) if x.strip()]
    return chunks[:5]


def arc_choices_to_options(choices: dict[str, Any]) -> tuple[list[str], list[str]]:
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    pairs = list(zip(labels, texts))
    pairs.sort(key=lambda x: x[0])
    sorted_labels = [p[0] for p in pairs]
    sorted_texts = [normalize_space(str(p[1])) for p in pairs]
    return sorted_labels, sorted_texts


# ============================================================
# Prompt builders
# ============================================================


def build_mc_prompt(question, options, dataset_name, allow_e=False):
    system = "You are a helpful assistant."

    opts = []
    for i, opt in enumerate(options):
        if i >= (5 if allow_e else 4):
            break
        opts.append(f"{CHOICE_LETTERS[i]}. {opt}")

    label_set = "A, B, C, D" + (", E" if allow_e else "")

    user = (
        "Read the question carefully and consider all options.\n"
        "Explain your reasoning, then give the final answer.\n"
        f"The answer must be one of {label_set}.\n\n"
        f"Question:\n{question}\n\n"
        "Options:\n" + "\n".join(opts) + "\n\n"
        "End with:\nFINAL: <LETTER>\n"
    )

    stop_regex = r"^FINAL:\s*[A-F]\s*$"
    return system, user, stop_regex


def build_gsm8k_prompt(question):
    system = "You are a helpful assistant."
    user = (
        "Solve the following math problem step by step.\n"
        "The final answer must be written as:\n"
        "FINAL ANSWER: \\boxed{<number>}\n\n"
        f"Problem:\n{question}\n"
    )

    stop_regex = r"^FINAL ANSWER:\s*\\boxed\{\s*[-+]?\d+(?:\.\d+)?\s*\}\s*$"
    return system, user, stop_regex


def build_proofwriter_prompt(theory, question):
    system = "You are a helpful assistant."
    user = (
        "You are given a theory consisting of facts and rules, and a query.\n"
        "Determine whether the query logically follows from the theory.\n\n"
        "The conclusion must be one of: True, False, or Unknown.\n\n"
        f"Theory:\n{theory}\n\n"
        f"Query:\n{question}\n\n"
        "End with:\nFINAL: <LABEL>\n"
    )

    stop_regex = r"^FINAL:\s*(True|False|Unknown)\s*$"
    return system, user, stop_regex


def build_logicnli_prompt(premise, hypothesis):
    system = "You are a helpful assistant."
    user = (
        "This is a natural language inference task.\n"
        "Determine the relationship between the premise and the hypothesis.\n\n"
        "The label must be one of:\n"
        "entailment, contradiction, neutral, self_contradiction.\n\n"
        f"Premise:\n{premise}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "End with:\nFINAL: <LABEL>\n"
    )

    stop_regex = r"^FINAL:\s*(entailment|contradiction|neutral|self_contradiction)\s*$"
    return system, user, stop_regex


def build_reflection_user(original_user, first_response, final_rule):
    return (
        original_user + "\n\nPrevious response:\n" + (first_response or "").strip() + "\n\nReview the reasoning and answer above.\n"
        "If needed, correct the answer and explain briefly.\n\n"
        "End with:\n"
        f"{final_rule}\n"
    )


# ============================================================
# Model runner
# ============================================================


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class StopOnRegex(StoppingCriteria):
    def __init__(self, tokenizer, start_len, pattern):
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.regex = re.compile(pattern, flags=re.I | re.M)

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0, self.start_len :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return bool(self.regex.search(text))


def build_chat_prompt(tokenizer, system, user):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return system + "\n\n" + user


class HFGenerator:
    def __init__(
        self,
        model,
        device_map="auto",
        dtype="auto",
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=42,
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = "auto" if dtype == "auto" else torch.float16 if dtype == "fp16" else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model, device_map=device_map, torch_dtype=torch_dtype).eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p

    @torch.inference_mode()
    def generate(self, system, user, stop_regex=None):
        prompt = build_chat_prompt(self.tokenizer, system, user)
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = toks["input_ids"].shape[1]

        stopping = StoppingCriteriaList([StopOnRegex(self.tokenizer, prompt_len, stop_regex)]) if stop_regex else None

        do_sample = self.temperature > 0

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=stopping,
        )

        if do_sample:
            gen_kwargs["temperature"] = float(self.temperature)
            gen_kwargs["top_p"] = float(self.top_p)

            # Top-k: 0 disables (do not pass)
            if isinstance(self.top_k, int) and self.top_k > 0:
                gen_kwargs["top_k"] = int(self.top_k)

            # Min-p: 0 disables (do not pass)
            if isinstance(self.min_p, (int, float)) and float(self.min_p) > 0.0:
                gen_kwargs["min_p"] = float(self.min_p)

        out = self.model.generate(**toks, **gen_kwargs)

        gen = out[0][prompt_len:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return GenResult(text, prompt_len, len(gen))


# ============================================================
# Evaluators
# ============================================================


def eval_multiple_choice(
    gen: HFGenerator,
    examples: list[dict[str, Any]],
    dataset_name: str,
    get_q_opts_gold,
    reflection: bool,
    out_path: str,
    allow_e: bool = False,
) -> dict[str, Any]:
    correct_1 = 0
    correct_2 = 0
    pt1, ct1, pt2, ct2 = [], [], [], []

    final_rule = "FINAL: <LETTER>"

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            q, opts, gold = get_q_opts_gold(ex)
            system, user, stop_regex = build_mc_prompt(q, opts, dataset_name, allow_e=allow_e)

            r1 = gen.generate(system, user, stop_regex=stop_regex)
            pred1 = extract_final_choice(r1.text)
            ok1 = pred1 == gold
            correct_1 += int(ok1)

            pt1.append(r1.prompt_tokens)
            ct1.append(r1.completion_tokens)

            rec = {
                "dataset": dataset_name,
                "prompt": build_chat_prompt(gen.tokenizer, system, user),
                "gold": gold,
                "pred_1": pred1,
                "raw_1": r1.text,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
            }

            if reflection:
                label_set = "A, B, C, D" + (", E" if allow_e else "")
                ref_user = build_reflection_user(
                    user,
                    r1.text,
                    final_rule=f"FINAL: <LETTER> where <LETTER> is one of {label_set}",
                )
                r2 = gen.generate(system, ref_user, stop_regex=stop_regex)
                pred2 = extract_final_choice(r2.text)
                ok2 = pred2 == gold
                correct_2 += int(ok2)

                pt2.append(r2.prompt_tokens)
                ct2.append(r2.completion_tokens)

                rec.update(
                    {
                        "pred_2": pred2,
                        "raw_2": r2.text,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples)
    res = {
        "dataset": dataset_name,
        "n": n,
        "acc_1": correct_1 / n if n else 0.0,
        "avg_prompt_tokens_1": safe_mean(pt1),
        "avg_completion_tokens_1": safe_mean(ct1),
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "avg_prompt_tokens_2": safe_mean(pt2),
                "avg_completion_tokens_2": safe_mean(ct2),
            }
        )
    return res


def eval_gsm8k(
    gen: HFGenerator,
    examples: list[dict[str, Any]],
    reflection: bool,
    out_path: str,
) -> dict[str, Any]:
    correct_1 = 0
    correct_2 = 0
    pt1, ct1, pt2, ct2 = [], [], [], []

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            q = ex["question"]
            gold = extract_gsm8k_gold(ex["answer"])

            system, user, stop_regex = build_gsm8k_prompt(q)
            r1 = gen.generate(system, user, stop_regex=stop_regex)

            pred1 = extract_final_number(r1.text)
            ok1 = pred1 is not None and gold is not None and pred1 == gold
            correct_1 += int(ok1)

            pt1.append(r1.prompt_tokens)
            ct1.append(r1.completion_tokens)

            rec = {
                "dataset": "gsm8k",
                "prompt": build_chat_prompt(gen.tokenizer, system, user),
                "gold": gold,
                "pred_1": pred1,
                "raw_1": r1.text,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
            }

            if reflection:
                ref_user = build_reflection_user(
                    user,
                    r1.text,
                    final_rule="FINAL ANSWER: \\boxed{<number>}",
                )
                r2 = gen.generate(system, ref_user, stop_regex=stop_regex)
                pred2 = extract_final_number(r2.text)
                ok2 = pred2 is not None and gold is not None and pred2 == gold
                correct_2 += int(ok2)

                pt2.append(r2.prompt_tokens)
                ct2.append(r2.completion_tokens)

                rec.update(
                    {
                        "pred_2": pred2,
                        "raw_2": r2.text,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples)
    res = {
        "dataset": "gsm8k",
        "n": n,
        "acc_1": correct_1 / n if n else 0.0,
        "avg_prompt_tokens_1": safe_mean(pt1),
        "avg_completion_tokens_1": safe_mean(ct1),
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "avg_prompt_tokens_2": safe_mean(pt2),
                "avg_completion_tokens_2": safe_mean(ct2),
            }
        )
    return res


def eval_proofwriter(
    gen: HFGenerator,
    examples: list[dict[str, Any]],
    reflection: bool,
    out_path: str,
) -> dict[str, Any]:
    correct_1 = 0
    correct_2 = 0
    pt1, ct1, pt2, ct2 = [], [], [], []

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            theory = ex["theory"]
            question = ex["question"]
            gold = ex["answer"]  # True/False/Unknown

            system, user, stop_regex = build_proofwriter_prompt(theory, question)
            r1 = gen.generate(system, user, stop_regex=stop_regex)
            pred1 = extract_final_bool3(r1.text)
            ok1 = pred1 == gold
            correct_1 += int(ok1)

            pt1.append(r1.prompt_tokens)
            ct1.append(r1.completion_tokens)

            rec = {
                "dataset": "proofwriter",
                "prompt": build_chat_prompt(gen.tokenizer, system, user),
                "gold": gold,
                "pred_1": pred1,
                "raw_1": r1.text,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
            }

            if reflection:
                ref_user = build_reflection_user(
                    user,
                    r1.text,
                    final_rule="FINAL: <LABEL> where <LABEL> is one of True, False, Unknown",
                )
                r2 = gen.generate(system, ref_user, stop_regex=stop_regex)
                pred2 = extract_final_bool3(r2.text)
                ok2 = pred2 == gold
                correct_2 += int(ok2)

                pt2.append(r2.prompt_tokens)
                ct2.append(r2.completion_tokens)

                rec.update(
                    {
                        "pred_2": pred2,
                        "raw_2": r2.text,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples)
    res = {
        "dataset": "proofwriter",
        "n": n,
        "acc_1": correct_1 / n if n else 0.0,
        "avg_prompt_tokens_1": safe_mean(pt1),
        "avg_completion_tokens_1": safe_mean(ct1),
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "avg_prompt_tokens_2": safe_mean(pt2),
                "avg_completion_tokens_2": safe_mean(ct2),
            }
        )
    return res


def eval_logicnli(
    gen: HFGenerator,
    examples: list[dict[str, Any]],
    reflection: bool,
    out_path: str,
) -> dict[str, Any]:
    correct_1 = 0
    correct_2 = 0
    pt1, ct1, pt2, ct2 = [], [], [], []

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            premise = ex["premise"]
            hypothesis = ex["hypothesis"]
            gold = ex["label"]

            system, user, stop_regex = build_logicnli_prompt(premise, hypothesis)
            r1 = gen.generate(system, user, stop_regex=stop_regex)
            pred1 = extract_final_nli(r1.text)
            ok1 = pred1 == gold
            correct_1 += int(ok1)

            pt1.append(r1.prompt_tokens)
            ct1.append(r1.completion_tokens)

            rec = {
                "dataset": "logicnli",
                "prompt": build_chat_prompt(gen.tokenizer, system, user),
                "gold": gold,
                "pred_1": pred1,
                "raw_1": r1.text,
                "prompt_tokens_1": r1.prompt_tokens,
                "completion_tokens_1": r1.completion_tokens,
            }

            if reflection:
                ref_user = build_reflection_user(
                    user,
                    r1.text,
                    final_rule="FINAL: <LABEL> where <LABEL> is one of entailment, contradiction, neutral, self_contradiction",
                )
                r2 = gen.generate(system, ref_user, stop_regex=stop_regex)
                pred2 = extract_final_nli(r2.text)
                ok2 = pred2 == gold
                correct_2 += int(ok2)

                pt2.append(r2.prompt_tokens)
                ct2.append(r2.completion_tokens)

                rec.update(
                    {
                        "pred_2": pred2,
                        "raw_2": r2.text,
                        "prompt_tokens_2": r2.prompt_tokens,
                        "completion_tokens_2": r2.completion_tokens,
                    }
                )

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(examples)
    res = {
        "dataset": "logicnli",
        "n": n,
        "acc_1": correct_1 / n if n else 0.0,
        "avg_prompt_tokens_1": safe_mean(pt1),
        "avg_completion_tokens_1": safe_mean(ct1),
    }
    if reflection:
        res.update(
            {
                "acc_2": correct_2 / n if n else 0.0,
                "avg_prompt_tokens_2": safe_mean(pt2),
                "avg_completion_tokens_2": safe_mean(ct2),
            }
        )
    return res


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--device_map", default="auto", help='Transformers device_map, e.g. "auto"')
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"])
    ap.add_argument("--max_new_tokens", type=int, default=256)

    # Sampling params (NEW)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0, help="Top-K sampling (0 disables)")
    ap.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling (0.0 disables)")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--tasks",
        default="proofwriter,gsm8k,mathqa,logicnli,logiqa,mmlu_redux,arc_c",
        help="comma-separated: proofwriter,gsm8k,mathqa,logicnli,logiqa,mmlu_redux,arc_c",
    )
    ap.add_argument("--n_samples", type=int, default=200, help="Per task random sample. 0 means all.")
    ap.add_argument("--reflection", action="store_true", help="Enable reflection second pass")

    ap.add_argument(
        "--mmlu_redux_only_ok", action="store_true", help="If set, evaluate only rows with error_type == 'ok' (when column exists)."
    )

    ap.add_argument("--out_dir", default="eval_outputs", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gen = HFGenerator(
        model=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary: list[dict[str, Any]] = []

    for task in tasks:
        out_path = os.path.join(args.out_dir, f"{task}.jsonl")

        if task == "proofwriter":
            ds_dict = load_dataset("tasksource/proofwriter")
            split = pick_split(ds_dict, preferred=("test", "validation", "train"))
            examples = sample_examples(ds_dict[split], args.n_samples, args.seed)
            res = eval_proofwriter(gen, examples, args.reflection, out_path)
            summary.append(res)

        elif task == "gsm8k":
            ds_dict = load_dataset("madrylab/gsm8k-platinum")  # madrylab/gsm8k-platinum
            split = pick_split(ds_dict, preferred=("test"))
            examples = sample_examples(ds_dict[split], args.n_samples, args.seed)
            res = eval_gsm8k(gen, examples, args.reflection, out_path)
            summary.append(res)

        elif task == "mathqa":
            ds_dict = load_dataset("allenai/math_qa")
            split = pick_split(ds_dict, preferred=("test", "validation", "train"))
            ds = ds_dict[split]
            examples = sample_examples(ds, args.n_samples, args.seed)

            def get_q_opts_gold(ex):
                q = ex.get("Problem", ex.get("question", ""))
                opts = parse_mathqa_options(ex.get("options", ""))
                gold_raw = ex.get("correct", ex.get("answer", ""))
                letter = str(gold_raw).strip().upper()
                if letter in ["A", "B", "C", "D", "E"]:
                    gold = letter
                elif letter.lower() in ["a", "b", "c", "d", "e"]:
                    gold = letter.upper()
                else:
                    gold = None
                return q, opts, gold

            res = eval_multiple_choice(gen, examples, "mathqa", get_q_opts_gold, args.reflection, out_path, allow_e=True)
            summary.append(res)

        elif task == "logicnli":
            ds_dict = load_dataset("tasksource/LogicNLI")
            split = pick_split(ds_dict, preferred=("test", "validation", "train"))
            examples = sample_examples(ds_dict[split], args.n_samples, args.seed)
            res = eval_logicnli(gen, examples, args.reflection, out_path)
            summary.append(res)

        elif task == "logiqa":
            ds_dict = load_dataset("lucasmccabe/logiqa")
            split = pick_split(ds_dict, preferred=("test", "validation", "train"))
            ds = ds_dict[split]
            examples = sample_examples(ds, args.n_samples, args.seed)

            def get_q_opts_gold(ex):
                context = ex.get("context", ex.get("passage", ex.get("text", "")))
                question = ex.get("question", "")
                q = (str(context).strip() + "\n\n" + str(question).strip()).strip()

                opts = ex.get("options", ex.get("choices", ex.get("answer_options", [])))
                if isinstance(opts, list):
                    opts = [normalize_space(str(x)) for x in opts]
                else:
                    opts = [normalize_space(x) for x in str(opts).split("\n") if x.strip()]

                gold_raw = ex.get("label", ex.get("answer", ex.get("correct", None)))
                if isinstance(gold_raw, int):
                    gold = CHOICE_LETTERS[gold_raw]
                else:
                    s = str(gold_raw).strip().upper()
                    gold = s if s in ["A", "B", "C", "D"] else None
                return q, opts, gold

            res = eval_multiple_choice(gen, examples, "logiqa", get_q_opts_gold, args.reflection, out_path, allow_e=False)
            summary.append(res)

        elif task == "mmlu_redux":
            from datasets import concatenate_datasets, get_dataset_config_names

            cfgs = get_dataset_config_names("edinburgh-dawg/mmlu-redux")
            all_ds_parts = []

            for cfg in cfgs:
                ds_cfg = load_dataset("edinburgh-dawg/mmlu-redux", cfg)
                split = pick_split(ds_cfg, preferred=("test", "validation", "dev", "train"))
                part = ds_cfg[split]

                if "subject" not in part.column_names:
                    part = part.add_column("subject", [cfg] * len(part))

                if args.mmlu_redux_only_ok and "error_type" in part.column_names:
                    part = part.filter(lambda x: x.get("error_type", "") == "ok")

                all_ds_parts.append(part)

            ds = concatenate_datasets(all_ds_parts) if len(all_ds_parts) > 1 else all_ds_parts[0]
            examples = sample_examples(ds, args.n_samples, args.seed)

            def get_q_opts_gold(ex):
                q = ex["question"]
                choices = ex["choices"]
                if isinstance(choices, str):
                    try:
                        choices = json.loads(choices.replace("'", '"'))
                    except Exception:
                        choices = [c.strip() for c in choices.strip("[]").split(",")][:4]
                opts = [normalize_space(str(x)) for x in choices]
                gold = CHOICE_LETTERS[int(ex["answer"])]
                return q, opts, gold

            correct_1 = 0
            correct_2 = 0
            pt1, ct1, pt2, ct2 = [], [], [], []

            with open(out_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    subject = ex.get("subject", "")
                    q, opts, gold = get_q_opts_gold(ex)

                    system, user, stop_regex = build_mc_prompt(q, opts, "mmlu_redux", allow_e=False)
                    r1 = gen.generate(system, user, stop_regex=stop_regex)
                    pred1 = extract_final_choice(r1.text)
                    ok1 = pred1 == gold
                    correct_1 += int(ok1)
                    pt1.append(r1.prompt_tokens)
                    ct1.append(r1.completion_tokens)

                    rec = {
                        "dataset": "mmlu_redux",
                        "subject": subject,
                        "prompt": build_chat_prompt(gen.tokenizer, system, user),
                        "gold": gold,
                        "pred_1": pred1,
                        "raw_1": r1.text,
                        "prompt_tokens_1": r1.prompt_tokens,
                        "completion_tokens_1": r1.completion_tokens,
                    }

                    if args.reflection:
                        ref_user = build_reflection_user(
                            user,
                            r1.text,
                            final_rule="FINAL: <LETTER> where <LETTER> is one of A, B, C, D",
                        )
                        r2 = gen.generate(system, ref_user, stop_regex=stop_regex)
                        pred2 = extract_final_choice(r2.text)
                        ok2 = pred2 == gold
                        correct_2 += int(ok2)
                        pt2.append(r2.prompt_tokens)
                        ct2.append(r2.completion_tokens)

                        rec.update(
                            {
                                "pred_2": pred2,
                                "raw_2": r2.text,
                                "prompt_tokens_2": r2.prompt_tokens,
                                "completion_tokens_2": r2.completion_tokens,
                            }
                        )

                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n = len(examples)
            res = {
                "dataset": "mmlu_redux",
                "n": n,
                "acc_1": correct_1 / n if n else 0.0,
                "avg_prompt_tokens_1": safe_mean(pt1),
                "avg_completion_tokens_1": safe_mean(ct1),
            }
            if args.reflection:
                res.update(
                    {
                        "acc_2": correct_2 / n if n else 0.0,
                        "avg_prompt_tokens_2": safe_mean(pt2),
                        "avg_completion_tokens_2": safe_mean(ct2),
                    }
                )
            summary.append(res)

        elif task == "arc_c":
            ds_dict = load_dataset("allenai/ai2_arc", "ARC-Challenge")
            split = pick_split(ds_dict, preferred=("test", "validation", "train"))
            ds = ds_dict[split]
            examples = sample_examples(ds, args.n_samples, args.seed)

            def get_q_opts_gold(ex):
                q = ex["question"]
                _, texts = arc_choices_to_options(ex["choices"])
                gold = str(ex["answerKey"]).strip().upper()
                if gold not in ["A", "B", "C", "D", "E"]:
                    gold = None
                return q, texts, gold

            res = eval_multiple_choice(gen, examples, "arc_c", get_q_opts_gold, args.reflection, out_path, allow_e=True)
            summary.append(res)

        else:
            raise ValueError(f"Unknown task: {task}")

        print(f"[DONE] {task}: {summary[-1]}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Per-task JSONL saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
