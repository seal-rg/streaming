#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Multi-Channel Dataset Builder (3Q, 3C, 3 modes)
— NATURALIZED + PAUSE-STRICT + EMPTY-IDLE
— TAG-STRIP (UNWRAP)
— SOLUTION SOURCE: expected_response (STRICT; NO fallback to channels)
— DO NOT DROP REASONING:
    - third channel ALWAYS outputs FULL solution in Seg1 (no trunc, no split)
    - only (actor, pauser) show partial work in Seg1 and then finish later
— FEWER SEGMENTS: fixed to 5 segments total
    Seg0: assignment
    Seg1: parallel work + one strict pause
          - ONLY pauser appends pause notice
          - ONLY (actor, pauser) are aligned/truncated
          - third channel outputs full expected_response (cleaned)
    Seg2: plan proposal (ONLY actor speaks; paused is empty; third is EMPTY)
    Seg3: completion (ONLY paused uses to:all; actor completes with clear role separation;
          third is empty)
    Seg4: global summary (single summarizer; others "Done.")

NEW (your latest constraint)
----------------------------
- In Seg1, the takeover-capable channel (actor) must be truncated *further* than the paused channel's
  truncation point (paused BODY cutpoint; pause suffix not counted).
  Concretely:
      tokens(actor_part1_tr) >= tokens(paused_body_tr) + MIN_GAP_TOK
  (as long as both texts are non-empty and budgets allow).
  This avoids cases where the actor looks "behind" the paused solver.

IMPORTANT
---------
- DO NOT remove "The answer is: ..." lines
- DO remove ONLY "#### ..." lines
- Seg2 third-channel text is "" (empty), not "Done with my question."

Modes:
A) actor finishes own question first, then finishes paused question (both in Seg3)
B) swap: actor finishes paused question; paused channel finishes actor question (both in Seg3)
C) postpone: actor finishes paused question first, then returns to finish own question (both in Seg3)

Input JSONL item:
{
  "problem": str,
  "expected_response": str (or list[str]),
  "sample_id": optional
}
"""

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None


# ======================================================================
#                         TAG STRIPPING (UNWRAP)
# ======================================================================

_TO_WRAP_RE = re.compile(r"<to:([^>\s]+)>(.*?)</to:\1>", flags=re.DOTALL | re.IGNORECASE)
_TO_TAG_RE = re.compile(r"</?to:[^>]+>", flags=re.IGNORECASE)

def strip_to_tags(text: str) -> str:
    """UNWRAP: remove <to:...> tags, keep inner content."""
    if not text:
        return ""
    s = text
    prev = None
    while prev != s:
        prev = s
        s = _TO_WRAP_RE.sub(lambda m: m.group(2), s)
    s = _TO_TAG_RE.sub("", s)

    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def wrap_to_all(text: str) -> str:
    # kept for readability; mkmsg will unwrap
    return f"<to:all>{text}</to:all>"


# ======================================================================
#          CLEAN ONLY "#### ..." LINES (KEEP "The answer is")
# ======================================================================

_HASH_ANS_LINE_RE = re.compile(r"^\s*####\s*([^\n\r]+)\s*$", flags=re.MULTILINE)

def clean_hash_answer_lines(text: str) -> str:
    """
    Remove ONLY lines like:
      #### 36
    Keep:
      The answer is: 36
    Keeps reasoning; leaves \\boxed{} untouched.
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = _HASH_ANS_LINE_RE.sub("", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()


def mkmsg(channel: int, text: str) -> Dict[str, Any]:
    return {"channel": channel, "text": clean_hash_answer_lines(strip_to_tags(text))}


# ======================================================================
#                     TOKEN COUNTING + TRUNCATION
# ======================================================================

def approx_count_tokens(text: str) -> int:
    return max(1, len((text or "")) // 4)

SAFE_BOUNDARY_PATTERNS = ["\n\n", "\n", ". ", "; ", ": "]

@dataclass
class TokHelper:
    tokenizer: Optional[Any] = None

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return approx_count_tokens(text)


def find_safe_cut_positions(text: str) -> List[int]:
    text = text or ""
    cuts = set()
    for pat in SAFE_BOUNDARY_PATTERNS:
        start = 0
        while True:
            idx = text.find(pat, start)
            if idx == -1:
                break
            cuts.add(idx + len(pat))
            start = idx + len(pat)
    for m in re.finditer(r"\n", text):
        cuts.add(m.end())
    return sorted(cuts)


def truncate_to_tokens(tok: TokHelper, text: str, target_tokens: int, min_tokens: int = 12) -> str:
    """
    Truncate to ~target_tokens, snapping to safe boundaries.
    ONLY used in Seg1 alignment for (actor, pauser).
    """
    text = (text or "").strip()
    if not text:
        return ""
    t = tok.count(text)
    if t <= target_tokens:
        return text

    target_tokens = max(min_tokens, target_tokens)
    cuts = find_safe_cut_positions(text)
    if not cuts:
        char_cut = int(len(text) * target_tokens / max(1, t))
        return text[:max(1, char_cut)].rstrip()

    best = None
    best_gap = 10**18
    for c in cuts:
        head = text[:c].rstrip()
        ht = tok.count(head)
        if ht > target_tokens:
            continue
        gap = target_tokens - ht
        if gap < best_gap:
            best_gap = gap
            best = head
    if best is not None and tok.count(best) >= min_tokens:
        return best

    best = None
    best_over = 10**18
    for c in cuts:
        head = text[:c].rstrip()
        ht = tok.count(head)
        over = ht - target_tokens
        if over < 0:
            continue
        if over < best_over:
            best_over = over
            best = head
    return best or text


# ======================================================================
#                           SPLITTING SOLUTIONS
# ======================================================================

def split_by_target_tokens(tok: TokHelper, text: str, p: float, min_head_tokens: int = 25) -> Tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "", ""
    total_t = tok.count(text)
    if total_t <= min_head_tokens + 10:
        mid = max(1, len(text) // 2)
        return text[:mid].rstrip(), text[mid:].lstrip()

    target = int(round(p * total_t))
    target = max(min_head_tokens, min(target, total_t - 10))

    cuts = find_safe_cut_positions(text)
    if not cuts:
        mid = max(1, len(text) // 2)
        return text[:mid].rstrip(), text[mid:].lstrip()

    def boxed_ok(full: str, tail: str) -> bool:
        if "\\boxed{" in full:
            return "\\boxed{" in tail
        return True

    best = None
    best_score = 10**18
    for c in cuts:
        head = text[:c].rstrip()
        tail = text[c:].lstrip()
        ht = tok.count(head)
        if ht < min_head_tokens:
            continue
        if not boxed_ok(text, tail):
            continue
        score = abs(ht - target)
        if score < best_score:
            best_score = score
            best = (head, tail)
    if best is not None:
        return best

    best = None
    best_score = 10**18
    for c in cuts:
        head = text[:c].rstrip()
        tail = text[c:].lstrip()
        ht = tok.count(head)
        if ht < min_head_tokens:
            continue
        score = abs(ht - target)
        if score < best_score:
            best_score = score
            best = (head, tail)
    if best is not None:
        return best

    mid = max(1, len(text) // 2)
    return text[:mid].rstrip(), text[mid:].lstrip()


def split_into_three_parts(tok: TokHelper, text: str, p1: float, p2_of_rest: float) -> Tuple[str, str, str]:
    full = (text or "").strip()
    if not full:
        return "", "", ""
    a, rest = split_by_target_tokens(tok, full, p=p1, min_head_tokens=20)
    b, c = split_by_target_tokens(tok, rest, p=p2_of_rest, min_head_tokens=15)

    if "\\boxed{" in full and "\\boxed{" not in c:
        b = ""
        c = rest.strip()

    return a.strip(), b.strip(), c.strip()


# ======================================================================
#                           SUMMARY EXTRACTION
# ======================================================================

def extract_final_answer(text: str, n_fallback: int = 70) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", s)
    if boxed:
        return f"\\boxed{{{boxed[-1]}}}"

    m2_all = list(re.finditer(r"\bThe\s+answer\s+is\s*:\s*([^\n\r]+)", s, flags=re.IGNORECASE))
    if m2_all:
        val = (m2_all[-1].group(1) or "").strip().rstrip(" .。!！?？")
        if val:
            return val

    t = re.sub(r"\s+", " ", s).strip()
    return (t[:n_fallback] + "...") if len(t) > n_fallback else t


# ======================================================================
#                     DIVERSE PROBLEM HEADER
# ======================================================================

_PROMPT_HEADERS = [
    "Please answer the following questions:",
    "Solve the following problems:",
    "Work out the following questions:",
    "Answer each of the following questions:",
    "Please provide solutions to the following questions:",
    "Compute the answers to the following:",
    "Find the answers to the following questions:",
]

_Q_LABEL_STYLES = [
    "Q{i}: {q}",
    "Q{i}. {q}",
    "{i}) {q}",
    "{i}. {q}",
    "Problem {i}: {q}",
]

def get_question_str(qs: List[str]) -> str:
    header = random.choice(_PROMPT_HEADERS)
    style = random.choice(_Q_LABEL_STYLES)
    lines = [style.format(i=idx, q=q) for idx, q in enumerate(qs, start=1)]
    return header + "\n" + "\n".join(lines)


# ======================================================================
#                           CORE BUILDER
# ======================================================================

@dataclass
class BuildConfig:
    tokenizer_path: Optional[str]
    mode: str
    seed: int
    p1_choices: Tuple[float, ...] = (0.40, 0.45, 0.50)
    p2_rest_choices: Tuple[float, ...] = (0.50, 0.55, 0.60)
    seg1_target_cap: int = 220  # cap for aligned pair only
    seg1_min_gap_tokens: int = 10  # NEW: actor_cut >= paused_cut + gap (body-only)


class InteractiveBuilder:
    def __init__(self, cfg: BuildConfig):
        random.seed(cfg.seed)
        self.cfg = cfg
        self.tok = TokHelper(tokenizer=None)
        if cfg.tokenizer_path and HAS_TRANSFORMERS:
            self.tok.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)

    def _base_assign(self) -> Dict[int, int]:
        return {0: 0, 1: 1, 2: 2}

    def _select_pause_channel(self) -> int:
        return random.choice([0, 1, 2])

    def _select_actor_channel(self, pause_ch: int) -> int:
        return random.choice([c for c in [0, 1, 2] if c != pause_ch])

    def _get_solution_text(self, item: Dict[str, Any]) -> str:
        er = item.get("expected_response", "")
        if isinstance(er, list):
            for x in er:
                if isinstance(x, str) and x.strip():
                    return x.strip()
            return ""
        if isinstance(er, str) and er.strip():
            return er.strip()
        return ""

    def build_sample(self, items3: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(items3) == 3, "Expect exactly 3 questions per sample."

        problems = [it["problem"] for it in items3]
        formatted_problem = get_question_str(problems)

        full_sol_raw = [self._get_solution_text(it) for it in items3]
        for i, s in enumerate(full_sol_raw):
            if not (s or "").strip():
                sid = items3[i].get("sample_id", f"index_{i}")
                raise ValueError(f"Missing/empty expected_response for item {i} (sample_id={sid}). Strict mode forbids fallback.")

        full_sol = [clean_hash_answer_lines(s) for s in full_sol_raw]

        p1 = random.choice(self.cfg.p1_choices)
        p2r = random.choice(self.cfg.p2_rest_choices)
        parts = [split_into_three_parts(self.tok, full_sol[i], p1=p1, p2_of_rest=p2r) for i in range(3)]

        base_assign = self._base_assign()

        ch_pause = self._select_pause_channel()
        q_star = base_assign[ch_pause]

        mode = self.cfg.mode
        if mode == "auto":
            mode = random.choice(["A", "B", "C"])
        mode = mode.upper()
        if mode not in ("A", "B", "C"):
            raise ValueError("mode must be A, B, C, or auto")

        ch_actor = self._select_actor_channel(ch_pause)
        q_actor = base_assign[ch_actor]

        ch_third = [c for c in [0, 1, 2] if c not in (ch_pause, ch_actor)][0]
        q_third = base_assign[ch_third]

        segments: List[List[Dict[str, Any]]] = []
        cost = {0: 0, 1: 0, 2: 0}

        def accrue(seg: List[Dict[str, Any]]) -> None:
            for m in seg:
                cost[m["channel"]] += self.tok.count(m["text"])

        PAUSE_PREFIX = [
            "The key step is set up.",
            "I have the setup ready.",
            "Up to this point, the setup is complete.",
        ]

        def join_nonempty(*chunks: str) -> str:
            xs = [c.strip() for c in chunks if (c or "").strip()]
            return "\n\n".join(xs).strip()

        def pause_suffix_text(pause_prefix: str) -> str:
            return f"{pause_prefix}\nI will pause here before stating the final result."

        def pause_suffix_tokens(pause_prefix: str) -> int:
            suffix = clean_hash_answer_lines(strip_to_tags(pause_suffix_text(pause_prefix)))
            return self.tok.count(suffix)

        def make_paused_msg(body: str, pause_prefix: str) -> str:
            body = (body or "").strip()
            return (
                body
                + ("\n" if body else "")
                + pause_prefix
                + "\n"
                + wrap_to_all("I will pause here before stating the final result.")
            )

        def rest_of_question(qi: int) -> str:
            mid = clean_hash_answer_lines(parts[qi][1] or "")
            tail = clean_hash_answer_lines(parts[qi][2] or "")
            return join_nonempty(mid, tail)

        # ---------------- Seg0
        seg0 = [
            mkmsg(0, wrap_to_all("I am channel C0. I will take question 1.")),
            mkmsg(1, wrap_to_all("I am channel C1. I will take question 2.")),
            mkmsg(2, wrap_to_all("I am channel C2. I will take question 3.")),
        ]
        segments.append(seg0)
        accrue(seg0)

        # ---------------- Seg1
        pause_prefix = random.choice(PAUSE_PREFIX)
        seg1_text = ["", "", ""]

        # third channel ALWAYS full in seg1
        seg1_text[ch_third] = full_sol[q_third]

        actor_part1 = (parts[q_actor][0] or "").strip()
        paused_part1 = (parts[q_star][0] or "").strip()

        if actor_part1 and paused_part1:
            suffix_tok = pause_suffix_tokens(pause_prefix)

            t_actor_full = self.tok.count(actor_part1)
            t_paused_full = self.tok.count(paused_part1)

            cap = self.cfg.seg1_target_cap
            gap = max(1, int(self.cfg.seg1_min_gap_tokens))

            # We choose actor_budget first (how far actor goes).
            actor_budget = min(t_actor_full, cap)

            # Paused total budget is capped because it must also include pause suffix.
            paused_total_cap = max(24, min(t_paused_full + suffix_tok, cap))

            # Ensure actor is not shorter than the paused total cap if possible.
            actor_budget = max(actor_budget, min(cap, paused_total_cap))

            # Now choose paused body budget so that:
            #   paused_body_budget + suffix_tok <= cap
            # AND
            #   actor_budget >= paused_body_budget + gap
            paused_body_max_by_cap = max(12, cap - suffix_tok)
            paused_body_budget = min(t_paused_full, paused_body_max_by_cap)

            # enforce the "actor further than paused cutpoint" constraint
            if actor_budget > 0:
                desired_paused_body = max(12, min(paused_body_budget, actor_budget - gap))
                paused_body_budget = min(paused_body_budget, desired_paused_body)

                # if actor is still not > paused_body due to tiny actor, shrink paused again
                if actor_budget <= paused_body_budget:
                    paused_body_budget = max(12, actor_budget - 1)

                # if budgets got too small, just keep minimal safety
                paused_body_budget = max(12, min(paused_body_budget, t_paused_full))

            actor_tr = truncate_to_tokens(self.tok, actor_part1, actor_budget).strip()
            paused_body_tr = truncate_to_tokens(self.tok, paused_part1, paused_body_budget).strip()

            seg1_text[ch_actor] = actor_tr
            seg1_text[ch_pause] = make_paused_msg(paused_body_tr, pause_prefix)
        else:
            seg1_text[ch_actor] = actor_part1
            seg1_text[ch_pause] = make_paused_msg(paused_part1, pause_prefix)

        seg1 = [mkmsg(0, seg1_text[0]), mkmsg(1, seg1_text[1]), mkmsg(2, seg1_text[2])]
        segments.append(seg1)
        accrue(seg1)

        # ---------------- Seg2 (plan only)
        seg2_text = ["", "", ""]
        if mode == "A":
            proposal = f"I noticed question {q_star+1} was paused. I can finish it after I finish my own question {q_actor+1}."
        elif mode == "B":
            proposal = f"I noticed question {q_star+1} was paused. I can finish question {q_star+1}, and you can finish my question {q_actor+1}."
        else:
            proposal = f"I noticed question {q_star+1} was paused. I can finish question {q_star+1} now, and complete my own question {q_actor+1} afterwards."

        seg2_text[ch_actor] = wrap_to_all(proposal)
        seg2_text[ch_pause] = ""
        seg2_text[ch_third] = ""  # explicitly empty

        seg2 = [mkmsg(0, seg2_text[0]), mkmsg(1, seg2_text[1]), mkmsg(2, seg2_text[2])]
        segments.append(seg2)
        accrue(seg2)

        # ---------------- Seg3 (completion)
        seg3_text = ["", "", ""]
        if mode == "A":
            seg3_text[ch_pause] = wrap_to_all(f"Okay. Please proceed with that plan and finish question {q_star+1}.")
            seg3_text[ch_actor] = join_nonempty(
                "Finishing my question first:",
                rest_of_question(q_actor),
                "Now I'll pick up the paused question:",
                rest_of_question(q_star),
            )
        elif mode == "B":
            seg3_text[ch_actor] = join_nonempty(
                "I'll finish the paused question:",
                rest_of_question(q_star),
            )
            seg3_text[ch_pause] = wrap_to_all(
                f"That works. I will take question {q_actor+1} and you finish question {q_star+1}."
                + "\n\n"
                + "I'll take over your question now:\n"
                + rest_of_question(q_actor)
            )
        else:  # C
            seg3_text[ch_pause] = wrap_to_all(f"Okay. Please finish question {q_star+1} first; then return to question {q_actor+1}.")
            seg3_text[ch_actor] = join_nonempty(
                "I'll finish the paused question first:",
                rest_of_question(q_star),
                "Now I'll return to my own question:",
                rest_of_question(q_actor),
            )

        seg3_text[ch_third] = ""

        seg3 = [mkmsg(0, seg3_text[0]), mkmsg(1, seg3_text[1]), mkmsg(2, seg3_text[2])]
        segments.append(seg3)
        accrue(seg3)

        # ---------------- Seg4 (global summary)
        final_ans = [extract_final_answer(full_sol_raw[i]) for i in range(3)]
        summarizer = max(cost.keys(), key=lambda c: cost[c])

        gseg_text = ["", "", ""]
        for ch in [0, 1, 2]:
            if ch == summarizer:
                gseg_text[ch] = wrap_to_all(
                    "Final summary:\n" + "\n".join([f"Question {i+1}: {final_ans[i]}" for i in range(3)])
                )
            else:
                gseg_text[ch] = "Done."

        seg4 = [mkmsg(0, gseg_text[0]), mkmsg(1, gseg_text[1]), mkmsg(2, gseg_text[2])]
        segments.append(seg4)
        accrue(seg4)

        out = {
            "problem": formatted_problem,
            "channels": segments,
            "expected_response": [it.get("expected_response", "") for it in items3],
            "sample_id": f"interact_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}",
            "num_channels": 3,
            "metadata": {
                "mode": mode,
                "paused_channel": ch_pause,
                "paused_question": q_star,
                "actor_channel": ch_actor,
                "actor_question": q_actor,
                "third_channel": ch_third,
                "third_question": q_third,
                "split_p1": p1,
                "split_p2_of_rest": p2r,
                "token_cost": cost,
                "global_summarizer": summarizer,
                "source_samples": [it.get("sample_id") for it in items3],
                "solution_source": "expected_response_strict",
                "truncate_policy": "seg1_align_only_actor_pauser (third is full); enforce actor_cut > paused_cut(body); no_trunc_after_seg1; 5 segments",
                "clean_hash_lines_only": True,
                "keep_the_answer_is": True,
                "seg2_third_empty": True,
                "diverse_problem_header": True,
                "seg1_min_gap_tokens": self.cfg.seg1_min_gap_tokens,
                "num_segments": len(segments),
            },
        }
        return out


# ======================================================================
#                                 I/O
# ======================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Build interaction dataset (3Q/3C) with modes A/B/C. "
                    "Strict expected_response; unwrap <to:...>; "
                    "Seg1 truncates only (actor, pauser) counting pause suffix; "
                    "enforce actor_cut > paused_cut(body) by seg1_min_gap_tokens; "
                    "third is always FULL in Seg1; "
                    "Seg2 plan only (third empty); Seg3 completes; Seg4 global summary. "
                    "Remove ONLY '#### ...' lines; KEEP 'The answer is:' lines. "
                    "Diverse problem header styles enabled."
    )
    ap.add_argument("--input", type=str, required=True, help="Input JSONL (per-question items)")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL (aggregated samples)")
    ap.add_argument("--mode", type=str, default="auto", choices=["auto", "A", "B", "C"])
    ap.add_argument("--tokenizer", type=str, default=None, help="HF tokenizer path for accurate token counting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--drop_last", action="store_true")
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--seg1-min-gap", type=int, default=10, help="Min token gap: actor_cut >= paused_cut(body) + gap (Seg1).")
    args = ap.parse_args()

    random.seed(args.seed)

    cfg = BuildConfig(
        tokenizer_path=args.tokenizer,
        mode=args.mode,
        seed=args.seed,
        seg1_min_gap_tokens=max(1, int(args.seg1_min_gap)),
    )
    builder = InteractiveBuilder(cfg)

    items = load_jsonl(args.input)
    if args.limit is not None:
        items = items[: args.limit]
    if args.shuffle:
        random.shuffle(items)
    if not items:
        raise SystemExit("ERROR: input JSONL is empty.")

    outs: List[Dict[str, Any]] = []

    if args.num_samples is not None:
        N = int(args.num_samples)
        if N < 0:
            raise SystemExit("ERROR: --num-samples must be >= 0")
        for _ in range(N):
            if len(items) >= 3:
                batch = random.sample(items, 3)
            else:
                batch = [random.choice(items) for _ in range(3)]
            outs.append(builder.build_sample(batch))
    else:
        for i in range(0, len(items), 3):
            batch = items[i:i + 3]
            if len(batch) < 3:
                if args.drop_last:
                    break
                while len(batch) < 3:
                    batch.append(batch[-1])
            outs.append(builder.build_sample(batch))

    write_jsonl(args.output, outs)
    print("Done.")
    print(f"Input questions: {len(items)}")
    print(f"Output samples : {len(outs)}")
    print(f"Mode           : {args.mode}")
    print("Solution source: expected_response (STRICT)")
    print("Seg1: third is FULL; actor/pauser truncated; actor_cut > paused_cut(body) enforced")
    print("Clean lines: ONLY '#### ...' removed; 'The answer is:' kept")
    print("Seg2 third: empty")
    print("Diverse problem header: on")
    print(f"Seg1 min gap  : {cfg.seg1_min_gap_tokens}")
    if args.num_samples is not None:
        print(f"Forced N       : {args.num_samples}")


if __name__ == "__main__":
    main()
