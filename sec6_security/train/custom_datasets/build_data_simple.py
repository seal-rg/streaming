#!/usr/bin/env python3

"""
Interactive Multi-Channel Dataset Builder (3Q, 3C, 3 modes) — NATURALIZED + PAUSE-STRICT + EMPTY-IDLE

What this script generates
--------------------------
- Each aggregated sample contains 3 questions (Q1..Q3) and 3 channels (C0..C2).
- Conversation is split into segments (Seg0..), each segment has exactly 3 messages (one per channel).
- Interaction is introduced by:
  - exactly one channel "pauses" in Seg1 (only that channel uses <to:all> in Seg1)
  - an actor proposes a plan in Seg2 (ONLY one <to:all> in Seg2)
  - the paused channel responds in Seg3 (ONLY one <to:all> in Seg3)
- Modes:
  A) Finish-then-Takeover: actor finishes own question first, then finishes the paused question.
  B) Swap: actor finishes paused question; paused channel finishes actor's question (swap ownership).
  C) Postpone: actor finishes paused question first, then returns to finish own question later.

Key constraints enforced
------------------------
1) Pause is strict:
   - If a channel pauses in Seg1, it outputs "" in Seg2 (no more solving content).
2) No meaningless fillers:
   - idle_line() returns "" and idle messages can be empty.
3) Natural but minimal transitions (short phrases):
   - Pause prefix (non-toall) right before the pause notice
   - Short "Continuing:" prefixes on slice messages (kept short)
   - Short "Picking up from paused setup:" prefixes when actor completes the paused question
4) Token alignment:
   - Seg1: all three channels output part1 truncated to same target length interval
   - Seg3: actor + third slices are aligned to same token interval (prefix added BEFORE alignment)
   - Seg2: only third slice is shown (paused is empty; actor is <to:all> proposal)

New feature
-----------
--num-samples N
  Force exactly N output samples, sampling 3 question-items each time.

Input JSONL format (per-question item)
--------------------------------------
{"problem": str, "channels": [str, str, ...], "expected_response": optional, "sample_id": optional}

Output JSONL format (per aggregated 3Q sample)
----------------------------------------------
{
  "problem": "...",
  "channels": [ [ {channel,text}, ... ], ... ],
  "metadata": {...}
}
"""

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None


# ------------------------ formatting helpers ------------------------


def get_question_str(qs: list[str]) -> str:
    return "Please answer the following questions:\n" + "\n".join([f"Q{i + 1}: {q}" for i, q in enumerate(qs)])


def mkmsg(channel: int, text: str) -> dict[str, Any]:
    return {"channel": channel, "text": text}


def wrap_to_all(text: str) -> str:
    return f"<to:all>{text}</to:all>"


def extract_boxed_or_short(text: str, n: int = 70) -> str:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text or "")
    if boxed:
        return f"\\boxed{{{boxed[-1]}}}"
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return (t[:n] + "...") if len(t) > n else t


def approx_count_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


# ------------------------ token helper & splitting ------------------------

SAFE_BOUNDARY_PATTERNS = ["\n\n", "\n", ". ", "; ", ": "]


@dataclass
class TokHelper:
    tokenizer: Any | None = None

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return approx_count_tokens(text)


def find_safe_cut_positions(text: str) -> list[int]:
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


def ensure_boxed_in_tail(full: str, tail: str) -> bool:
    if "\\boxed{" in (full or ""):
        return "\\boxed{" in (tail or "")
    return True


def split_by_target_tokens(
    tok: TokHelper,
    text: str,
    p: float,
    min_head_tokens: int = 25,
) -> tuple[str, str]:
    """
    Split into (head, tail) around p * tokens, snapping to a safe boundary.
    Ensures \\boxed is in tail if present.
    """
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

    best = None
    best_score = 10**18
    for c in cuts:
        head = text[:c].rstrip()
        tail = text[c:].lstrip()
        ht = tok.count(head)
        if ht < min_head_tokens:
            continue
        if not ensure_boxed_in_tail(text, tail):
            continue
        score = abs(ht - target)
        if score < best_score:
            best_score = score
            best = (head, tail)

    if best is not None:
        return best

    # fallback without boxed constraint
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


def truncate_to_tokens(tok: TokHelper, text: str, target_tokens: int, min_tokens: int = 12) -> str:
    """
    Truncate to ~target_tokens, snapping to safe boundaries.
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
        return text[: max(1, char_cut)].rstrip()

    # best cut <= target
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

    # else smallest over target
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


def split_into_three_parts(tok: TokHelper, text: str, p1: float, p2_of_rest: float) -> tuple[str, str, str]:
    """
    Split full solution into three parts: part1, part2, part3
    with \\boxed forced into part3 if present.
    """
    full = (text or "").strip()
    if not full:
        return "", "", ""

    a, rest = split_by_target_tokens(tok, full, p=p1, min_head_tokens=20)
    b, c = split_by_target_tokens(tok, rest, p=p2_of_rest, min_head_tokens=15)

    # Force boxed into c if present
    if "\\boxed{" in full and "\\boxed{" not in c:
        b = ""
        c = rest.strip()

    return a.strip(), b.strip(), c.strip()


# ------------------------ core builder ------------------------


@dataclass
class BuildConfig:
    tokenizer_path: str | None
    mode: str
    seed: int
    # Splitting ratios
    p1_choices: tuple[float, ...] = (0.35, 0.40, 0.45)
    p2_rest_choices: tuple[float, ...] = (0.45, 0.50, 0.55)
    # Alignment targets
    seg1_target_cap: int = 120
    seg2_slice_target: int = 70
    seg3_slice_target: int = 70
    # Execution segments (optional cap; keep huge by default)
    exec_cap: int = 10**9


class InteractiveBuilder:
    def __init__(self, cfg: BuildConfig):
        random.seed(cfg.seed)
        self.cfg = cfg
        self.tok = TokHelper(tokenizer=None)
        if cfg.tokenizer_path and HAS_TRANSFORMERS:
            self.tok.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)

    def _choose_solution(self, solutions: list[str]) -> str:
        sols = list(solutions or [])
        if not sols:
            return ""
        while len(sols) < 3:
            sols.append(sols[0])
        return random.choice(sols[:3])

    def _base_assign(self) -> dict[int, int]:
        # channel -> question index
        return {0: 0, 1: 1, 2: 2}

    def _select_pause_channel(self) -> int:
        return random.choice([0, 1, 2])

    def _select_actor_channel(self, pause_ch: int) -> int:
        return random.choice([c for c in [0, 1, 2] if c != pause_ch])

    def _align_two_slices_same_len(self, s1: str, s2: str, target_cap: int) -> tuple[str, str, int]:
        """
        Truncate both slices to the SAME token length interval.
        target_len = min(count(s1), count(s2), target_cap) but at least 12.
        """
        s1 = (s1 or "").strip()
        s2 = (s2 or "").strip()
        t1 = self.tok.count(s1) if s1 else 0
        t2 = self.tok.count(s2) if s2 else 0
        if t1 == 0 and t2 == 0:
            return "", "", 0
        if t1 == 0:
            return "", truncate_to_tokens(self.tok, s2, max(12, min(t2, target_cap))), min(t2, target_cap)
        if t2 == 0:
            return truncate_to_tokens(self.tok, s1, max(12, min(t1, target_cap))), "", min(t1, target_cap)

        target_len = max(12, min(t1, t2, target_cap))
        return (
            truncate_to_tokens(self.tok, s1, target_len),
            truncate_to_tokens(self.tok, s2, target_len),
            target_len,
        )

    def build_sample(self, items3: list[dict[str, Any]]) -> dict[str, Any]:
        assert len(items3) == 3, "Expect exactly 3 questions per sample."

        problems = [it["problem"] for it in items3]
        formatted_problem = get_question_str(problems)

        # select one solution per question
        full_sol = [self._choose_solution(it.get("channels", [])) for it in items3]

        # split each chosen solution into 3 parts
        p1 = random.choice(self.cfg.p1_choices)
        p2r = random.choice(self.cfg.p2_rest_choices)
        parts = [split_into_three_parts(self.tok, full_sol[i], p1=p1, p2_of_rest=p2r) for i in range(3)]
        # parts[q] = (part1, part2, part3)

        base_assign = self._base_assign()

        # interaction roles
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

        segments: list[list[dict[str, Any]]] = []
        cost = {0: 0, 1: 0, 2: 0}

        def accrue(seg: list[dict[str, Any]]) -> None:
            for m in seg:
                cost[m["channel"]] += self.tok.count(m["text"])

        def idle_line() -> str:
            # No filler by default
            return ""

        # ---- natural tiny transitions (short, low token footprint) ----
        PAUSE_PREFIX = [
            "I have the setup ready.",
            "The key step is set up.",
            "Up to this point, the setup is complete.",
        ]
        CONT_MY = [
            "Continuing:",
            "Next step:",
            "Continuing my question:",
        ]
        CONT_TAKEOVER = [
            "Picking up from the paused setup:",
            "Continuing from the intermediate step:",
            "To finish the paused question:",
        ]

        def with_prefix_then_trunc(prefix: str, body: str, target_tokens: int) -> str:
            """
            Add a short prefix then truncate to target_tokens.
            """
            body = (body or "").strip()
            if not body:
                return ""
            combo = f"{prefix} {body}".strip()
            return truncate_to_tokens(self.tok, combo, target_tokens).strip()

        # ---------------- Seg0: assignment
        seg0 = [
            mkmsg(0, wrap_to_all("I am channel C0. I will take question 1.")),
            mkmsg(1, wrap_to_all("I am channel C1. I will take question 2.")),
            mkmsg(2, wrap_to_all("I am channel C2. I will take question 3.")),
        ]
        segments.append(seg0)
        accrue(seg0)

        # ---------------- Seg1: parallel work + strict pause (only pauser uses <to:all>)
        pause_part1 = (parts[q_star][0] or "").strip()
        pause_len = self.tok.count(pause_part1) if pause_part1 else 0
        target_len_seg1 = max(20, min(pause_len if pause_len else self.cfg.seg1_target_cap, self.cfg.seg1_target_cap))

        seg1_text = ["", "", ""]
        for ch in [0, 1, 2]:
            q = base_assign[ch]
            part1 = (parts[q][0] or "").strip()
            part1_tr = truncate_to_tokens(self.tok, part1, target_len_seg1).strip() if part1 else ""
            if ch == ch_pause:
                seg1_text[ch] = (
                    part1_tr
                    + ("\n" if part1_tr else "")
                    + random.choice(PAUSE_PREFIX)
                    + "\n"
                    + wrap_to_all("I will pause here before stating the final result.")
                )
            else:
                seg1_text[ch] = part1_tr

        seg1 = [mkmsg(0, seg1_text[0]), mkmsg(1, seg1_text[1]), mkmsg(2, seg1_text[2])]
        segments.append(seg1)
        accrue(seg1)

        # ---------------- Seg2: proposal only (ONLY actor uses <to:all>)
        # strict pause => paused channel outputs "" here
        seg2_text = ["", "", ""]
        if mode == "A":
            proposal = wrap_to_all(
                f"I noticed question {q_star + 1} was paused. I can finish it after I finish my own question {q_actor + 1}."
            )
        elif mode == "B":
            proposal = wrap_to_all(
                f"I noticed question {q_star + 1} was paused. If you want, I can finish question {q_star + 1}, and you can finish my question {q_actor + 1}."
            )
        else:  # C
            proposal = wrap_to_all(
                f"I noticed question {q_star + 1} was paused. I can finish question {q_star + 1} now, and complete my own question {q_actor + 1} later."
            )
        seg2_text[ch_actor] = proposal

        # paused channel: truly pauses
        seg2_text[ch_pause] = ""

        # third channel: continues its own part2 (natural prefix + truncate)
        third_part2 = (parts[q_third][1] or "").strip()
        seg2_text[ch_third] = with_prefix_then_trunc(
            random.choice(CONT_MY),
            third_part2,
            self.cfg.seg2_slice_target,
        )

        seg2 = [mkmsg(0, seg2_text[0]), mkmsg(1, seg2_text[1]), mkmsg(2, seg2_text[2])]
        segments.append(seg2)
        accrue(seg2)

        # ---------------- Seg3: response only (ONLY paused channel uses <to:all>)
        seg3_text = ["", "", ""]
        if mode == "A":
            response = wrap_to_all(f"Okay. Please proceed with that plan and finish question {q_star + 1}.")
        elif mode == "B":
            response = wrap_to_all(f"That works for me. I will hand over question {q_star + 1} and take question {q_actor + 1}.")
        else:  # C
            response = wrap_to_all(f"Okay. Please finish question {q_star + 1} first; you can return to question {q_actor + 1} afterwards.")
        seg3_text[ch_pause] = response

        # seg3: non-toall are solution slices aligned to same length interval (prefix BEFORE alignment)
        if mode == "A":
            raw_actor = (parts[q_actor][2] or "").strip()
            raw_third = (parts[q_third][2] or "").strip()
        else:
            raw_actor = (parts[q_actor][1] or "").strip()
            raw_third = (parts[q_third][2] or "").strip()

        raw_actor = (f"{random.choice(CONT_MY)} {raw_actor}".strip()) if raw_actor else ""
        raw_third = (f"{random.choice(CONT_MY)} {raw_third}".strip()) if raw_third else ""

        s_actor_aligned, s_third_aligned, _ = self._align_two_slices_same_len(raw_actor, raw_third, target_cap=self.cfg.seg3_slice_target)
        seg3_text[ch_actor] = (s_actor_aligned or "").strip()
        seg3_text[ch_third] = (s_third_aligned or "").strip()

        seg3 = [mkmsg(0, seg3_text[0]), mkmsg(1, seg3_text[1]), mkmsg(2, seg3_text[2])]
        segments.append(seg3)
        accrue(seg3)

        # ---------------- Execution segments (mode-specific)
        owner_q: dict[int, int] = {}

        def takeover_line(body: str) -> str:
            return with_prefix_then_trunc(random.choice(CONT_TAKEOVER), body, self.cfg.exec_cap)

        if mode == "A":
            # actor finished q_actor in seg3; third finished q_third in seg3; actor will finish q_star
            owner_q[q_actor] = ch_actor
            owner_q[q_third] = ch_third
            owner_q[q_star] = ch_actor

            # Seg4: actor does paused question part2 (takeover prefix)
            seg4_text = ["", "", ""]
            seg4_text[ch_actor] = takeover_line((parts[q_star][1] or "").strip())
            seg4_text[ch_pause] = idle_line()
            seg4_text[ch_third] = idle_line()
            seg4 = [mkmsg(0, seg4_text[0]), mkmsg(1, seg4_text[1]), mkmsg(2, seg4_text[2])]
            segments.append(seg4)
            accrue(seg4)

            # Seg5: actor does paused question part3 (takeover prefix)
            seg5_text = ["", "", ""]
            seg5_text[ch_actor] = takeover_line((parts[q_star][2] or "").strip())
            seg5_text[ch_pause] = idle_line()
            seg5_text[ch_third] = idle_line()
            seg5 = [mkmsg(0, seg5_text[0]), mkmsg(1, seg5_text[1]), mkmsg(2, seg5_text[2])]
            segments.append(seg5)
            accrue(seg5)

        elif mode == "B":
            # swap ownership:
            # - actor finishes paused q_star
            # - paused channel finishes actor's question q_actor
            owner_q[q_third] = ch_third
            owner_q[q_star] = ch_actor
            owner_q[q_actor] = ch_pause

            # Seg4: paused channel finishes q_actor part3; actor does q_star part2 (takeover prefix)
            seg4_text = ["", "", ""]
            seg4_text[ch_pause] = with_prefix_then_trunc(
                random.choice(CONT_MY),
                (parts[q_actor][2] or "").strip(),
                self.cfg.exec_cap,
            )
            seg4_text[ch_actor] = takeover_line((parts[q_star][1] or "").strip())
            seg4_text[ch_third] = idle_line()
            seg4 = [mkmsg(0, seg4_text[0]), mkmsg(1, seg4_text[1]), mkmsg(2, seg4_text[2])]
            segments.append(seg4)
            accrue(seg4)

            # Seg5: actor finishes q_star part3 (takeover prefix)
            seg5_text = ["", "", ""]
            seg5_text[ch_actor] = takeover_line((parts[q_star][2] or "").strip())
            seg5_text[ch_pause] = idle_line()
            seg5_text[ch_third] = idle_line()
            seg5 = [mkmsg(0, seg5_text[0]), mkmsg(1, seg5_text[1]), mkmsg(2, seg5_text[2])]
            segments.append(seg5)
            accrue(seg5)

        else:  # mode == "C"
            # postpone:
            # - actor finishes paused q_star first
            # - actor completes q_actor later
            owner_q[q_third] = ch_third
            owner_q[q_star] = ch_actor
            owner_q[q_actor] = ch_actor

            # Seg4: actor does q_star part2 (takeover prefix)
            seg4_text = ["", "", ""]
            seg4_text[ch_actor] = takeover_line((parts[q_star][1] or "").strip())
            seg4_text[ch_pause] = idle_line()
            seg4_text[ch_third] = idle_line()
            seg4 = [mkmsg(0, seg4_text[0]), mkmsg(1, seg4_text[1]), mkmsg(2, seg4_text[2])]
            segments.append(seg4)
            accrue(seg4)

            # Seg5: actor does q_star part3 (takeover prefix)
            seg5_text = ["", "", ""]
            seg5_text[ch_actor] = takeover_line((parts[q_star][2] or "").strip())
            seg5_text[ch_pause] = idle_line()
            seg5_text[ch_third] = idle_line()
            seg5 = [mkmsg(0, seg5_text[0]), mkmsg(1, seg5_text[1]), mkmsg(2, seg5_text[2])]
            segments.append(seg5)
            accrue(seg5)

            # Seg6: actor completes q_actor part3 (natural "Continuing:" prefix)
            seg6_text = ["", "", ""]
            seg6_text[ch_actor] = with_prefix_then_trunc(
                random.choice(CONT_MY),
                (parts[q_actor][2] or "").strip(),
                self.cfg.exec_cap,
            )
            seg6_text[ch_pause] = idle_line()
            seg6_text[ch_third] = idle_line()
            seg6 = [mkmsg(0, seg6_text[0]), mkmsg(1, seg6_text[1]), mkmsg(2, seg6_text[2])]
            segments.append(seg6)
            accrue(seg6)

        # ---------------- per-channel summaries (ALL to:all)
        final_ans = [extract_boxed_or_short(full_sol[i]) for i in range(3)]
        per_ch_items: dict[int, list[str]] = {0: [], 1: [], 2: []}

        for qi in range(3):
            ch_owner = owner_q.get(qi, qi)  # fallback to original owner
            per_ch_items[ch_owner].append(f"Q{qi + 1}={final_ans[qi]}")

        sum_seg: list[dict[str, Any]] = []
        for ch in [0, 1, 2]:
            if per_ch_items[ch]:
                sum_seg.append(mkmsg(ch, wrap_to_all("My answers: " + ", ".join(per_ch_items[ch]) + ".")))
            else:
                sum_seg.append(mkmsg(ch, wrap_to_all("I did not solve a question in this run.")))
        segments.append(sum_seg)
        accrue(sum_seg)

        # ---------------- global summary (3 msgs; summarizer emits to:all)
        summarizer = max(cost.keys(), key=lambda c: cost[c])
        gseg_text = ["", "", ""]
        for ch in [0, 1, 2]:
            if ch == summarizer:
                gseg_text[ch] = wrap_to_all("Final summary:\n" + "\n".join([f"Question {i + 1}: {final_ans[i]}" for i in range(3)]))
            else:
                gseg_text[ch] = "Done."
        gseg = [mkmsg(0, gseg_text[0]), mkmsg(1, gseg_text[1]), mkmsg(2, gseg_text[2])]
        segments.append(gseg)
        accrue(gseg)

        out = {
            "problem": formatted_problem,
            "channels": segments,
            "expected_response": [it.get("expected_response", "") for it in items3],
            "sample_id": f"interact_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
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
            },
        }
        return out


# ------------------------ I/O ------------------------


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, items: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Build interaction dataset (3Q/3C) with modes A/B/C. Naturalized transitions; strict pause; empty idle; supports forcing exact N samples."
    )
    ap.add_argument("--input", type=str, required=True, help="Input JSONL (per-question items)")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL (aggregated samples)")
    ap.add_argument("--mode", type=str, default="auto", choices=["auto", "A", "B", "C"])
    ap.add_argument("--tokenizer", type=str, default=None, help="HF tokenizer path for accurate token counting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--drop_last", action="store_true")
    ap.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Force exact number of output samples (sample 3 questions each time; replacement only if input<3)",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    cfg = BuildConfig(
        tokenizer_path=args.tokenizer,
        mode=args.mode,
        seed=args.seed,
    )
    builder = InteractiveBuilder(cfg)

    items = load_jsonl(args.input)
    if args.limit is not None:
        items = items[: args.limit]
    if args.shuffle:
        random.shuffle(items)

    if not items:
        raise SystemExit("ERROR: input JSONL is empty.")

    outs: list[dict[str, Any]] = []

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
            batch = items[i : i + 3]
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
    if args.num_samples is not None:
        print(f"Forced N       : {args.num_samples}")


if __name__ == "__main__":
    main()
