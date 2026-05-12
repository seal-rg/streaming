# #!/usr/bin/env python3

# import argparse
# import glob
# import json
# import re
# from pathlib import Path
# from typing import Any, Dict, Optional, Tuple, List


# # ============================================================
# # Regex helpers
# # ============================================================

# _RE_FRACTION = re.compile(r"(?<!\d)(-?\d+)\s*/\s*(-?\d+)(?!\d)")
# _RE_INT = re.compile(r"\b-?\d+\b")


# def _gcd(x: int, y: int) -> int:
#     while y:
#         x, y = y, x % y
#     return x


# # ============================================================
# # Text cleaning
# # ============================================================

# def _strip_tags(s: str) -> str:
#     s = re.sub(r"<\s*/?\s*(Solver|AltSolver|Auditor)\s*>", " ", s, flags=re.IGNORECASE)
#     s = re.sub(r"<\s*(EOS|EOT|END)\s*>", " ", s, flags=re.IGNORECASE)
#     s = s.replace("GLOBAL_READY", " ")
#     s = re.sub(r"<\|im_start\|>|<\|im_end\|>", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s


# # ============================================================
# # Extract final answer (with method)
# # ============================================================

# def _extract_after_markers(text: str) -> Tuple[Optional[str], Optional[str]]:
#     markers = [
#         r"therefore[,:\s]+the answer is\s+",
#         r"final answer\s*[:：]\s*",
#         r"answer\s*[:：]\s*",
#         r"the answer is\s+",
#     ]

#     best = None
#     best_pos = -1
#     for m in markers:
#         for mo in re.finditer(m, text, flags=re.IGNORECASE):
#             if mo.start() >= best_pos:
#                 best_pos = mo.start()
#                 best = mo.end()

#     if best is None:
#         return None, None

#     tail = text[best:].strip()
#     tail = re.split(r"\n|<END>|<EOT>|</", tail)[0].strip()
#     return tail, "marker"


# def _extract_boxed(text: str) -> Tuple[Optional[str], Optional[str]]:
#     m = re.search(r"\\boxed\s*\{([^}]*)\}", text)
#     if m:
#         return m.group(1).strip(), "boxed"
#     return None, None


# def _extract_last_line(text: str) -> Tuple[Optional[str], Optional[str]]:
#     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
#     if not lines:
#         return None, None
#     return lines[-1], "lastline"


# def extract_final_answer_with_method(raw_text: Any) -> Tuple[Optional[str], Optional[str]]:
#     if not isinstance(raw_text, str) or not raw_text.strip():
#         return None, None

#     cleaned = _strip_tags(raw_text)

#     a, m = _extract_after_markers(cleaned)
#     if a:
#         return a, m

#     a, m = _extract_boxed(cleaned)
#     if a:
#         return a, m

#     a, m = _extract_last_line(cleaned)
#     return a, m


# # ============================================================
# # Normalization (核心逻辑)
# # ============================================================

# def normalize_answer(ans_raw: Optional[str]) -> Optional[str]:
#     if not ans_raw:
#         return None

#     a = _strip_tags(ans_raw).lower()
#     a = a.replace("**", " ")
#     a = re.sub(r"\$+", " ", a)
#     a = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", a)
#     a = re.sub(r"\s+", " ", a).strip()

#     # 去掉常见前缀
#     a = re.sub(r"^(the\s+)?final\s+answer\s+is\s+", "", a)
#     a = re.sub(r"^(the\s+)?answer\s+is\s+", "", a)
#     a = re.sub(r"^answer\s*[:：]\s*", "", a)

#     # 优先取 boxed 内容
#     m = re.search(r"\\boxed\s*\{([^}]*)\}", a)
#     if m:
#         a = m.group(1).strip()

#     a = a.strip()

#     # 去尾部标点和右括号
#     a = re.sub(r"[)\]}]+$", "", a)
#     a = re.sub(r"[.,;:!?]+$", "", a)
#     a = a.strip()

#     # =========================
#     # 选择题统一 (B) / B / (b)
#     # =========================
#     mopt = re.fullmatch(r"\(?\s*([a-z])\s*\)?", a)
#     if mopt:
#         return mopt.group(1)

#     # =========================
#     # 分数
#     # =========================
#     fm = _RE_FRACTION.search(a)
#     if fm:
#         num = int(fm.group(1))
#         den = int(fm.group(2))
#         if den != 0:
#             g = _gcd(abs(num), abs(den))
#             num //= g
#             den //= g
#             if den < 0:
#                 num = -num
#                 den = -den
#             return f"{num}/{den}"
#         return f"{num}/0"

#     # =========================
#     # 整数
#     # =========================
#     ints = _RE_INT.findall(a)
#     if len(ints) == 1:
#         return str(int(ints[0]))

#     return a[:220]


# # ============================================================
# # Input expansion
# # ============================================================

# def expand_inputs(in_jsonl, in_glob, in_dir):
#     paths = []
#     for p in in_jsonl:
#         paths.append(Path(p))
#     for pat in in_glob:
#         for m in glob.glob(pat, recursive=True):
#             paths.append(Path(m))
#     for d in in_dir:
#         dd = Path(d)
#         if dd.is_dir():
#             paths.extend(sorted(dd.rglob("*.jsonl")))

#     uniq = []
#     seen = set()
#     for p in paths:
#         rp = str(p.resolve())
#         if rp not in seen and p.exists():
#             uniq.append(p)
#             seen.add(rp)
#     return uniq


# # ============================================================
# # Main
# # ============================================================

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--in_jsonl", nargs="*", default=[])
#     ap.add_argument("--in_glob", nargs="*", default=[])
#     ap.add_argument("--in_dir", nargs="*", default=[])
#     ap.add_argument("--out_jsonl", required=True)
#     ap.add_argument("--debug_jsonl", required=True)

#     ap.add_argument("--deepseek_field", default="deepseek_thinking_trajectory_sequential")
#     ap.add_argument("--gemini_field", default="gemini_attempt")

#     args = ap.parse_args()

#     in_files = expand_inputs(args.in_jsonl, args.in_glob, args.in_dir)
#     if not in_files:
#         raise SystemExit("No input files found.")

#     kept = 0
#     total = 0

#     with open(args.out_jsonl, "w", encoding="utf-8") as fout, \
#          open(args.debug_jsonl, "w", encoding="utf-8") as fdbg:

#         for fp in in_files:
#             with open(fp, "r", encoding="utf-8") as fin:
#                 for line_no, line in enumerate(fin, 1):
#                     line = line.strip()
#                     if not line:
#                         continue
#                     total += 1

#                     obj = json.loads(line)

#                     # ---------- Teacher ----------
#                     t_raw, t_method = extract_final_answer_with_method(obj.get("teacher_output", ""))
#                     t_norm = normalize_answer(t_raw)

#                     # ---------- DeepSeek ----------
#                     ds_text = obj.get(args.deepseek_field, "")
#                     ds_raw, ds_method = extract_final_answer_with_method(ds_text)
#                     ds_norm = normalize_answer(ds_raw)

#                     # ---------- Gemini ----------
#                     gm_text = obj.get(args.gemini_field, "")
#                     gm_raw, gm_method = extract_final_answer_with_method(gm_text)
#                     gm_norm = normalize_answer(gm_raw)

#                     match_ds = bool(t_norm and ds_norm and t_norm == ds_norm)
#                     match_gm = bool(t_norm and gm_norm and t_norm == gm_norm)
#                     keep = match_ds or match_gm

#                     debug = {
#                         "src_file": str(fp),
#                         "src_line": line_no,
#                         "id": obj.get("id"),

#                         "teacher_method": t_method,
#                         "teacher_ans_raw": t_raw,
#                         "teacher_norm": t_norm,

#                         "deepseek_method": ds_method,
#                         "deepseek_ans_raw": ds_raw,
#                         "deepseek_norm": ds_norm,
#                         "match_deepseek": match_ds,

#                         "gemini_method": gm_method,
#                         "gemini_ans_raw": gm_raw,
#                         "gemini_norm": gm_norm,
#                         "match_gemini": match_gm,

#                         "keep": keep
#                     }

#                     fdbg.write(json.dumps(debug, ensure_ascii=False) + "\n")

#                     if keep:
#                         fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
#                         kept += 1

#     print(f"[DONE] total={total}, kept={kept}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any

# ===============================
# Clean utils
# ===============================

TAG_RE = re.compile(r"<EOQ>|<EOS>|<EOT>", re.IGNORECASE)


def clean_text_single_line(s: Any) -> str:
    """
    - remove <EOQ>/<EOS>/<EOT>
    - collapse all whitespace (including newlines) into single spaces
    """
    if not isinstance(s, str) or not s:
        return ""
    s = TAG_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_leading_index(s: str) -> str:
    """Remove leading indices like '13. '."""
    return re.sub(r"^\d+\.\s*", "", s.strip())


# ===============================
# Question / Context extraction
# ===============================


def extract_question_generic(rec: dict[str, Any]) -> str:
    """
    Try multiple fields in priority.
    """
    for k in ("question", "raw_context"):
        q = clean_text_single_line(rec.get(k, ""))
        if q:
            return q
    # fallback
    raise ValueError(f"Missing question fields (id={rec.get('id')})")


def extract_context_generic(rec: dict[str, Any]) -> str:
    """
    - If context_eos exists, split by <EOS> into lines, remove header "Context",
      strip leading indices, and keep one sentence per line.
    - Otherwise fallback to empty.
    """
    ctx = rec.get("context_eos", "")
    if not isinstance(ctx, str) or not ctx.strip():
        # fallback: sometimes context == question
        ctx2 = rec.get("context", "")
        ctx2 = ctx2 if isinstance(ctx2, str) else ""
        return ctx2.strip()

    lines: list[str] = []
    for part in ctx.split("<EOS>"):
        part = clean_text_single_line(part)
        if not part:
            continue

        if part.lower().startswith("context"):
            part = re.sub(r"^context\s*:?\s*", "", part, flags=re.IGNORECASE).strip()
            if not part:
                continue

        part = strip_leading_index(part)
        if part:
            lines.append(part)

    return "\n".join(lines)


# ===============================
# Output cleaning + Persona parsing
# ===============================


def cut_before_first_end(raw: str) -> str:
    """Keep only content before the first <END> (exclusive)."""
    if not isinstance(raw, str):
        return ""
    idx = raw.find("<END>")
    if idx == -1:
        return raw.strip()
    return raw[:idx].strip()


def clean_output_keep_lines(raw: Any) -> str:
    """
    - remove <EOQ>/<EOS>/<EOT> but keep newlines
    - rstrip each line
    """
    if not isinstance(raw, str) or not raw:
        return ""
    raw = TAG_RE.sub("", raw)
    raw = "\n".join(line.rstrip() for line in raw.splitlines())
    return raw.strip()


def _canon_persona(name: str) -> str:
    name = name.strip().lower()
    if name == "solver":
        return "Solver"
    if name == "altsolver":
        return "AltSolver"
    if name == "auditor":
        return "Auditor"
    return name


# Supports:
#   "<Solver>" alone on a line
#   "<Solver> content..." on the same line
PERSONA_TAG_RE = re.compile(r"^\s*<\s*(Solver|AltSolver|Auditor)\s*>\s*(.*)$", re.IGNORECASE)


def parse_personas(teacher_output_cleaned: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Parse persona blocks from teacher output.

    Rules:
      - persona tag appears at line start: <Solver>/<AltSolver>/<Auditor>
      - content may be on same line or following lines until next persona tag
      - skip numeric-only lines
      - skip GLOBAL_READY lines (robust)
      - keep intra-segment blank lines
      - optionally insert TRANSITION_SENTENCE at beginning of last Solver segment
    """
    turns: dict[str, list[str]] = {"Solver": [], "AltSolver": [], "Auditor": []}
    cur: str = ""
    buf: list[str] = []

    TRANSITION_SENTENCE = (
        "Based on the step-by-step reasoning and verification above, we can now summarize the key points and reach a conclusion."
    )

    def flush():
        nonlocal cur, buf
        if cur:
            text = "\n".join(buf).strip()
            if text:
                turns[cur].append(text)
        buf = []

    for line in (teacher_output_cleaned or "").splitlines():
        line_stripped = line.strip()

        # skip GLOBAL_READY (robust)
        if re.fullmatch(r"GLOBAL_READY", line_stripped, flags=re.IGNORECASE):
            continue

        # persona tag (supports same-line content)
        m = PERSONA_TAG_RE.match(line_stripped)
        if m:
            flush()
            cur = _canon_persona(m.group(1))
            rest = (m.group(2) or "").rstrip()
            if rest:
                buf.append(rest)
            continue

        # preserve blank lines inside a persona block
        if not line_stripped:
            if cur and (not buf or buf[-1] != ""):
                buf.append("")
            continue

        # skip numeric-only line (segment index)
        if re.fullmatch(r"\d+", line_stripped):
            continue

        if cur:
            buf.append(line.rstrip())

    flush()

    # Insert transition sentence at BEGINNING of last Solver segment
    if turns["Solver"]:
        last = turns["Solver"][-1].strip()
        if last and (TRANSITION_SENTENCE not in last):
            turns["Solver"][-1] = TRANSITION_SENTENCE + "\n\n" + last

    personas_joined: dict[str, str] = {k: ("\n\n".join(v).strip() if v else "") for k, v in turns.items()}
    return personas_joined, turns


# ===============================
# Gold extraction / normalization
# ===============================

_UNKNOWN_SET = {"", "unknown", "none", "null", "nan", "n/a"}


def _is_unknown(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (int, float)):
        return False
    if not isinstance(x, str):
        x = str(x)
    return x.strip().lower() in _UNKNOWN_SET


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


# Common patterns near the end
RE_ANSWER_COLON = re.compile(r"(?:^|\n)\s*Answer\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_FINAL_ANSWER_COLON = re.compile(r"(?:^|\n)\s*Final\s*Answer\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_CORRECT_ANSWER_COLON = re.compile(r"(?:^|\n)\s*Correct\s*Answer\s*:\s*(.+?)\s*$", re.IGNORECASE)

# "answer is ..." family
RE_ANSWER_IS = re.compile(
    r"(?:^|\n)\s*(?:Therefore,?\s*)?"
    r"(?:(?:the\s*)?answer|final\s*answer|correct\s*answer)\s*(?:is|=)\s*(.+?)\s*\.?\s*$",
    re.IGNORECASE,
)

# LaTeX boxed
RE_BOXED = re.compile(r"\\boxed\s*\{([^}]+)\}", re.IGNORECASE)

# Choice-style (A/B/C/D)
RE_CHOICE_1 = re.compile(r"(?:^|\n)\s*Answer\s*:\s*\(?\s*([A-D])\s*\)?\s*$", re.IGNORECASE)
RE_CHOICE_2 = re.compile(r"(?:^|\n)\s*(?:Correct\s*)?(?:choice|option)\s*(?:is|:)\s*\(?\s*([A-D])\s*\)?\s*\.?\s*$", re.IGNORECASE)
RE_CHOICE_3 = re.compile(r"(?:^|\n)\s*Thus,?\s*(?:the\s*)?(?:answer|correct\s*choice)\s*is\s*\(?\s*([A-D])\s*\)?\s*\.?\s*$", re.IGNORECASE)

# Sometimes they end with just: "(A) 90954 m" or "A" on the last line.
# We'll do a conservative last-line heuristic only if it looks like a clean choice marker.
RE_LASTLINE_CHOICE = re.compile(r"^\(?\s*([A-D])\s*\)?(?:\s*[:\-].*)?$", re.IGNORECASE)


def _clean_answer_text(ans: str) -> str:
    ans = _as_str(ans).strip()
    ans = TAG_RE.sub("", ans).strip()
    # strip trailing <END> if accidentally present
    ans = re.sub(r"\s*<END>\s*$", "", ans, flags=re.IGNORECASE).strip()
    # strip common "Answer:" remnants
    ans = re.sub(r"^\s*(?:Answer|Final\s*Answer|Correct\s*Answer)\s*[:=]\s*", "", ans, flags=re.IGNORECASE).strip()
    return ans


def extract_gold_from_teacher_output(teacher_clean: str) -> str:
    """
    teacher_clean should already be: clean_output_keep_lines + cut_before_first_end
    Extract answer from the tail region to avoid mid-body false matches.
    """
    if not teacher_clean:
        return ""

    tail_lines = teacher_clean.splitlines()[-60:]  # look at last ~60 lines
    tail = "\n".join(tail_lines).strip()
    if not tail:
        return ""

    # 1) boxed
    m = RE_BOXED.search(tail)
    if m:
        return _clean_answer_text(m.group(1))

    # 2) explicit choice lines
    for rgx in (RE_CHOICE_1, RE_CHOICE_2, RE_CHOICE_3):
        m = rgx.search(tail)
        if m:
            return _clean_answer_text(m.group(1).upper())

    # 3) colon styles
    for rgx in (RE_FINAL_ANSWER_COLON, RE_CORRECT_ANSWER_COLON, RE_ANSWER_COLON):
        m = rgx.search(tail)
        if m:
            return _clean_answer_text(m.group(1))

    # 4) "answer is ..." family
    m = RE_ANSWER_IS.search(tail)
    if m:
        return _clean_answer_text(m.group(1))

    # 5) fallback: last non-empty line if it looks like a clean choice marker
    for line in reversed(tail_lines):
        ls = line.strip()
        if not ls:
            continue
        m = RE_LASTLINE_CHOICE.match(ls)
        if m:
            return _clean_answer_text(m.group(1).upper())
        break

    return ""


def resolve_gold(rec: dict[str, Any], teacher_clean: str) -> str:
    """
    Gold resolution priority:
      1) gold
      2) answer
      3) solution (if exists and looks short-ish)
      4) solution_orig
      5) infer from teacher output tail
      else "Unknown"
    """
    # 1) gold
    gold = rec.get("gold", None)
    if not _is_unknown(gold):
        return _clean_answer_text(_as_str(gold))

    # 2) answer
    ans = rec.get("answer", None)
    if not _is_unknown(ans):
        return _clean_answer_text(_as_str(ans))

    # 3) solution (sometimes long; keep if short)
    sol = rec.get("solution", None)
    if isinstance(sol, str) and sol.strip():
        s = sol.strip()
        if len(s) <= 200:  # conservative
            return _clean_answer_text(s)

    # 4) solution_orig
    sol0 = rec.get("solution_orig", None)
    if isinstance(sol0, str) and sol0.strip():
        s0 = sol0.strip()
        if len(s0) <= 200:
            return _clean_answer_text(s0)

    # 5) infer from teacher output tail
    inferred = extract_gold_from_teacher_output(teacher_clean)
    if inferred:
        return _clean_answer_text(inferred)

    return "Unknown"


# ===============================
# Record conversion
# ===============================


def convert_one_record(rec: dict[str, Any], src_path: Path, line_no: int) -> dict[str, Any]:
    ds = (rec.get("dataset") or "").strip()

    # teacher output -> cleaned
    teacher_raw = rec.get("teacher_output", "")
    teacher_clean = clean_output_keep_lines(teacher_raw)
    teacher_clean = cut_before_first_end(teacher_clean)

    # parse personas
    personas, persona_turns = parse_personas(teacher_clean)

    # question/context (generic)
    question = extract_question_generic(rec)
    context = extract_context_generic(rec)

    # gold resolve
    gold = resolve_gold(rec, teacher_clean)

    new_rec = {
        "id": rec.get("id"),
        "dataset": ds,
        "split": rec.get("split"),
        "src_index": rec.get("src_index"),
        "gold": gold,
        "answer": gold,
        # trace info (useful when merging many files)
        "src_file": str(src_path.name),
        "src_line": line_no,
        # optional raw fields for traceability
        "raw_context": rec.get("raw_context"),
        "context_eos": rec.get("context_eos"),
        "teacher_output": rec.get("teacher_output"),
        # keep original solution fields if present (optional but helpful)
        "question_orig": rec.get("question_orig"),
        "solution_orig": rec.get("solution_orig"),
        "solution": rec.get("solution"),
        "metadata": rec.get("metadata"),
        # structured fields
        "question": question,
        "context": context,
        "output": teacher_clean,
        # persona parsing results
        "personas": personas,
        "persona_turns": persona_turns,
    }
    return new_rec


# ===============================
# Directory conversion: all kept.jsonl -> single output jsonl
# ===============================


def convert_directory(in_dir: str, out_path: str, recursive: bool = False) -> None:
    in_dir_p = Path(in_dir)
    if not in_dir_p.exists() or not in_dir_p.is_dir():
        raise ValueError(f"--in_dir must be an existing directory: {in_dir_p}")

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    if recursive:
        files = sorted([p for p in in_dir_p.rglob("*") if p.is_file() and p.name == "kept.jsonl"])
    else:
        files = sorted([p for p in in_dir_p.iterdir() if p.is_file() and p.name == "kept.jsonl"])

    if not files:
        raise ValueError(f"No kept.jsonl found in {in_dir_p} (recursive={recursive})")

    with out_path_p.open("w", encoding="utf-8") as fout:
        for file in files:
            with file.open("r", encoding="utf-8") as fin:
                for ln, line in enumerate(fin, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception as e:
                        raise RuntimeError(f"JSON parse error: file={file} line={ln}: {e}") from e

                    try:
                        new_rec = convert_one_record(rec, src_path=file, line_no=ln)
                    except Exception as e:
                        raise RuntimeError(f"Convert error: file={file} line={ln}: {e}") from e

                    fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")


# ===============================
# CLI
# ===============================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="input directory containing kept.jsonl files")
    ap.add_argument("--out", required=True, help="output merged jsonl file")
    ap.add_argument("--recursive", action="store_true", help="scan subdirectories recursively")
    args = ap.parse_args()

    convert_directory(args.in_dir, args.out, recursive=args.recursive)


if __name__ == "__main__":
    main()
