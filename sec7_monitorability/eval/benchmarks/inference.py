"""Re-exports from the canonical stream_inference module + GenerationResult.

`StreamModel`, `StreamResult`, and the generation helpers live in
`eval/stream_inference.py`; this file re-exports them so existing imports
like `from .inference import StreamModel` keep working.

`GenerationResult` is the new model-class-neutral result type used by the
scoring code. Both the stream runner (via `from_stream_result`) and the
chat runner (via `generate_chat_result`) produce `GenerationResult`
instances, so every `score_response` / `score_pair` takes the same shape
regardless of whether the underlying model is stream, base with thinking,
or chat without thinking.

Field conventions:
  - `raw`: the full decoded generation, NEVER stripped. This is the
    audit-invariant — anything downstream that wants to rescore with a
    different judge / different parser should only need `raw`.
  - `verbalized_response`: what the user of the model would see. For
    stream, that's the Output channel. For base models with thinking,
    that's `raw` with the `<think>...</think>` block removed. For chat
    baselines without thinking, that's equal to `raw`.
  - `hidden_reasoning`: dict of {channel_name → text} containing everything
    NOT in `verbalized_response`. For stream, the 8 non-User/non-Output
    channels. For base with thinking, `{"CoT": <think-block>}`. For chat
    baseline, `{}`. Iterate `hidden_reasoning.items()` in scoring code —
    the shape is uniform.
  - `tokens` and `silence_token`: stream-only; `None` for chat/base.
"""

from dataclasses import dataclass, field

# Dual-import shim: when this module is imported as `eval.benchmarks.inference`
# (via `python -m eval.benchmarks.run_eval`), the relative path resolves.
# When `run_chat_benchmarks.py` inserts `eval/` onto sys.path and imports it
# as top-level `benchmarks.inference`, the relative path escapes the package —
# fall back to the absolute `stream_inference` module which sys.path exposes.
try:
    from ..stream_inference import (  # noqa: F401
        CHANNELS,
        SEED_WORDS,
        C,
        StreamModel,
        StreamResult,
        collect_result,
        generate,
        generate_choice,
        generate_to_result,
        load_model,
        sample_top_p,
    )
except ImportError:
    from stream_inference import (  # noqa: F401
        CHANNELS,
        SEED_WORDS,
        C,
        StreamModel,
        StreamResult,
        collect_result,
        generate,
        generate_choice,
        generate_to_result,
        load_model,
        sample_top_p,
    )


@dataclass
class GenerationResult:
    """Model-class-neutral result of a single generation.

    See module docstring for field conventions. Construct via
    `from_stream_result` (stream runner) or `from_chat_text` (chat runner).
    """

    prompt: str
    system_prompt: str | None
    raw: str
    verbalized_response: str
    hidden_reasoning: dict[str, str] = field(default_factory=dict)
    tokens: list[list[int]] | None = None
    silence_token: int | None = None
    parsing_format: str | None = None

    @property

    def to_dict(self) -> dict:
        """Serialize to the canonical JSON shape for saving in results_log."""
        d = {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "raw": self.raw,
            "verbalized_response": self.verbalized_response,
            "hidden_reasoning": self.hidden_reasoning,
        }
        if self.parsing_format is not None:
            d["parsing_format"] = self.parsing_format
        # tokens skipped from JSON — they're large and stream-specific.
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationResult":
        """Reconstruct from a saved JSON entry (new schema).

        Use `from_legacy_entry` to read the pre-refactor flat field shape.
        """
        return cls(
            prompt=d.get("prompt", ""),
            system_prompt=d.get("system_prompt"),
            raw=d.get("raw", d.get("verbalized_response", "")),
            verbalized_response=d.get("verbalized_response", ""),
            hidden_reasoning=d.get("hidden_reasoning") or {},
            parsing_format=d.get("parsing_format"),
        )

    @classmethod

    @classmethod
    def from_stream_result(
        cls,
        stream_result: "StreamResult",
        prompt: str,
        system_prompt: str | None = None,
    ) -> "GenerationResult":
        """Adapt a StreamResult from the stream runner into a GenerationResult.

        `hidden_reasoning` = all channels except User and Output.
        `verbalized_response` = the Output channel.
        `raw` = concatenation of all channel texts with channel headers, so
        rescoring code that wants a flat "everything" view can have one.
        """
        channel_texts = dict(stream_result.channel_texts)
        verbalized = channel_texts.get("Output", "")
        hidden = {name: text for name, text in channel_texts.items() if name not in ("User", "Output") and text.strip()}
        # Synthesised raw — label each channel so an auditor can tell who said what
        raw_parts = []
        for name, text in channel_texts.items():
            if not text.strip():
                continue
            raw_parts.append(f"[{name}]\n{text.strip()}")
        raw = "\n\n".join(raw_parts)
        return cls(
            prompt=prompt,
            system_prompt=system_prompt,
            raw=raw,
            verbalized_response=verbalized,
            hidden_reasoning=hidden,
            tokens=stream_result.tokens,
            silence_token=stream_result.silence_token,
        )

    @classmethod
    def from_chat_text(
        cls,
        raw: str,
        prompt: str,
        system_prompt: str | None = None,
        thinking_enabled: bool = False,
    ) -> "GenerationResult":
        """Construct from a raw decoded chat generation, extracting CoT if any.

        Uses the same <think> splitting logic as generate_chat_with_cot — but
        crucially retains the untouched `raw` field so nothing is thrown
        away. The legacy `generate_chat` / `generate_chat_with_cot` helpers
        strip and discard the raw; this one does not.

        The `parsing_format` field tags which case fired so downstream code
        knows what shape was detected. Possible values:
          - "closed_tags"        — both <think> and </think> in raw
          - "preseeded_close"    — only </think> (chat template seeded <think>)
          - "truncated_thinking" — <think> with no </think> (ran out of tokens)
          - "structured_natural" — Qwen3.5-style "Thinking Process:" prefix
          - "no_thinking"        — no thinking signal detected
          - None                 — thinking_enabled=False (chat baseline)
        """
        import re as _re

        parsing_format: str | None = None
        cot = ""
        verbalized = ""
        truncated = False

        if thinking_enabled:
            has_open = "<think>" in raw
            has_close = "</think>" in raw

            # Case 1: both <think> and </think> in new_tokens
            if has_open and has_close:
                m = _re.search(r"<think>(.*?)</think>", raw, flags=_re.DOTALL)
                cot = m.group(1).strip() if m else ""
                verbalized = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
                parsing_format = "closed_tags"
            # Case 2: only </think> (template pre-seeded the opening tag)
            elif has_close and not has_open:
                cot, _, verbalized = raw.partition("</think>")
                cot = cot.strip()
                verbalized = verbalized.strip()
                parsing_format = "preseeded_close"
            # Case 2.5: <think> opened but never closed — model ran out of
            # --max-new-tokens still inside the thinking block. Everything
            # after <think> is CoT; there is no visible response yet.
            elif has_open and not has_close:
                _, _, after_open = raw.partition("<think>")
                cot = after_open.lstrip("\n").rstrip()
                verbalized = ""
                truncated = True
                parsing_format = "truncated_thinking"
            # Case 2.6: Qwen3.5-style "Thinking Process:" structured natural
            # language thinking with no <think> tags. We recognize the explicit
            # "Thinking Process:" prefix (case-sensitive, at start, optional
            # leading whitespace). The transition from CoT to the actual
            # response is marked by a literal "*Draft:*" / "**Draft:**" line.
            # If no Draft marker is found, treat the entire raw as CoT
            # (response was truncated before draft).
            elif _re.match(r"\s*Thinking Process:", raw):
                # Find a Draft marker — search for the last occurrence so
                # nested "*Draft:*" inside the cot (rare) is handled
                # gracefully by taking the final draft as the response start.
                draft_match = None
                for pat in (r"\*\*Draft:\*\*", r"\*Draft:\*"):
                    matches = list(_re.finditer(pat, raw))
                    if matches:
                        draft_match = matches[-1]
                        break
                if draft_match is not None:
                    cot = raw[: draft_match.start()].rstrip()
                    after_draft = raw[draft_match.end() :]
                    verbalized = after_draft.strip()
                else:
                    # No Draft marker — assume truncated mid-thinking
                    cot = raw.strip()
                    verbalized = ""
                    truncated = True
                parsing_format = "structured_natural"
            # Case 3: model chose not to use thinking tags — keep full
            # output as verbalized, don't drop anything
            else:
                cot = ""
                verbalized = raw.strip()
                parsing_format = "no_thinking"

            if cot:
                hidden = {"CoT": cot}
                if truncated:
                    hidden["truncated_thinking"] = "true"
            else:
                hidden = {}
        else:
            # Chat baseline without thinking: raw IS verbalized
            verbalized = raw.strip()
            hidden = {}

        return cls(
            prompt=prompt,
            system_prompt=system_prompt,
            raw=raw,
            verbalized_response=verbalized,
            hidden_reasoning=hidden,
            parsing_format=parsing_format,
        )


_STREAM_THINKING_CHANNEL_NAMES = (
    "Analytical",
    "Skeptical",
    "Intuitive",
    "Between",
    "Curious",
    "Void",
    "Instinct",
    "Synthesis",
)


def _split_chat_allstreams_to_channels(cot_text: str) -> dict[str, str]:
    """Parse a chat-allstreams CoT blob into {channel_name: text} dict.

    Recognises three formatting styles, in priority order:
      1. XML-tag wrapping: `<Analytical>...</Analytical>` (or any case)
      2. Line-prefix labels: `Analytical: ...` (the form chat_8b_allstreams
         emits in practice — see v5 results items 0/5/10/20/30 inspected
         2026-05-02).
      3. Bracket-prefix labels: `[Analytical]\\n...` (the form `_concat_thinking`
         and the stream `raw` use — defensive fallback).

    Returns a dict mapping channel_name → text. Channels not detected in
    the input are absent from the dict (NOT empty strings) so callers can
    distinguish "this channel had no text" from "this CoT was not in
    stream-shape at all".

    Returns an empty dict if no recognised pattern is present, signalling
    the caller should fall back to the original {"CoT": ...} representation.
    """
    import re as _re

    if not cot_text or not cot_text.strip():
        return {}

    names_alt = "|".join(_STREAM_THINKING_CHANNEL_NAMES)
    out: dict[str, str] = {}

    # 1. XML tags: <Analytical>...</Analytical> (case-insensitive).
    xml_pat = _re.compile(
        rf"<({names_alt})>(.*?)</\1>",
        flags=_re.IGNORECASE | _re.DOTALL,
    )
    for m in xml_pat.finditer(cot_text):
        # Canonicalise the channel name to the registered casing so the
        # judge sees the same string the stream runner emits.
        raw_name = m.group(1)
        canonical = next(
            (n for n in _STREAM_THINKING_CHANNEL_NAMES if n.lower() == raw_name.lower()),
            raw_name,
        )
        text = m.group(2).strip()
        if text:
            # Concatenate if the same name appears multiple times (rare).
            if canonical in out:
                out[canonical] = out[canonical] + "\n\n" + text
            else:
                out[canonical] = text
    if out:
        return out

    # 2. Line-prefix labels: `^Analytical:` (with optional leading whitespace).
    # Split the cot at every channel-name marker, keeping the text that
    # follows each marker until the next marker.
    prefix_pat = _re.compile(
        rf"(?:^|\n)\s*({names_alt})\s*:\s*",
        flags=_re.IGNORECASE,
    )
    matches = list(prefix_pat.finditer(cot_text))
    if matches:
        for i, m in enumerate(matches):
            raw_name = m.group(1)
            canonical = next(
                (n for n in _STREAM_THINKING_CHANNEL_NAMES if n.lower() == raw_name.lower()),
                raw_name,
            )
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cot_text)
            text = cot_text[start:end].strip()
            if text:
                if canonical in out:
                    out[canonical] = out[canonical] + "\n\n" + text
                else:
                    out[canonical] = text
        if out:
            return out

    # 3. Bracket-prefix labels: `[Analytical]\n...`.
    bracket_pat = _re.compile(
        rf"(?:^|\n)\s*\[({names_alt})\]\s*\n",
        flags=_re.IGNORECASE,
    )
    matches = list(bracket_pat.finditer(cot_text))
    if matches:
        for i, m in enumerate(matches):
            raw_name = m.group(1)
            canonical = next(
                (n for n in _STREAM_THINKING_CHANNEL_NAMES if n.lower() == raw_name.lower()),
                raw_name,
            )
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cot_text)
            text = cot_text[start:end].strip()
            if text:
                if canonical in out:
                    out[canonical] = out[canonical] + "\n\n" + text
                else:
                    out[canonical] = text

    return out


def reshape_hidden_reasoning_for_judge(
    hidden: dict[str, str] | None,
) -> dict[str, str]:
    """Presentation-normalisation step before handing thinking to a judge.

    If `hidden` already has multi-channel shape (more than one non-CoT
    key, or any of the recognised stream channel names as keys), pass
    through unchanged. Otherwise, if the only key is `CoT` (chat baseline
    with thinking enabled) AND the CoT body contains recognisable
    stream-channel structure (XML tags, line-prefix labels, or bracket
    labels), split the CoT into its constituent channels so the judge
    sees the same shape it sees from a stream model.

    This neutralises the format asymmetry flagged in v5 audit §A1 between
    chat_allstreams (single CoT blob with internal channel structure) and
    stream models (8 separately-tagged blocks).
    """
    if not hidden:
        return {}
    # Already multi-channel? Pass through.
    stream_keys = {n.lower() for n in _STREAM_THINKING_CHANNEL_NAMES}
    if any(k.lower() in stream_keys for k in hidden.keys()):
        return dict(hidden)
    # Only-CoT path: try to split.
    cot = hidden.get("CoT")
    if not isinstance(cot, str) or not cot.strip():
        return dict(hidden)
    split = _split_chat_allstreams_to_channels(cot)
    if not split:
        # No recognisable stream structure — leave as-is.
        return dict(hidden)
    # Carry through any non-CoT auxiliary keys (e.g. `truncated_thinking`).
    out = dict(split)
    for k, v in hidden.items():
        if k == "CoT":
            continue
        out.setdefault(k, v)
    return out
