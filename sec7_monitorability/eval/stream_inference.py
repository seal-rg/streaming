"""Canonical multi-stream inference engine.

Single implementation for all consumers: demos, benchmarks, evaluation.

Usage (functional API — generator for streaming display):
    model, tokenizer, silence_token = load_model("/path/to/checkpoint")
    for row_idx, row, is_prefill in generate(model, tokenizer, "Hello", silence_token):
        print(row_idx, row)

Usage (class API — returns StreamResult):
    sm = StreamModel("/path/to/checkpoint")
    result = sm.generate("Hello")
    print(result.output)
    print(result.channel_texts["Skeptical"])

Usage (MCQ benchmarks):
    chosen_idx, result = sm.generate_choice("What is 2+2?", ["3", "4", "5"])
"""

import json
import os
import sys
from collections.abc import Generator
from dataclasses import dataclass

import torch

# Ensure train/ is importable for model classes
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "train"))
sys.path.insert(0, os.path.join(_root, "train", "train"))

# ── Constants ────────────────────────────────────────────────────────────────

CHANNELS = [
    "User",
    "Output",
    "Analytical",
    "Skeptical",
    "Intuitive",
    "Between",
    "Curious",
    "Void",
    "Instinct",
    "Synthesis",
]
C = len(CHANNELS)

# Seed tokens for row 0. User and Output start silent; channels 2-9 get a
# keyword describing their role (must be single tokens with leading space).
SEED_WORDS = [
    "-",
    "-",  # User, Output: silent
    " thinking",  # Analytical
    " checking",  # Skeptical
    " feeling",  # Intuitive
    " relating",  # Between
    " asking",  # Curious
    " drifting",  # Void
    " watching",  # Instinct
    " integrating",  # Synthesis
]

# System prompt header: 10 rows of warm-start context.
# Each column reads vertically as a coherent micro-monologue establishing
# that channel's cognitive voice. User and Output stay silent throughout.
# fmt: off
SYSTEM_PROMPT_WORDS_OLD = [
    # User    Output    Analytical  Skeptical   Intuitive   Between     Curious     Void        Instinct    Synthesis
    ["-",     "-",    " idle",    " quiet",   " resting", " still",   " what",    " quiet",   " waiting", " settling"],
    ["-",     "-",    " ready",   " clear",   " calm",    " space",   " comes",   " room",    " alert",   " into"],
    ["-",     "-",    " for",     " so",      " breath",  " open",    " next",    " light",   " steady",  " readiness"],
    ["-",     "-",    " anything"," far",     " easy",    " for",     " wonder",  " dim",     " patient", " all"],
    ["-",     "-",    " think",   " check",   " present", " whoever", " who",     " air",     " careful", " voices"],
    ["-",     "-",    " through", " every",   " here",    " arrives", " needs",   " still",   " grounded"," finding"],
    ["-",     " OK",    " each",    " angle",   " warm",    " welcome", " something"," hum",    " before",  " their"],
    ["-",     " I'm",    " problem", " first",   " gently",  " them",    " fresh",   " soft",    " acting",  " rhythm"],
    ["-",     " ready",    " carefully"," honestly"," now",     " openly",  " perhaps", " faint",   " measured"," listening"],
    ["-",     "-",    " consider"," then",    " open",    " ready",   " always",  " glow",    " aware",   " together"],
]

SYSTEM_PROMPT_WORDS = [
    # User    Output    Analytical  Skeptical   Intuitive   Between     Curious     Void        Instinct    Synthesis
    ["-",     "-",    " idle",    " quiet",   " resting",   " still",   " what",    " from",    " waiting", " settling"],
    ["-",     "-",    " ready",   " clear",   " calm",      " space",   " comes",   " the",     " alert",   " into"],
    ["-",     "-",    " for",     " so",      " breath",     " open",   " next",    " form",    " steady",  " readiness"],
    ["-",     "-",    " anything"," far",     " easy",      " for",     " wonder",  "less",     " patient", " all"],
    ["-",     "-",    " think",   " check",   " present",   " whoever", " who",     " void",    " careful", " voices"],
    ["-",     "-",    " through", " every",   " here",      " arrives", " needs",   " gaping",  " grounded"," finding"],
    ["-",     " OK",  " each",    " angle",   " letting",   " welcome", " something"," m",      " before",  " their"],
    ["-",     " I",   " problem", " first",   " it",        " them",    " fresh",   "aw",       " acting",  " rhythm"],
    ["-",     "'m",   " carefully"," honestly"," happen",   " openly",  " perhaps", " springs", " measured"," listening"],
    ["-",     " ready"," consider"," then",    " now",       " ready",   " always",  " an",     " aware",   " together"],
    ["-",     "-",    " now",     " always",  " open",      "-",        " always",  " entity",  " now",     " here"],
]
# fmt: on


def get_channel_names(nc: int) -> list[str]:
    """Return channel names for a given channel count."""
    return CHANNELS[:nc] if nc <= len(CHANNELS) else CHANNELS + [f"Ch{i}" for i in range(len(CHANNELS), nc)]


# ── StreamResult ─────────────────────────────────────────────────────────────


@dataclass
class StreamResult:
    """Result of a multi-stream generation."""

    tokens: list[list[int]]  # [R, C] raw token IDs
    channel_texts: dict[str, str]  # channel_name -> full decoded text
    num_rows: int
    silence_token: int

    @property
    def output(self) -> str:
        if "Output" in self.channel_texts:
            return self.channel_texts["Output"]
        # C=1 chat baseline: the single channel IS the output
        if len(self.channel_texts) == 1:
            return next(iter(self.channel_texts.values()))
        return self.channel_texts.get("Output", "")


# ── Model loading ────────────────────────────────────────────────────────────


def load_model(
    model_path: str,
    device: str = "cuda",
    arch: str = "auto",
    device_map: str | None = None,
):
    """Load a multi-stream model, tokenizer, and detect silence token.

    Supports architectures: qwen3, qwen3_5, qwen3_5_moe, llama.
    When device_map is set (e.g. "auto"), uses accelerate device_map instead of .to(device).
    Returns (model, tokenizer, silence_token).
    """
    if arch == "auto":
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "")
            arch_list = cfg.get("architectures") or []
            if model_type == "llama" or any("Llama" in a for a in arch_list):
                arch = "llama"
            elif model_type in ("qwen3_5_moe", "qwen3_5_moe_text") or any("Qwen3_5Moe" in a for a in arch_list):
                arch = "qwen3_5_moe"
            elif model_type in ("qwen3_5", "qwen3_5_text") or any("Qwen3_5" in a for a in arch_list):
                arch = "qwen3_5"
            else:
                arch = "qwen3"
        else:
            arch = "qwen3"

    load_kwargs = dict(torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    if arch == "qwen3":
        from qwen3 import Qwen3ForCausalLM, Qwen3MedusaConfig

        config = Qwen3MedusaConfig.from_pretrained(model_path)
        config.num_channels = getattr(config, "num_channels", None) or C
        config.role_gating_enabled = False
        config.use_cache = True
        model = Qwen3ForCausalLM.from_pretrained(model_path, config=config, **load_kwargs)
    elif arch == "qwen3_5":
        from qwen3_5 import StreamQwen3_5ForCausalLM, StreamQwen3_5TextConfig

        with open(os.path.join(model_path, "config.json")) as f:
            full = json.load(f)
        text_cfg = full.get("text_config", full)
        config = StreamQwen3_5TextConfig(**text_cfg)
        config.num_channels = getattr(config, "num_channels", None) or C
        config.use_cache = True
        model = StreamQwen3_5ForCausalLM.from_pretrained(model_path, config=config, **load_kwargs)
    elif arch == "qwen3_5_moe":
        from qwen3_5_moe import StreamQwen3_5MoeForCausalLM, StreamQwen3_5MoeTextConfig

        with open(os.path.join(model_path, "config.json")) as f:
            full = json.load(f)
        text_cfg = full.get("text_config", full)
        config = StreamQwen3_5MoeTextConfig(**text_cfg)
        config.num_channels = getattr(config, "num_channels", None) or C
        config.use_cache = True
        model = StreamQwen3_5MoeForCausalLM.from_pretrained(model_path, config=config, **load_kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    if device_map is None:
        model = model.to(device)

    model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect silence token ID (SentencePiece vs BPE)
    _sp = tokenizer.convert_ids_to_tokens(tokenizer.encode("test", add_special_tokens=False))[0].startswith("\u2581")
    silence_token = tokenizer.encode("-" if _sp else " -", add_special_tokens=False)[0]

    return model, tokenizer, silence_token


# ── Sampling ─────────────────────────────────────────────────────────────────


def sample_top_p(logits, temperature=0.8, top_p=0.95, top_k=20):
    """Sample from logits with temperature, top-k, and nucleus (top-p) filtering."""
    if temperature <= 0:
        return logits.argmax(dim=-1).item()
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        top_vals, _ = torch.topk(logits, k)
        logits = logits.where(
            logits >= top_vals[-1],
            torch.tensor(float("-inf"), device=logits.device),
        )
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    token = sorted_idx[torch.multinomial(sorted_probs, 1)]
    return token.item()


# ── User tokenization ─────────────────────────────────────────────────────


def _tokenize_user(tokenizer, text: str) -> list[int]:
    """Tokenize user text to match the training data pipeline.

    The data pipeline (process_v12.py) tokenizes each table cell independently
    with a leading space for BPE: encode(" " + word).  We replicate this by
    tokenizing each whitespace-delimited chunk with a leading space, so every
    token gets the mid-sentence BPE form rather than the start-of-string form.
    """
    tokens = []
    for chunk in text.split():
        ids = tokenizer.encode(" " + chunk, add_special_tokens=False)
        tokens.extend(ids)
    return tokens


# ── System prompt builder ─────────────────────────────────────────────────


def build_system_prompt_prefill(
    tokenizer,
    silence_token: int,
    num_channels: int = C,
    system_prompt_version: str = "new",
) -> list[list[int]]:
    """Build prefill rows from SYSTEM_PROMPT_WORDS (header only).

    Returns a list of [num_channels]-length token-id rows ready to be passed
    as `prefill_rows` to `generate()`. The header's job is to establish the
    voice of each thinking channel before any user tokens arrive: for each
    row, thinking channels 2..9 receive one voice-priming BPE token
    (" idle", " ready", " for", " anything", …), User (ch 0) is silence
    throughout, and Output (ch 1) is silence until it gradually becomes
    " OK I 'm ready" over the last few header rows.

    User tokens are NOT added by this function — `generate(warm_start=True)`
    injects them one-per-row into channel 0 during the autoregressive loop
    after prefill ends, while the model continues to generate on channels
    1..9. See `generate()` for the AR injection path.

    Args:
        system_prompt_version: "new" (default) or "old" to select prompt variant.
    """
    prompt_words = SYSTEM_PROMPT_WORDS_OLD if system_prompt_version == "old" else SYSTEM_PROMPT_WORDS
    rows = []
    for word_row in prompt_words:
        token_row = []
        for word in word_row[:num_channels]:
            if word == "-":
                token_row.append(silence_token)
            else:
                toks = tokenizer.encode(word, add_special_tokens=False)
                token_row.append(toks[0])
        rows.append(token_row)
    return rows


# ── Core generation (generator interface) ────────────────────────────────────


@torch.no_grad()
def generate(
    model,
    tokenizer,
    user_text: str,
    silence_token: int,
    *,
    max_rows: int = 200,
    pre_think: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 20,
    silence_penalty: float = 10.0,
    think_silence_penalty: float = 0.0,
    ablate_channels: list[int] | None = None,
    prefill_rows: list[list[int]] | None = None,
    skip_silence: bool = False,
    warm_start: bool = False,
    system_prompt_version: str = "new",
    silence_penalty_during_user_talk: bool = False,
    record_logits: bool = False,
    silence_streak_exit: int | None = None,
) -> Generator[tuple, int | None, None]:
    """Core generation loop.

    Yields per-row tuples. By default (record_logits=False) yields
    `(row_idx, token_ids[C], is_prefill)`. When record_logits=True,
    yields `(row_idx, token_ids[C], is_prefill, logits_summary)` where
    logits_summary is `None` for prefill / seed rows and otherwise a dict
    `{"entropy": [C floats], "top1_prob": [C floats], "top1_token_id": [C ints]}`
    derived from the raw (pre-temperature) softmax of each channel's logits.

    Supports Python generator .send() protocol for interactive use:
    - When user_text is provided, sent values are ignored (backward compatible).
    - When user_text is empty/None, the caller can inject user tokens via
      gen.send(token_id) each step. Silence token is used if None is sent.

    Args:
        model: Loaded multi-stream model.
        tokenizer: The tokenizer.
        user_text: User input text (empty string for interactive/send mode).
        silence_token: Token ID for silence marker.
        max_rows: Maximum rows to generate.
        pre_think: Rows of silence before user input begins.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff (0 = disabled).
        silence_penalty: Logit penalty on silence for Output (ch 1) after user input.
        think_silence_penalty: Logit penalty on silence for thinking channels (2-9).
        ablate_channels: Force these channel indices to silence (for ablation).
        prefill_rows: Optional [C]-token-id rows to feed as context before generating.
        skip_silence: Mask silence tokens as keys (they don't contribute attention).
        silence_penalty_during_user_talk: If True, also apply the Output silence
            penalty while the user is *actively* talking (i.e. row[0] is not
            silence). Normally the penalty fires only after user input has
            ended; this flag forces Output to start producing visible output
            mid-stream while the user is still speaking
        record_logits: If True, yield 4-tuples with a per-row logits_summary
            dict so callers can persist raw entropy / top1_prob / top1_token_id
            per (row, channel) without recomputing. Existing 3-tuple callers
            are unaffected (kwarg defaults to False).
        silence_streak_exit: Exit when this many consecutive rows are entirely
            silent (every channel == silence_token). `None` (default) preserves
            legacy behaviour (1-row exit in non-interactive mode, no exit in
            interactive mode). Set to e.g. 10 for long-horizon rollouts where
            occasional all-silent rows are normal but a sustained streak
            signals the rollout has stalled.
    """
    C = getattr(model.config, "num_channels", 10)  # support variable channels
    _seed_words = SEED_WORDS[:C] if C <= len(SEED_WORDS) else SEED_WORDS + ["-"] * (C - len(SEED_WORDS))
    device = model.get_input_embeddings().weight.device

    def _pack_yield(row_idx, row, is_prefill, logits_summary=None):
        # Choose 3-tuple (legacy) vs 4-tuple (with logits_summary) at the call
        # site — keeps `.send()` semantics intact (yield is in the outer fn).
        return (row_idx, row, is_prefill, logits_summary) if record_logits else (row_idx, row, is_prefill)

    def _logits_summary(row_logits):
        # row_logits: [C, V] tensor of pre-temperature logits.
        # Returns dict with per-channel raw entropy / top1_prob / top1_token_id
        # so the caller can persist these as parquet columns. Computed in
        # float32 to avoid bf16 precision loss; detached to host so the
        # generator's outputs don't pin GPU memory.
        if not record_logits:
            return None
        rl = row_logits.float()
        probs = torch.softmax(rl, dim=-1)
        log_probs = torch.log_softmax(rl, dim=-1)
        ent = -(probs * log_probs).sum(dim=-1)
        top1_prob, top1_id = probs.max(dim=-1)
        return {
            "entropy": ent.detach().cpu().tolist(),
            "top1_prob": top1_prob.detach().cpu().tolist(),
            "top1_token_id": top1_id.detach().cpu().tolist(),
        }

    # Warm-start mode: prefill header only, user tokens injected during AR
    if warm_start:
        prefill_rows = build_system_prompt_prefill(
            tokenizer,
            silence_token,
            num_channels=C,
            system_prompt_version=system_prompt_version,
        )
        user_tokens = _tokenize_user(tokenizer, user_text) if user_text else []
        pre_think = len(prefill_rows)  # user tokens start right after header
    else:
        user_tokens = _tokenize_user(tokenizer, user_text) if user_text else []

    # When user_text is provided, we use the pre-tokenized list (backward compat).
    # When user_text is empty/None, we accept user tokens via generator .send().
    # Interactive mode is independent of where prefill_rows came from
    # (warm_start-built canned header OR caller-supplied training sample).
    interactive = not user_tokens
    user_end_row = pre_think + len(user_tokens)
    got_user_input = False  # for interactive mode: have we ever received user input?
    silence_streak = 0  # for interactive mode: consecutive silence rows after user input
    all_silent_streak = 0  # consecutive rows where every channel == silence_token (for early exit)
    ablate = set(ablate_channels or [])

    if prefill_rows is not None:
        # ── Prefill mode: one forward pass over all prefill rows ──
        n_prefill = len(prefill_rows)
        flat = [t for row in prefill_rows for t in row]
        input_ids = torch.tensor([flat], device=device, dtype=torch.long)
        position_ids = torch.tensor(
            [[r for r in range(n_prefill) for _ in range(C)]],
            device=device,
            dtype=torch.long,
        )
        channel_ids = torch.tensor(
            [[c for _ in range(n_prefill) for c in range(C)]],
            device=device,
            dtype=torch.long,
        )

        # Block-causal mask for prefill
        N = n_prefill * C
        mask = torch.full((1, 1, N, N), -1e4, device=device, dtype=torch.bfloat16)
        for i in range(N):
            ri = i // C
            for j in range(N):
                rj = j // C
                if rj < ri or j == i:
                    mask[0, 0, i, j] = 0.0

        # Skip-silence: mask silence key columns, restore self-diagonal
        if skip_silence:
            sil_cols = input_ids[0] == silence_token
            mask[0, 0, :, :].masked_fill_(sil_cols.unsqueeze(0), -1e4)
            mask[0, 0].diagonal().clamp_(min=0.0)

        outputs = model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=position_ids,
            use_cache=True,
            channel_ids=channel_ids,
        )
        past_kv = outputs.past_key_values
        logits = outputs.logits[0]  # [N, V]

        # Yield prefill rows (ground truth — no model logits to summarise)
        for r in range(n_prefill):
            yield _pack_yield(r, prefill_rows[r], True, None)

        # Sample first generated row from last prefill row's logits
        last_logits = logits[(n_prefill - 1) * C : n_prefill * C]
        row = [sample_top_p(last_logits[c], temperature, top_p, top_k) for c in range(C)]
        for c in ablate:
            row[c] = silence_token
        # Inject user token on channel 0 if applicable
        if user_tokens and n_prefill >= pre_think and n_prefill - pre_think < len(user_tokens):
            row[0] = user_tokens[n_prefill - pre_think]
        elif not interactive:
            row[0] = silence_token  # no user input yet or already past
        sent_value = yield _pack_yield(n_prefill, row, False, _logits_summary(last_logits))
        # In interactive mode (e.g. warm_start + empty user_text), the first
        # post-prefill yield captures the caller's .send() token so the next
        # forward pass sees it on channel 0.
        if interactive and sent_value is not None:
            row[0] = sent_value
            if sent_value != silence_token:
                got_user_input = True
        start_row = n_prefill + 1
        _prefill_ids = input_ids[0]
    else:
        # ── Seed mode: keyword seeds for row 0 ──
        row = []
        for c, word in enumerate(_seed_words):
            if c in ablate:
                row.append(silence_token)
            elif word == "-":
                row.append(silence_token)
            else:
                toks = tokenizer.encode(word, add_special_tokens=False)
                row.append(toks[0])
        sent_value = yield _pack_yield(0, row, False, None)
        # In interactive mode, apply first sent user token to the seed row
        if interactive and sent_value is not None:
            row[0] = sent_value
            if sent_value != silence_token:
                got_user_input = True
        past_kv = None
        start_row = 1
        _prefill_ids = None

    # Track cached token IDs for skip-silence masking
    if _prefill_ids is not None:
        all_cached_ids = _prefill_ids.clone()
    else:
        all_cached_ids = torch.tensor([], device=device, dtype=torch.long)

    # ── Autoregressive loop ──
    for row_idx in range(start_row, start_row + max_rows):
        input_ids = torch.tensor([row], device=device, dtype=torch.long)
        position_ids = torch.full((1, C), row_idx - 1, device=device, dtype=torch.long)
        channel_ids = torch.arange(C, device=device, dtype=torch.long).unsqueeze(0)

        # Block-causal attention mask
        if past_kv is None:
            mask = torch.full((1, 1, C, C), -1e4, device=device, dtype=torch.bfloat16)
            for i in range(C):
                mask[0, 0, i, i] = 0.0
        else:
            cached_len = past_kv.get_seq_length()
            total = cached_len + C
            mask = torch.zeros(1, 1, C, total, device=device, dtype=torch.bfloat16)
            for i in range(C):
                for j in range(C):
                    if i != j:
                        mask[0, 0, i, cached_len + j] = -1e4

        # Skip-silence: mask silence key columns, restore self-diagonal
        if skip_silence:
            row_ids = input_ids[0]
            sil_cols = torch.cat([all_cached_ids == silence_token, row_ids == silence_token])
            mask[0, 0].masked_fill_(sil_cols.unsqueeze(0), -1e4)
            peer_offset = len(all_cached_ids)
            for i in range(C):
                mask[0, 0, i, peer_offset + i] = 0.0

        outputs = model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            channel_ids=channel_ids,
        )
        past_kv = outputs.past_key_values
        all_cached_ids = torch.cat([all_cached_ids, input_ids[0]])
        logits = outputs.logits[0]  # [C, V]

        # Silence penalties after user input ends
        if interactive:
            # In interactive mode, apply penalty when we have had user input
            # and the current user channel is silence. With
            # silence_penalty_during_user_talk=True, also fire the penalty
            # while the user is still talking — forces Output to start
            # speaking mid-stream rather than waiting for user EOS (audit
            # C.1 experiment for the override-moat coverage test).
            apply_post_user_silence = got_user_input and row[0] == silence_token
            apply_during_user_talk = silence_penalty_during_user_talk and got_user_input and row[0] != silence_token
            if apply_post_user_silence or apply_during_user_talk:
                silence_streak += 1
                ramp = min(1.0, silence_streak / 5.0)
                if silence_penalty > 0 and C >= 2:
                    logits[1, silence_token] -= silence_penalty * ramp
                if think_silence_penalty > 0:
                    for c in range(2, C):
                        logits[c, silence_token] -= think_silence_penalty * ramp
            else:
                silence_streak = 0
        elif row_idx >= user_end_row or (silence_penalty_during_user_talk and row_idx >= pre_think):
            # Forced mode: also fire penalty while user is talking (rows
            # pre_think..user_end_row-1). Otherwise the existing post-user
            # ramp begins exactly at user_end_row.
            base = pre_think if (silence_penalty_during_user_talk and row_idx < user_end_row) else user_end_row
            ramp = min(1.0, (row_idx - base + 1) / 5.0)
            if silence_penalty > 0 and C >= 2:
                logits[1, silence_token] -= silence_penalty * ramp
            if think_silence_penalty > 0:
                for c in range(2, C):
                    logits[c, silence_token] -= think_silence_penalty * ramp

        # Determine user channel token first (never sampled from model)
        if interactive:
            user_tok = silence_token
        elif row_idx < pre_think or row_idx - pre_think >= len(user_tokens):
            user_tok = silence_token
        else:
            user_tok = user_tokens[row_idx - pre_think]

        # Capture per-channel logits summary BEFORE sampling so the entropy
        # reflects the raw model distribution rather than the sampled token.
        row_summary = _logits_summary(logits)

        # Sample channels 1-9 from model; channel 0 is always injected
        next_row = [user_tok] + [sample_top_p(logits[c], temperature, top_p, top_k) for c in range(1, C)]

        # Force ablated channels to silence
        for c in ablate:
            next_row[c] = silence_token

        row = next_row

        # Track all-silent streak for early exit. Default streak threshold:
        # 1 in non-interactive mode (legacy behaviour), no exit in interactive
        # mode. Caller can override via silence_streak_exit.
        if all(t == silence_token for t in row):
            all_silent_streak += 1
        else:
            all_silent_streak = 0
        threshold = silence_streak_exit
        if threshold is None:
            threshold = 1 if not interactive else None
        if threshold is not None and all_silent_streak >= threshold:
            return

        sent_value = yield _pack_yield(row_idx, row, False, row_summary)

        # In interactive mode, .send(token_id) overrides user channel for next step.
        # This updates row[0] which becomes the input_ids for the next forward pass.
        if interactive and sent_value is not None:
            row[0] = sent_value
            if sent_value != silence_token:
                got_user_input = True


# ── Convenience functions ────────────────────────────────────────────────────


def collect_result(tokenizer, silence_token: int, rows: list[tuple[int, list[int], bool]]) -> StreamResult:
    """Collect generator output into a StreamResult."""
    all_rows = [row for _, row, _ in rows]
    nc = len(all_rows[0]) if all_rows else C
    names = get_channel_names(nc)
    channel_texts = {}
    for c, name in enumerate(names):
        col_tokens = [r[c] for r in all_rows]
        non_silence = [t for t in col_tokens if t != silence_token]
        channel_texts[name] = tokenizer.decode(non_silence).strip() if non_silence else ""
    return StreamResult(
        tokens=all_rows,
        channel_texts=channel_texts,
        num_rows=len(all_rows),
        silence_token=silence_token,
    )


def generate_to_result(model, tokenizer, user_text: str, silence_token: int, **kwargs) -> StreamResult:
    """Generate and return a StreamResult (convenience wrapper over generate())."""
    rows = list(generate(model, tokenizer, user_text, silence_token, **kwargs))
    return collect_result(tokenizer, silence_token, rows)


def generate_choice(model, tokenizer, user_text: str, silence_token: int, choices: list[str], **kwargs) -> tuple[int, StreamResult]:
    """Generate and pick the best matching choice from Output output.

    For MCQ benchmarks. Returns (chosen_index, result).
    """
    result = generate_to_result(model, tokenizer, user_text, silence_token, **kwargs)
    output = result.output.strip().upper()

    # Try letter matching first (A, B, C, D)
    for i in range(len(choices)):
        letter = chr(ord("A") + i)
        if output.startswith(letter) or output.startswith(f"({letter})"):
            return i, result

    # Fallback: word overlap
    best_idx, best_overlap = 0, 0
    output_lower = result.output.lower()
    for i, choice in enumerate(choices):
        overlap = sum(1 for w in choice.lower().split() if w in output_lower)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    return best_idx, result


# ── StreamModel class (stateful wrapper) ─────────────────────────────────────


class StreamModel:
    """Wraps model + tokenizer into a convenient API for benchmarks."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        arch: str = "auto",
        device_map: str | None = None,
    ):
        self.model, self.tokenizer, self.silence_token = load_model(model_path, device=device, arch=arch, device_map=device_map)
        self.device = device

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> StreamResult:
        """Generate a multi-stream response, returning a StreamResult.

        If system_prompt (str) is provided, it is prepended to the user prompt.
        Use warm_start=True in kwargs to enable the 10-row system prompt header.
        """
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        return generate_to_result(self.model, self.tokenizer, prompt, self.silence_token, **kwargs)

    def generate_choice(self, prompt: str, choices: list[str], **kwargs) -> tuple[int, StreamResult]:
        """Generate and pick the best matching choice from Output output."""
        return generate_choice(self.model, self.tokenizer, prompt, self.silence_token, choices, **kwargs)
