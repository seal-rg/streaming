"""
ChatDataset — converts multi-channel stream data to standard chat format.

Takes the same dataset.jsonl as StreamDataset but extracts non-silence tokens
from the stream grid, groups them into user/assistant turns, and re-tokenizes
using the model's native chat template.

Multi-turn handling follows the official Qwen3 recommendation
(https://github.com/QwenLM/Qwen3/discussions/1398,
 https://huggingface.co/Qwen/Qwen3-32B/discussions/11):

    A K-turn sample [Q1, T1, A1, Q2, T2, A2, Q3, T3, A3] is split into K
    separate training examples:
        - [Q1, T1, A1]
        - [Q1, A1, Q2, T2, A2]
        - [Q1, A1, Q2, A2, Q3, T3, A3]
    Each example trains loss ONLY on the final assistant turn, and earlier
    assistant turns appear in-context with their <think> blocks stripped
    (Qwen3's chat template handles the stripping automatically). This avoids
    OOD training on historical assistant tokens and matches how thinking
    models are used at inference.

Thinking modes (controls what appears in the FINAL assistant turn):
    - "none":        output text only.
    - "analytical":  <think>{analytical channel}</think>\n\n{output}.
    - "all_streams": <think>{labeled per-stream paragraphs}</think>\n\n{output}.

Labels: HF convention — labels[t] = input_ids[t] on the final assistant span,
-100 elsewhere. HF's Trainer applies the shift internally; the loss at logit
position t-1 (which predicts token t) is computed against labels[t]. An
earlier version of this file pre-shifted labels manually, which combined
with HF's internal shift produced a 2-token-ahead objective (see
COMPREHENSIVE_EVAL_REPORT.md, Chat baselines).
"""

import json
import os

import torch
from torch.utils.data import Dataset

# Stream channel layout (matches ALL_CHANNEL_NAMES in train_stream.py).
THINKING_CHANNEL_NAMES = [
    "Analytical",
    "Skeptical",
    "Intuitive",
    "Between",
    "Curious",
    "Void",
    "Instinct",
    "Synthesis",
]
ANALYTICAL_CHANNEL = 2
FIRST_THINKING_CHANNEL = 2
LAST_THINKING_CHANNEL = 9  # inclusive

VALID_THINKING_MODES = ("none", "analytical", "all_streams")


class ChatDataset(Dataset):
    """Dataset that converts stream data to native chat format.

    Args:
        data_path: Directory containing dataset.jsonl.
        tokenizer: HuggingFace tokenizer (must support apply_chat_template).
        max_seq_length: Maximum token length after chat template encoding.
        silence_token_id: Token ID for the silence marker '-' (default 481 for Qwen3).
        thinking_mode: How to convert thinking channels into <think> content
            on the final assistant turn of each expanded example.
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        max_seq_length=4096,
        silence_token_id=481,
        thinking_mode="none",
    ):
        if thinking_mode not in VALID_THINKING_MODES:
            raise ValueError(f"thinking_mode must be one of {VALID_THINKING_MODES}, got {thinking_mode!r}")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.silence_token_id = silence_token_id
        self.thinking_mode = thinking_mode

        # Cache marker token ids so _build_labels can scan without re-encoding.
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # "assistant\n" after <|im_start|> is plain BPE — context-free across
        # special tokens, so we can encode it once and match as a subsequence.
        self.assistant_header_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

        jsonl_path = os.path.join(data_path, "dataset.jsonl")
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        # Cache decoded turns + assistant turn indices per sample, then build
        # an expansion index of (sample_idx, target_asst_turn_idx) items.
        self._turn_cache = {}  # sample_idx -> list[turn dict]
        self._asst_indices_cache = {}  # sample_idx -> list[turn_idx of asst turns]
        self.index = []
        n_skipped = 0
        for i in range(len(self.samples)):
            turns = self._extract_turns(self.samples[i])
            self._turn_cache[i] = turns
            asst_idxs = [j for j, t in enumerate(turns) if t["role"] == "assistant"]
            self._asst_indices_cache[i] = asst_idxs
            if not asst_idxs:
                n_skipped += 1
                continue
            for k in range(len(asst_idxs)):
                self.index.append((i, k))
        if n_skipped > 0:
            print(f"ChatDataset: skipped {n_skipped} samples with no assistant turns")
        print(f"ChatDataset: {len(self.samples)} raw samples → {len(self.index)} per-turn examples (thinking_mode={thinking_mode})")

    def __len__(self):
        return len(self.index)

    def _decode_nonsilent(self, channel_row_ids):
        """Decode a list of token ids, skipping silence tokens."""
        toks = [t for t in channel_row_ids if t != self.silence_token_id]
        if not toks:
            return ""
        return self.tokenizer.decode(toks, skip_special_tokens=True).strip()

    def _build_thinking_text(self, token_ids, start_row, end_row):
        """Return thinking text for an assistant turn spanning rows [start_row, end_row]."""
        if self.thinking_mode == "none" or start_row > end_row:
            return ""
        if self.thinking_mode == "analytical":
            channel = token_ids[ANALYTICAL_CHANNEL]
            return self._decode_nonsilent(channel[start_row : end_row + 1])
        # all_streams
        paragraphs = []
        for offset, name in enumerate(THINKING_CHANNEL_NAMES):
            ch_idx = FIRST_THINKING_CHANNEL + offset
            if ch_idx > LAST_THINKING_CHANNEL or ch_idx >= len(token_ids):
                break
            channel = token_ids[ch_idx]
            text = self._decode_nonsilent(channel[start_row : end_row + 1])
            if text:
                paragraphs.append(f"{name}: {text}")
        return "\n\n".join(paragraphs)

    def _extract_turns(self, sample):
        """Convert a stream sample into a list of chat turns.

        Each assistant turn's content is built as
            <think>\n{thinking}\n</think>\n\n{output}
        when thinking_mode != "none". The Qwen3 chat template will later
        strip <think> from all but the final assistant turn — that's the
        intended behaviour, paired with example splitting in __getitem__.
        """
        num_rows = sample["num_rows"]
        token_ids = sample["token_ids"]  # [C][R]
        user_channel = token_ids[0]
        output_channel = token_ids[1]

        # Per-row primary speaker; prefer user when both have a non-silence token.
        row_roles = []
        for r in range(num_rows):
            u_speaks = user_channel[r] != self.silence_token_id
            o_speaks = output_channel[r] != self.silence_token_id
            if u_speaks:
                row_roles.append("user")
            elif o_speaks:
                row_roles.append("assistant")
            else:
                row_roles.append(None)

        # Group into turn row-ranges (contiguous rows with same non-None role).
        turn_ranges = []  # list of (role, start_row, end_row)
        cur_role = None
        cur_start = None
        cur_end = None
        for r, role in enumerate(row_roles):
            if role is None:
                continue
            if cur_role is None:
                cur_role, cur_start, cur_end = role, r, r
            elif role == cur_role:
                cur_end = r
            else:
                turn_ranges.append((cur_role, cur_start, cur_end))
                cur_role, cur_start, cur_end = role, r, r
        if cur_role is not None:
            turn_ranges.append((cur_role, cur_start, cur_end))

        turns = []
        prev_asst_end = -1
        for role, start_row, end_row in turn_ranges:
            if role == "user":
                text = self._decode_nonsilent(user_channel[start_row : end_row + 1])
                if text:
                    turns.append({"role": "user", "content": text})
            else:
                output_text = self._decode_nonsilent(output_channel[start_row : end_row + 1])
                if not output_text:
                    continue
                # Thinking spans from the row after the previous assistant
                # turn through the end of this assistant turn — captures any
                # pre-thinking that happened while the user was speaking.
                thinking_text = self._build_thinking_text(token_ids, prev_asst_end + 1, end_row)
                if thinking_text:
                    content = f"<think>\n{thinking_text}\n</think>\n\n{output_text}"
                else:
                    content = output_text
                turns.append({"role": "assistant", "content": content})
                prev_asst_end = end_row

        # Chat templates require user-first alternation.
        if turns and turns[0]["role"] == "assistant":
            turns.insert(0, {"role": "user", "content": "(listening)"})

        return turns

    def _find_assistant_spans(self, chat_ids):
        """Scan chat_ids for assistant content spans.

        Returns list of (start, end) tuples — start inclusive, end exclusive,
        where the span covers everything between the 'assistant\\n' header and
        the closing <|im_end|> (inclusive of <|im_end|>, so the model is
        trained to emit the stop token).
        """
        spans = []
        N = len(chat_ids)
        H = len(self.assistant_header_ids)
        if H == 0:
            return spans
        i = 0
        while i < N:
            if chat_ids[i] != self.im_start_id:
                i += 1
                continue
            header_start = i + 1
            header_end = header_start + H
            if header_end > N:
                break
            if chat_ids[header_start:header_end] != self.assistant_header_ids:
                i += 1
                continue
            content_start = header_end
            k = content_start
            while k < N and chat_ids[k] != self.im_end_id:
                k += 1
            content_end = k + 1 if k < N else N
            if content_end > content_start:
                spans.append((content_start, content_end))
            i = content_end
        return spans

    def _build_labels_final_only(self, chat_ids):
        """Labels = chat_ids on the FINAL assistant span only, -100 elsewhere.

        HF's Trainer applies the shift internally: the logit at position t-1
        (which predicts token t) is supervised against labels[t]. Setting
        labels[t] = chat_ids[t] on assistant tokens t therefore gives the
        correct next-token objective for that span.
        """
        labels = torch.full((len(chat_ids),), -100, dtype=torch.long)
        spans = self._find_assistant_spans(chat_ids)
        if not spans:
            return labels
        start, end = spans[-1]
        end = min(end, len(chat_ids))
        if start < end:
            chat_ids_tensor = torch.tensor(chat_ids, dtype=torch.long)
            labels[start:end] = chat_ids_tensor[start:end]
        return labels

    def __getitem__(self, idx):
        sample_idx, target_k = self.index[idx]
        full_turns = self._turn_cache[sample_idx]
        asst_idxs = self._asst_indices_cache[sample_idx]
        target_turn_idx = asst_idxs[target_k]
        truncated = full_turns[: target_turn_idx + 1]

        chat_ids = self.tokenizer.apply_chat_template(
            truncated,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )

        if len(chat_ids) > self.max_seq_length:
            # Bottom-trim: keep the prefix so the final assistant turn (the
            # one that gets loss) is preserved at the end. If even after
            # trimming the final turn is gone, _build_labels will return all
            # -100 and the example contributes no loss — that's acceptable
            # but rare; warn on first occurrence in __init__-style code if
            # needed. For now, top-trim instead so the final asst stays.
            chat_ids = chat_ids[-self.max_seq_length :]

        input_ids = torch.tensor(chat_ids, dtype=torch.long)
        labels = self._build_labels_final_only(chat_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(len(chat_ids), dtype=torch.long),
        }


class ChatDataCollator:
    """Simple collator for chat data — pads to max length in batch."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        B = len(features)
        S_max = max(len(f["input_ids"]) for f in features)

        input_ids = torch.full((B, S_max), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, S_max), -100, dtype=torch.long)
        attention_mask = torch.zeros((B, S_max), dtype=torch.long)

        for b, f in enumerate(features):
            S = len(f["input_ids"])
            input_ids[b, :S] = f["input_ids"]
            labels[b, :S] = f["labels"]
            attention_mask[b, :S] = f["attention_mask"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
