#!/usr/bin/env python3

"""
SingleFileDatasetProcessor (FULL, runnable, no omissions) -- MODIFIED VERSION

✅ Changes made (as requested)
=============================
1) Removed interaction augmentation completely (no imports / no args / no logic).
2) Removed <|wait|> logic completely:
   - No wait token added to tokenizer
   - No wait embedding init
   - Segmented alignment padding uses PAD token (pad_token_id) instead
3) Removed fixed system/user prompts:
   - system message uses --system-message (global default)
   - user content is ONLY the sample input/question (no fixed prompt prefix)

✅ NEW: Support instruction/input/output format
==============================================
Input JSON:
{
  "instruction": "...",   -> per-sample system prompt (override global)
  "input": "...",         -> user content (can be empty)
  "output": "..."         -> assistant (or list[str])
}
head=1 typical: set --fixed-heads 1

Other key behaviors kept:
- IO-list format: {"channels":[{"input":...,"output":...}, ...]}
  - user contains input
  - assistant heads contain outputs (pick exactly fixed_heads)
- Labels default: supervise ALL assistant heads (solution tokens + <|im_end|>)
  - Optional supervise_head_indices supported if item provides it
- Position IDs mapping (per your updated rule):
  - system: cid=0, y starts 0
  - user:   cid=1, y starts at y0=len(system_block)
  - assistant h: cid=2+h, y starts at same y0
- Attention mask:
  - Intra-block strict causal
  - Cross-channel rule: eligible keys with y_j < y_q
  - include_im_end_in_cross_channel toggle
  - allow_assistant_cross_channel toggle
- Per-sample .npz output + index.json
- Supports plain_heads / segmented_channels / channels_augmented packed format
- Optional assistant intro (I am channel C{idx}.) kept (NOT wait)

NOTE:
- attention_mask is NxN and can be extremely large for long sequences.
"""

import argparse
import json
import os
import re
import time
from typing import Any

import numpy as np
from transformers import AutoTokenizer


class SingleFileDatasetProcessor:
    """Processor that saves each sample as a separate .npz file"""

    def __init__(
        self,
        tokenizer_path: str,
        max_seq_length: int,
        fixed_heads: int,
        truncation_strategy: str = "random",
        max_position_embeddings: int = 32678,
        position_ids_2d: bool = True,
        system_message: str = "You are a helpful assistant.",
        # Length filter (assistant RAW content length only, pre-rebuild)
        max_head_length: int | None = 20000,
        max_length_ratio: float = 100.0,
        max_cv: float = 0.8,
        max_padding_percent: float = 100.0,
        enable_length_filter: bool = True,
        # segmented mode alignment
        align_first_k_segments: int = 1,
        # cross-channel eligibility toggle
        include_im_end_in_cross_channel: bool = True,
        # whether assistant heads can see other assistants cross-channel
        allow_assistant_cross_channel: bool = True,
        # assistant intro (kept)
        add_channel_intro: bool = True,
        channel_intro_template: str = "I am channel C{idx}.",
        # pick K heads from bigger pool
        head_indices: list[int] | None = None,
        head_pick_strategy: str = "first",  # first|random|sorted_key
    ):
        self.tokenizer_path = tokenizer_path
        self.max_seq_length = int(max_seq_length)
        self.fixed_heads = int(fixed_heads)
        self.truncation_strategy = truncation_strategy
        self.max_position_embeddings = int(max_position_embeddings)
        self.position_ids_2d = bool(position_ids_2d)
        self.system_message = str(system_message)

        # length filter
        self.max_head_length = max_head_length
        self.max_length_ratio = float(max_length_ratio)
        self.max_cv = float(max_cv)
        self.max_padding_percent = float(max_padding_percent)
        self.enable_length_filter = bool(enable_length_filter)

        # segmented alignment
        self.align_first_k_segments = int(align_first_k_segments) if align_first_k_segments is not None else 1
        if self.align_first_k_segments < 0:
            self.align_first_k_segments = 0

        # cross-channel eligibility
        self.include_im_end_in_cross_channel = bool(include_im_end_in_cross_channel)
        self.allow_assistant_cross_channel = bool(allow_assistant_cross_channel)

        # assistant intro
        self.add_channel_intro = bool(add_channel_intro)
        self.channel_intro_template = str(channel_intro_template)

        # pick heads from pool
        self.head_indices = head_indices
        self.head_pick_strategy = str(head_pick_strategy or "first").lower()

        # packed-format helpers
        self._packed_endoftext_str = "<|endoftext|>"

        print(f"Initialized with fixed assistant heads: {self.fixed_heads}")
        print(f"Max sequence length: {self.max_seq_length}")
        print("Mode: One sample per .npz file")
        print("Core: y-aligned multi-channel rebuild (system/user/assistants as channels)")
        print(f"Segment align (first k segments): {self.align_first_k_segments}")
        print(f"Cross-channel includes <|im_end|>: {self.include_im_end_in_cross_channel}")
        print(f"Cross-channel assistant<->assistant: {self.allow_assistant_cross_channel}")
        print(f"Assistant intro enabled: {self.add_channel_intro}")
        if self.add_channel_intro:
            print(f"  intro template: {self.channel_intro_template}")
        print(f"Head pick: indices={self.head_indices} | strategy={self.head_pick_strategy}")
        print("Rebuilt sequence: NO newline token after <|im_end|>")
        print(f"System message (global default): {self.system_message}")

        if self.enable_length_filter:
            print("\n=== Length Filter Configuration ===")
            if self.max_head_length:
                print(f"Max head length: {self.max_head_length}")
            print(f"Max length ratio: {self.max_length_ratio}x")
            print(f"Max CV: {self.max_cv} (computed; not enforced)")
            print(f"Max padding percent: {self.max_padding_percent}%")
            print(f"Filter enabled: {self.enable_length_filter}")
        else:
            print("\n⚠️  Length filter DISABLED")

        print(f"\nLoading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self._setup_special_tokens()

        self.filter_stats = {
            "total_processed": 0,
            "filtered_by_head_length": 0,
            "filtered_by_length_ratio": 0,
            "filtered_by_cv": 0,
            "filtered_by_padding": 0,
            "filtered_by_seq_len": 0,
            "passed_filter": 0,
        }

    # ===================== Pick K from pool =====================

    def _pick_k(self, candidates: list[Any], k: int, rng: np.random.Generator | None = None) -> list[Any]:
        """Pick exactly k items from candidates. If impossible, return []."""
        if k <= 0:
            return []
        if len(candidates) < k:
            return []
        if len(candidates) == k:
            return candidates

        # explicit indices first (deterministic)
        if self.head_indices is not None:
            idxs = [i for i in self.head_indices if isinstance(i, int) and 0 <= i < len(candidates)]
            if len(idxs) < k:
                return []
            return [candidates[i] for i in idxs[:k]]

        strat = (self.head_pick_strategy or "first").lower()
        if strat == "first":
            return candidates[:k]
        if strat == "random":
            if rng is None:
                rng = np.random.default_rng()
            perm = rng.permutation(len(candidates))[:k]
            return [candidates[i] for i in perm.tolist()]
        if strat == "sorted_key":
            return candidates[:k]
        return candidates[:k]

    # ===================== Token helpers =====================

    def _setup_special_tokens(self):
        """Pre-compute special tokens. (NO <|wait|> logic in this version)"""
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        nl = self.tokenizer.encode("\n", add_special_tokens=False)
        self.newline_token = nl[0] if len(nl) > 0 else None
        if self.newline_token is None:
            raise ValueError("Tokenizer cannot encode newline token.")

        # role tokens
        self.system_tokens = self.tokenizer.encode("system", add_special_tokens=False)
        self.user_tokens = self.tokenizer.encode("user", add_special_tokens=False)
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)

        # wrapper prefixes: <|im_start|> role \n
        self.system_prefix = [self.im_start] + self.system_tokens + [self.newline_token]
        self.user_prefix = [self.im_start] + self.user_tokens + [self.newline_token]
        self.assistant_prefix = [self.im_start] + self.assistant_tokens + [self.newline_token]

        self.prefix_len_system = len(self.system_prefix)
        self.prefix_len_user = len(self.user_prefix)
        self.prefix_len_assistant = len(self.assistant_prefix)

        print(f"Special tokens - im_start: {self.im_start}, im_end: {self.im_end}, newline_token: {self.newline_token}")
        print(f"Prefix lens - system:{self.prefix_len_system}, user:{self.prefix_len_user}, assistant:{self.prefix_len_assistant}")

    def _encode_channel_intro(self, idx: int) -> list[int]:
        txt = self.channel_intro_template.format(idx=idx)
        if not txt.endswith(" "):
            txt += " "
        return self.tokenizer.encode(txt, add_special_tokens=False)

    def _strip_segment0_channel_intro(self, text: str, channel_id: int) -> str:
        """Remove leading channel intro from segment0 text to avoid duplication with rebuild-intro."""
        if not isinstance(text, str) or not text.strip():
            return text

        s = text.lstrip()

        # exact template match first
        try:
            exact = self.channel_intro_template.format(idx=channel_id).strip()
            if s.startswith(exact):
                s2 = s[len(exact) :]
                return s2.lstrip(" \t\r\n")
        except Exception:
            pass

        cid = int(channel_id)
        patterns = [
            rf"^\s*i\s*am\s*channel\s*c{cid}\s*[\.\:\-–—]?\s*",
            rf"^\s*i['’]\s*m\s*channel\s*c{cid}\s*[\.\:\-–—]?\s*",
            rf"^\s*i\s*am\s*c{cid}\s*[\.\:\-–—]?\s*",
        ]
        for pat in patterns:
            s_new = re.sub(pat, "", s, flags=re.IGNORECASE)
            if s_new != s:
                return s_new.lstrip(" \t\r\n")

        return text

    # ===================== Packed (channels_augmented) cleaning =====================

    def _strip_trailing_special_runs(self, text: str, token_str: str) -> str:
        """Remove repeated special-token strings at the END of text."""
        if not isinstance(text, str) or not text:
            return text
        pat = rf"(?:\s*{re.escape(token_str)})+\s*$"
        return re.sub(pat, "", text)

    def _clean_channels_augmented_text(self, text: str, head_idx: int) -> str:
        """Clean channels_augmented text."""
        if not isinstance(text, str):
            return text
        s = text
        s = self._strip_trailing_special_runs(s, self._packed_endoftext_str)
        if self.add_channel_intro:
            s = self._strip_segment0_channel_intro(s, head_idx)
        return s

    # ===================== Length Filter =====================

    def _validate_head_lengths(self, head_lengths: list[int], sample_id: str = "") -> tuple[bool, str]:
        if not self.enable_length_filter:
            return True, "Filter disabled"
        if len(head_lengths) < 2:
            return True, "Only 1 head, no comparison needed"

        if self.max_head_length:
            max_head = max(head_lengths)
            if max_head > self.max_head_length:
                self.filter_stats["filtered_by_head_length"] += 1
                return False, f"Max head length {max_head} > {self.max_head_length}"

        min_len = min(head_lengths)
        max_len = max(head_lengths)
        length_ratio = max_len / max(min_len, 1)
        if length_ratio > self.max_length_ratio:
            self.filter_stats["filtered_by_length_ratio"] += 1
            return False, f"Length ratio {length_ratio:.1f}x > {self.max_length_ratio}x (lengths: {head_lengths})"

        # CV computed only (not enforced)
        mean_len = float(np.mean(head_lengths))
        std_len = float(np.std(head_lengths))
        _cv = std_len / max(mean_len, 1.0)
        _ = _cv

        padding_percentages = [(max_len - l) / max(max_len, 1) * 100 for l in head_lengths]
        max_padding = max(padding_percentages) if padding_percentages else 0.0
        if max_padding > self.max_padding_percent:
            self.filter_stats["filtered_by_padding"] += 1
            return False, f"Max padding {max_padding:.1f}% > {self.max_padding_percent}% (lengths: {head_lengths})"

        return True, "Passed all checks"

    # ===================== Save single sample =====================

    def save_single_sample(self, processed_sample: dict, output_dir: str, sample_idx: int) -> str:
        sample_file = os.path.join(output_dir, f"sample_{sample_idx:06d}.npz")
        boundaries = processed_sample.get("boundaries", {})
        boundaries_json = json.dumps(boundaries)

        sample_data = {
            "input_ids": np.array(processed_sample["input_ids"], dtype=np.int32),
            "labels": np.array(processed_sample["labels"], dtype=np.int32),
            "position_ids": np.array(processed_sample["position_ids"], dtype=np.int32),
            "attention_mask": processed_sample["attention_mask"].astype(np.float16),
            "num_heads": np.int16(processed_sample["num_heads"]),
            "seq_length": np.int32(processed_sample["seq_length"]),
            "boundaries_json": boundaries_json,
        }
        np.savez_compressed(sample_file, **sample_data)
        return sample_file

    # ===================== Multi-file processing =====================

    def process_jsonl_files(self, jsonl_paths: list[str], output_dir: str) -> str:
        if not isinstance(jsonl_paths, list) or len(jsonl_paths) == 0:
            raise ValueError("jsonl_paths must be a non-empty list")

        print(f"\nProcessing {len(jsonl_paths)} JSONL files into: {output_dir}")
        print(f"Using fixed assistant heads: {self.fixed_heads}")
        os.makedirs(output_dir, exist_ok=True)

        index_file_path = os.path.join(output_dir, "index.json")

        start_time = time.time()
        total_processed = 0
        total_failed = 0
        sample_files: list[dict[str, Any]] = []
        filtered_samples: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []

        print("\nStarting multi-file line-by-line processing...")

        for fp in jsonl_paths:
            if not os.path.exists(fp):
                print(f"⚠️  Skip missing file: {fp}")
                continue

            file_processed = 0
            file_failed = 0
            sources.append({"path": fp, "mtime": os.path.getmtime(fp)})

            print(f"\n--- File: {fp} ---")

            with open(fp, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    self.filter_stats["total_processed"] += 1

                    try:
                        item = json.loads(line)
                        standardized = self._standardize_item(item)
                        if not standardized:
                            total_failed += 1
                            file_failed += 1
                            continue

                        processed = self._process_single_item(standardized, line_num)
                        if not processed:
                            total_failed += 1
                            file_failed += 1
                            if "filter_reason" in standardized:
                                filtered_samples.append(
                                    {
                                        "file": fp,
                                        "line": line_num,
                                        "sample_id": standardized.get("sample_id", f"{os.path.basename(fp)}:{line_num}"),
                                        "reason": standardized["filter_reason"],
                                        "head_lengths": standardized.get("head_lengths", []),
                                    }
                                )
                            continue

                        sample_file = self.save_single_sample(processed, output_dir, total_processed)
                        sample_files.append(
                            {
                                "file": os.path.basename(sample_file),
                                "sample_idx": total_processed,
                                "seq_length": processed["seq_length"],
                                "num_heads": processed["num_heads"],
                                "source_file": os.path.basename(fp),
                                "source_line": line_num,
                            }
                        )

                        total_processed += 1
                        file_processed += 1

                        if total_processed % 100 == 0:
                            print(f"Processed: {total_processed}, Failed: {total_failed}")

                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse line {line_num} in {fp}")
                        total_failed += 1
                        file_failed += 1
                    except Exception as e:
                        print(f"Error processing {fp} line {line_num}: {e}")
                        total_failed += 1
                        file_failed += 1

            print(f"File done: {fp} | processed={file_processed}, failed={file_failed}")

        if total_processed == 0:
            raise ValueError("No valid processed data")

        print(f"\nAll files done: processed={total_processed}, failed={total_failed}")

        self._print_filter_statistics()

        index_data = {
            "version": "FULL-y-aligned-channels__NO_INTERACTION__NO_WAIT__NO_FIXED_USER_PROMPT__INPUT_IN_USER__MULTI_ASSISTANTS__INSTRUCTION_INPUT_OUTPUT",
            "total_samples": len(sample_files),
            "sources": sources,
            "cache_created": time.time(),
            "config": {
                "tokenizer_path": self.tokenizer_path,
                "max_seq_length": self.max_seq_length,
                "fixed_heads": self.fixed_heads,
                "truncation_strategy": self.truncation_strategy,
                "max_position_embeddings": self.max_position_embeddings,
                "position_ids_2d": self.position_ids_2d,
                "system_message": self.system_message,
                "align_first_k_segments": self.align_first_k_segments,
                "include_im_end_in_cross_channel": self.include_im_end_in_cross_channel,
                "allow_assistant_cross_channel": self.allow_assistant_cross_channel,
                "add_channel_intro": self.add_channel_intro,
                "channel_intro_template": self.channel_intro_template,
                "head_pick": {
                    "head_indices": self.head_indices,
                    "head_pick_strategy": self.head_pick_strategy,
                },
                "length_filter": {
                    "enabled": self.enable_length_filter,
                    "max_head_length": self.max_head_length,
                    "max_length_ratio": self.max_length_ratio,
                    "max_cv": self.max_cv,
                    "max_padding_percent": self.max_padding_percent,
                },
            },
            "sample_files": sample_files,
            "filter_statistics": self.filter_stats,
        }

        if filtered_samples:
            index_data["filtered_samples_preview"] = filtered_samples[:10]
            index_data["total_filtered"] = len(filtered_samples)

        with open(index_file_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        print(f"Index file saved: {index_file_path}")

        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")
        self._print_statistics(total_processed, output_dir)

        return output_dir

    def process_jsonl_file(self, jsonl_path: str, output_dir: str) -> str:
        return self.process_jsonl_files([jsonl_path], output_dir)

    # ===================== Standardize item =====================

    def _standardize_item(self, item: dict) -> dict | None:
        """
        Standardize different input formats into internal representation.

        PRIORITY:
        - instruction/input/output format
        - channels list of {input, output}
        - other formats...
        """

        # ==========================================================
        # [NEW] instruction/input/output format (alpaca-like)
        # instruction -> system (per-sample override)
        # input       -> user (can be empty)
        # output      -> assistant (str or list[str])
        # ==========================================================
        if isinstance(item, dict) and ("instruction" in item) and ("input" in item) and ("output" in item):
            instr = item.get("instruction", "")
            usr_in = item.get("input", "")
            out = item.get("output", "")

            if not isinstance(instr, str):
                instr = "" if instr is None else str(instr)
            if not isinstance(usr_in, str):
                usr_in = "" if usr_in is None else str(usr_in)

            outputs: list[str] = []
            if isinstance(out, str):
                if out.strip():
                    outputs = [out.strip()]
            elif isinstance(out, list):
                outputs = [x.strip() for x in out if isinstance(x, str) and x.strip()]

            if len(outputs) == 0:
                return None

            picked_outs = self._pick_k(outputs, self.fixed_heads)
            if not picked_outs:
                return None

            return {
                "mode": "plain_heads",
                "system_message": instr.strip() if instr.strip() else self.system_message,
                "question": (usr_in or "").strip(),  # allow empty
                "answers": picked_outs,
                "sample_id": item.get("sample_id") or item.get("id") or "unknown",
            }

        # ==========================================================
        # channels list of {input, output} -> input in USER, outputs in assistants
        # ==========================================================
        ch_list = item.get("channels", None)
        if isinstance(ch_list, list) and len(ch_list) > 0 and all(isinstance(x, dict) for x in ch_list):
            inputs = []
            outputs = []
            for d in ch_list:
                inp = d.get("input", "")
                out = d.get("output", "")
                if isinstance(inp, str) and inp.strip():
                    inputs.append(inp.strip())
                if isinstance(out, str) and out.strip():
                    outputs.append(out.strip())

            if len(outputs) == 0:
                return None

            input_text = inputs[0] if len(inputs) > 0 else ""

            picked_outs = self._pick_k(outputs, self.fixed_heads)
            if not picked_outs:
                return None

            return {
                "mode": "plain_heads",
                "question": input_text,
                "answers": picked_outs,
                "sample_id": item.get("sample_id", "unknown"),
            }

        # ==========================================================
        # [0] simple multichannel dict format: draft/correct/final
        # ==========================================================
        channels_dict = item.get("channels", None)
        if isinstance(channels_dict, dict):
            ordered_keys = ["draft", "correct", "final"]
            answers: list[str] = []

            for k in ordered_keys:
                if k not in channels_dict:
                    return None
                entry = channels_dict.get(k)
                if not isinstance(entry, dict):
                    return None
                text = entry.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    return None
                answers.append(text)

            if len(answers) != self.fixed_heads:
                return None

            question = item.get("user") or item.get("question") or item.get("problem") or item.get("prompt") or ""
            if not isinstance(question, str):
                question = ""

            return {
                "mode": "plain_heads",
                "question": question,
                "answers": answers,
                "sample_id": item.get("sample_id", "unknown"),
            }

        # ==========================================================
        # [1] segmented_channels format
        # ==========================================================
        question = item.get("problem") or item.get("question") or item.get("prompt")
        if question is None:
            question = ""
        if not isinstance(question, str):
            question = str(question)

        segmented = item.get("channels", None)
        num_channels = item.get("num_channels", None)

        if isinstance(segmented, list) and len(segmented) > 0 and isinstance(segmented[0], list):
            if len(segmented[0]) > 0 and isinstance(segmented[0][0], dict) and "channel" in segmented[0][0] and "text" in segmented[0][0]:
                if num_channels is None:
                    ch_ids = set()
                    for seg in segmented:
                        if not isinstance(seg, list):
                            continue
                        for d in seg:
                            if isinstance(d, dict) and "channel" in d:
                                try:
                                    ch_ids.add(int(d["channel"]))
                                except Exception:
                                    pass
                    num_channels = (max(ch_ids) + 1) if ch_ids else 0

                num_channels = int(num_channels)
                if num_channels < self.fixed_heads:
                    return None

                all_ch = list(range(num_channels))
                picked_ch = self._pick_k(all_ch, self.fixed_heads)
                if not picked_ch:
                    return None

                return {
                    "mode": "segmented_channels",
                    "question": question,
                    "segmented_channels": segmented,
                    "num_channels": int(self.fixed_heads),
                    "picked_channels": picked_ch,
                    "sample_id": item.get("sample_id", "unknown"),
                }

        # ==========================================================
        # [2] channels_augmented (packed format)
        # ==========================================================
        channels_aug = item.get("channels_augmented", None)
        if isinstance(channels_aug, list) and len(channels_aug) > 0:
            answers_all = [x for x in channels_aug if isinstance(x, str) and x.strip()]
            picked = self._pick_k(answers_all, self.fixed_heads)
            if not picked:
                return None

            answers = [self._clean_channels_augmented_text(ans, hi) for hi, ans in enumerate(picked)]

            return {
                "mode": "plain_heads",
                "question": question,
                "answers": answers,
                "sample_id": item.get("sample_id", "unknown"),
            }

        # ==========================================================
        # [3] plain heads fallback (agents / channels list)
        # ==========================================================
        agents = item.get("agents", {}) or item.get("channels", [])
        answers: list[str] = []

        if isinstance(agents, dict):
            keys = list(agents.keys())
            if self.head_pick_strategy == "sorted_key":
                keys = sorted(keys, key=lambda x: str(x))
            answers_all = [agents[k] for k in keys]
            answers_all = [resp for resp in answers_all if isinstance(resp, str) and resp.strip()]
            picked = self._pick_k(answers_all, self.fixed_heads)
            if not picked:
                return None
            answers = picked

        elif isinstance(agents, list):
            answers_all = [resp for resp in agents if isinstance(resp, str) and resp.strip()]
            picked = self._pick_k(answers_all, self.fixed_heads)
            if not picked:
                return None
            answers = picked
        else:
            return None

        return {
            "mode": "plain_heads",
            "question": question,
            "answers": answers,
            "sample_id": item.get("sample_id", "unknown"),
        }

    # ===================== Segmented building helpers =====================

    def _diversify_summary(self, summary_text: str, channel_id: int) -> str:
        variants = [
            "Overall summary:\n{S}",
            "Final check and wrap-up:\n{S}",
            "Consolidated results:\n{S}",
            "Brief recap:\n{S}",
            "Answer key (summary):\n{S}",
        ]
        tpl = variants[channel_id % len(variants)]
        return tpl.format(S=summary_text.strip())

    def _pad_to_length_pad(self, ids: list[int], target_len: int) -> list[int]:
        """Pad using PAD token for segmented alignment (NO <|wait|>)."""
        if len(ids) >= target_len:
            return ids[:target_len]
        return ids + [self.pad_token_id] * (target_len - len(ids))

    def _build_heads_from_segmented_channels(
        self,
        segmented_channels: list[list[dict[str, Any]]],
        num_channels: int,
        picked_channels: list[int] | None = None,
    ) -> list[list[int]]:
        """Return per-channel token lists (WITHOUT assistant wrappers)."""
        if picked_channels is None:
            picked_channels = list(range(num_channels))
        K = len(picked_channels)

        summary_segment = segmented_channels[-1] if len(segmented_channels) > 0 else []
        main_segments = segmented_channels[:-1] if len(segmented_channels) > 1 else []

        seg_tokens: list[list[list[int]]] = []

        for seg_idx, seg in enumerate(main_segments):
            ch2text: dict[int, str] = {}
            for d in seg:
                if isinstance(d, dict) and "channel" in d and "text" in d:
                    try:
                        ch2text[int(d["channel"])] = d["text"]
                    except Exception:
                        continue

            per_ch_ids: list[list[int]] = []
            for new_ch in range(K):
                orig_ch = int(picked_channels[new_ch])
                text = ch2text.get(orig_ch, "")

                if seg_idx == 0 and self.add_channel_intro:
                    text = self._strip_segment0_channel_intro(text, new_ch)

                ids = self.tokenizer.encode(text, add_special_tokens=False)
                per_ch_ids.append(ids)

            seg_tokens.append(per_ch_ids)

        # align first k segments with PAD
        k = max(0, int(self.align_first_k_segments))
        k = min(k, len(seg_tokens))
        for s in range(k):
            seg_s = seg_tokens[s]
            max_len_s = max((len(x) for x in seg_s), default=0)
            seg_tokens[s] = [self._pad_to_length_pad(x, max_len_s) for x in seg_s]

        heads_ids = [[] for _ in range(K)]
        for seg_idx in range(len(seg_tokens)):
            for ch in range(K):
                if len(heads_ids[ch]) > 0:
                    heads_ids[ch].append(self.newline_token)
                heads_ids[ch].extend(seg_tokens[seg_idx][ch])

        max_pre_summary = max((len(h) for h in heads_ids), default=0)
        heads_ids = [self._pad_to_length_pad(h, max_pre_summary) for h in heads_ids]

        summary_texts: list[str] = []
        for d in summary_segment:
            if isinstance(d, dict) and "text" in d:
                summary_texts.append(d["text"])
        summary_text = "\n".join(summary_texts).strip()

        if summary_text:
            for ch in range(K):
                diversified = self._diversify_summary(summary_text, ch)
                heads_ids[ch].append(self.newline_token)
                heads_ids[ch].extend(self.tokenizer.encode(diversified, add_special_tokens=False))

        return heads_ids

    def _build_chatml_input_ids_from_heads(self, question: str, heads_ids: list[list[int]]) -> list[int]:
        """Build RAW ChatML token sequence with delimiter newlines between blocks."""
        sys_msg = self.system_message
        usr_msg = (question or "").strip()

        prefix_text = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{usr_msg}<|im_end|>\n"
        input_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)

        for ch_ids in heads_ids:
            input_ids.extend(self.assistant_prefix)
            input_ids.extend(ch_ids)
            input_ids.append(self.im_end)
            input_ids.append(self.newline_token)

        return input_ids

    # ===================== Raw parsing (robust scan) =====================

    def _match_role_at(self, input_ids: list[int], idx_im_start: int) -> str | None:
        j = idx_im_start + 1
        if j >= len(input_ids):
            return None
        if input_ids[j : j + len(self.system_tokens)] == self.system_tokens:
            return "system"
        if input_ids[j : j + len(self.user_tokens)] == self.user_tokens:
            return "user"
        if input_ids[j : j + len(self.assistant_tokens)] == self.assistant_tokens:
            return "assistant"
        return None

    def _content_start_for_role(self, idx_im_start: int, role: str) -> int:
        if role == "system":
            return idx_im_start + 1 + len(self.system_tokens) + 1
        if role == "user":
            return idx_im_start + 1 + len(self.user_tokens) + 1
        if role == "assistant":
            return idx_im_start + 1 + len(self.assistant_tokens) + 1
        raise ValueError(f"Unknown role: {role}")

    def _find_blocks_raw(self, input_ids: list[int]) -> dict[str, Any]:
        """
        Robust: scan left-to-right, pair each <|im_start|> with the next <|im_end|>.
        """
        blocks = []
        i = 0
        n = len(input_ids)
        while i < n:
            if input_ids[i] != self.im_start:
                i += 1
                continue

            bs = i
            role = self._match_role_at(input_ids, bs) or "unknown"
            cs = self._content_start_for_role(bs, role) if role != "unknown" else bs + 2

            # find next im_end
            j = bs + 1
            while j < n and input_ids[j] != self.im_end:
                j += 1
            if j >= n:
                raise ValueError("Unmatched <|im_start|> (no following <|im_end|>)")

            be_token = j
            blocks.append(
                {
                    "role": role,
                    "block_start": bs,
                    "block_end_token": be_token,
                    "content_start": cs,
                    "content_end": be_token,
                }
            )
            i = be_token + 1

        if len(blocks) < 2:
            raise ValueError("Cannot find enough system/user blocks")

        return {"blocks": blocks}

    # ===================== ChatML formatting =====================

    def _format_conversation(self, question: str, answers: list[str], system_message: str | None = None) -> str:
        """
        Raw format includes \\n between blocks.
        IMPORTANT: user contains ONLY the question/input text (NO fixed prompt).
        system_message: optional per-sample override.
        """
        sys_msg = system_message if isinstance(system_message, str) and system_message.strip() else self.system_message
        usr_msg = (question or "").strip()  # allow empty

        conv = f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        conv += f"<|im_start|>user\n{usr_msg}<|im_end|>\n"
        for answer in answers:
            conv += f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        return conv

    # ===================== Core: y-aligned rebuild =====================

    def _rebuild_sequence_y_aligned(self, input_ids: list[int], raw: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
        """
        FINAL layout (no delimiter newline after im_end):
        system:     P_sys + sys_content + im_end
        user:       P_usr + usr_content + im_end
        assistant:  P_asst + (intro + solution) + im_end + trailing PAD

        trailing PAD is AFTER im_end, excluded from eligible cross-channel set via real_content_end

        IMPORTANT CHANGE:
        - NO truncation when exceeding max_seq_length
        - if it would exceed, raise ValueError (should be filtered upstream)
        """
        blocks = raw["blocks"]

        sys_block = next((b for b in blocks if b["role"] == "system"), None)
        usr_block = next((b for b in blocks if b["role"] == "user"), None)
        if sys_block is None or usr_block is None:
            raise ValueError("Missing system or user block")

        sys_content = input_ids[sys_block["content_start"] : sys_block["content_end"]]
        usr_content = input_ids[usr_block["content_start"] : usr_block["content_end"]]

        assistant_blocks = [b for b in blocks if b["role"] == "assistant"]
        if len(assistant_blocks) != self.fixed_heads:
            raise ValueError(f"Expected {self.fixed_heads} assistant blocks, got {len(assistant_blocks)}")

        raw_solutions: list[list[int]] = []
        for b in assistant_blocks:
            sol = input_ids[b["content_start"] : b["content_end"]]
            raw_solutions.append(sol)

        asst_real_contents: list[list[int]] = []
        asst_solution_start_offsets: list[int] = []
        for i, sol in enumerate(raw_solutions):
            if self.add_channel_intro:
                intro = self._encode_channel_intro(i)
                real = intro + sol
                sol_start = len(intro)
            else:
                real = sol
                sol_start = 0
            asst_real_contents.append(real)
            asst_solution_start_offsets.append(sol_start)

        max_real_len = max((len(x) for x in asst_real_contents), default=0)

        sys_len_rebuilt = len(self.system_prefix) + len(sys_content) + 1
        usr_len_rebuilt = len(self.user_prefix) + len(usr_content) + 1
        per_asst_prefix_and_end = len(self.assistant_prefix) + 1

        total_if_max = sys_len_rebuilt + usr_len_rebuilt + self.fixed_heads * (per_asst_prefix_and_end + max_real_len)
        if total_if_max > self.max_seq_length:
            raise ValueError(f"Exceed max_seq_length (no-trunc): total_if_max={total_if_max} > max_seq_length={self.max_seq_length}")

        target_real_len = max_real_len

        new_input_ids: list[int] = []
        all_heads: list[dict[str, Any]] = []
        assistant_head_indices: list[int] = []

        def add_block_system_user(role: str, prefix: list[int], content: list[int]):
            bs = len(new_input_ids)
            new_input_ids.extend(prefix)
            prefix_end = len(new_input_ids)

            content_start = len(new_input_ids)
            new_input_ids.extend(content)
            real_content_end = len(new_input_ids)

            im_end_pos = len(new_input_ids)
            new_input_ids.append(self.im_end)

            pad_start = len(new_input_ids)
            pad_end = len(new_input_ids)

            be = len(new_input_ids)

            h = {
                "role": role,
                "block_start": bs,
                "block_end": be,
                "prefix_start": bs,
                "prefix_end": prefix_end,
                "content_start": content_start,
                "content_end": real_content_end,
                "original_length": int(len(content)),
                "real_content_end": int(real_content_end),
                "im_end_pos": int(im_end_pos),
                "pad_start": int(pad_start),
                "pad_end": int(pad_end),
            }
            all_heads.append(h)

        def add_block_assistant(prefix: list[int], real_tokens: list[int], sol_start_offset: int):
            bs = len(new_input_ids)
            new_input_ids.extend(prefix)
            prefix_end = len(new_input_ids)

            content_start = len(new_input_ids)
            new_input_ids.extend(real_tokens)
            real_content_end = len(new_input_ids)

            im_end_pos = len(new_input_ids)
            new_input_ids.append(self.im_end)

            pad_len = max(0, target_real_len - len(real_tokens))
            pad_start = len(new_input_ids)
            if pad_len > 0:
                new_input_ids.extend([self.pad_token_id] * pad_len)
            pad_end = len(new_input_ids)

            be = len(new_input_ids)

            sol_start_offset = int(min(max(sol_start_offset, 0), max(0, len(real_tokens))))

            h = {
                "role": "assistant",
                "block_start": bs,
                "block_end": be,
                "prefix_start": bs,
                "prefix_end": prefix_end,
                "content_start": content_start,
                "content_end": real_content_end,
                "original_length": int(len(real_tokens)),
                "real_content_end": int(real_content_end),
                "im_end_pos": int(im_end_pos),
                "pad_start": int(pad_start),
                "pad_end": int(pad_end),
                "assistant_solution_start": int(content_start + sol_start_offset),
            }
            all_heads.append(h)

        add_block_system_user("system", self.system_prefix, sys_content)
        add_block_system_user("user", self.user_prefix, usr_content)

        for i, real in enumerate(asst_real_contents):
            add_block_assistant(self.assistant_prefix, real, asst_solution_start_offsets[i])
            assistant_head_indices.append(len(all_heads) - 1)

        if len(new_input_ids) > self.max_seq_length:
            raise ValueError(f"Final sequence length {len(new_input_ids)} exceeds max {self.max_seq_length}")

        boundaries = {
            "target_real_len": int(target_real_len),
            "all_heads": all_heads,
            "assistant_head_indices": assistant_head_indices,
            "include_im_end_in_cross_channel": bool(self.include_im_end_in_cross_channel),
            "allow_assistant_cross_channel": bool(self.allow_assistant_cross_channel),
            "add_channel_intro": bool(self.add_channel_intro),
            "pad_after_im_end": True,
        }

        return new_input_ids, boundaries

    # ===================== Labels =====================

    def _create_labels(self, input_ids: list[int], boundaries: dict[str, Any]) -> list[int]:
        """
        DEFAULT: supervise ALL assistant heads (solution tokens + <|im_end|>)
        If boundaries["supervise_head_indices"] exists: supervise only those.
        """
        labels = [-100] * len(input_ids)
        all_heads = boundaries["all_heads"]
        asst_his = boundaries["assistant_head_indices"]

        supervise_head_indices = boundaries.get("supervise_head_indices", None)
        if supervise_head_indices is None:
            supervise_set = set(range(len(asst_his)))
        else:
            supervise_set = set(int(x) for x in supervise_head_indices if isinstance(x, int) and 0 <= int(x) < len(asst_his))

        for local_asst_idx, hi in enumerate(asst_his):
            if local_asst_idx not in supervise_set:
                continue

            h = all_heads[hi]
            sol_start = int(h.get("assistant_solution_start", h["content_start"]))
            sol_end = min(int(h["real_content_end"]), len(input_ids))

            for i in range(sol_start, sol_end):
                labels[i] = input_ids[i]

            ie = int(h["im_end_pos"])
            if 0 <= ie < len(input_ids):
                labels[ie] = input_ids[ie]

        return labels

    # ===================== Position IDs =====================

    def _compute_local_y(self, i: int, h: dict[str, Any]) -> int:
        """
        local y inside a block:
          prefix:        y = 0..prefix_len-1
          real content:  y = prefix_len + (i - content_start)
          im_end:        y = prefix_len + real_content_len
          trailing pad:  y = prefix_len + real_content_len + 1 + (i - pad_start)
        """
        prefix_start = int(h["prefix_start"])
        prefix_end = int(h["prefix_end"])
        content_start = int(h["content_start"])
        content_end = int(h["content_end"])
        im_end_pos = int(h["im_end_pos"])
        pad_start = int(h.get("pad_start", im_end_pos + 1))
        pad_end = int(h.get("pad_end", pad_start))

        prefix_len = prefix_end - prefix_start
        real_len = max(0, content_end - content_start)

        if i < prefix_end:
            return i - prefix_start
        if i < content_end:
            return prefix_len + (i - content_start)
        if i == im_end_pos:
            return prefix_len + real_len
        if pad_start <= i < pad_end:
            return prefix_len + real_len + 1 + (i - pad_start)

        return prefix_len + real_len + 1 + max(0, i - pad_start)

    def _create_position_ids(self, input_ids: list[int], boundaries: dict[str, Any]):
        """
        Mapping:
          - system: cid=0, y from 0
          - user:   cid=1, y starts at y0 = len(system_block)
          - assistant h: cid=2+h, y starts at same y0
        """
        seq_len = len(input_ids)
        all_heads = boundaries["all_heads"]

        sys_hi = next((i for i, h in enumerate(all_heads) if h["role"] == "system"), None)
        usr_hi = next((i for i, h in enumerate(all_heads) if h["role"] == "user"), None)
        if sys_hi is None or usr_hi is None:
            raise ValueError("Missing system/user head for position ids")

        sys_h = all_heads[sys_hi]
        sys_bs, sys_be = int(sys_h["block_start"]), min(int(sys_h["block_end"]), seq_len)
        sys_len = max(0, sys_be - sys_bs)
        y0 = sys_len  # shared origin for user + all assistants

        assistant_his = [i for i, h in enumerate(all_heads) if h["role"] == "assistant"]
        assistant_cid_map = {hi: (2 + ai) for ai, hi in enumerate(assistant_his)}  # 2,3,4,...

        if not self.position_ids_2d:
            pos_ids = [0] * seq_len
            for hi, h in enumerate(all_heads):
                bs = int(h["block_start"])
                be = min(int(h["block_end"]), seq_len)

                offset = 0 if h["role"] == "system" else y0
                for i in range(bs, be):
                    y_local = self._compute_local_y(i, h)
                    y = int(offset) + int(y_local)
                    y = min(max(int(y), 0), self.max_position_embeddings - 1)
                    pos_ids[i] = y
            return pos_ids

        pos_ids = [[0, 0] for _ in range(seq_len)]
        for hi, h in enumerate(all_heads):
            bs = int(h["block_start"])
            be = min(int(h["block_end"]), seq_len)

            if h["role"] == "system":
                cid = 0
                offset = 0
            elif h["role"] == "user":
                cid = 1
                offset = y0
            else:
                cid = int(assistant_cid_map.get(hi, 2))
                offset = y0

            for i in range(bs, be):
                y_local = self._compute_local_y(i, h)
                y = int(offset) + int(y_local)
                y = min(max(int(y), 0), self.max_position_embeddings - 1)
                pos_ids[i] = [cid, y]

        return pos_ids

    # ===================== Attention Mask =====================

    def _create_attention_mask(self, input_ids: list[int], boundaries: dict[str, Any], position_ids) -> np.ndarray:
        """
        y-aligned cross-channel attention:

        Intra-block: strict causal.

        Cross-channel eligible keys:
          prefix + real_content (no trailing PAD) + im_end (optional)

        Cross-channel rule:
          for q in eligible, allow keys j in eligible if y_j < y_q.

        allow_assistant_cross_channel:
          - True: eligible keys include assistants too
          - False: eligible keys only system/user (assistants isolated cross-channel)
        """
        seq_len = len(input_ids)
        all_heads = boundaries["all_heads"]

        token_y = [0] * seq_len
        if self.position_ids_2d:
            for i in range(seq_len):
                token_y[i] = int(position_ids[i][1])
        else:
            for i in range(seq_len):
                token_y[i] = int(position_ids[i])

        cross_prefix = set()
        cross_real = set()
        cross_im_end = set()

        for h in all_heads:
            ps = int(h["prefix_start"])
            pe = min(int(h["prefix_end"]), seq_len)
            cs = int(h["content_start"])
            rce = min(int(h["real_content_end"]), seq_len)

            def add_block_to_cross():
                for ii in range(ps, pe):
                    cross_prefix.add(ii)
                for ii in range(cs, rce):
                    cross_real.add(ii)
                if self.include_im_end_in_cross_channel:
                    ie = int(h["im_end_pos"])
                    if 0 <= ie < seq_len:
                        cross_im_end.add(ie)

            if self.allow_assistant_cross_channel:
                add_block_to_cross()
            else:
                if h["role"] in ("system", "user"):
                    add_block_to_cross()

        eligible = cross_prefix.union(cross_real).union(cross_im_end)

        mask = np.zeros((seq_len, seq_len), dtype=bool)
        for i in range(seq_len):
            mask[i, i] = True

        # Intra-block strict causal
        for h in all_heads:
            bs = int(h["block_start"])
            be = min(int(h["block_end"]), seq_len)
            for q in range(bs, be):
                mask[q, bs : q + 1] = True

        # Cross-channel y-rule: y_j < y_q
        y_to_keys: dict[int, list[int]] = {}
        for j in eligible:
            y = token_y[j]
            y_to_keys.setdefault(y, []).append(j)

        ys_sorted = sorted(y_to_keys.keys())
        unique_yq = sorted({token_y[q] for q in eligible})

        yq_to_visible: dict[int, list[int]] = {}
        cumulative: list[int] = []
        y_ptr = 0
        for yq in unique_yq:
            while y_ptr < len(ys_sorted) and ys_sorted[y_ptr] < yq:
                cumulative.extend(y_to_keys[ys_sorted[y_ptr]])
                y_ptr += 1
            yq_to_visible[yq] = list(cumulative)

        for q in eligible:
            yq = token_y[q]
            visible_js = yq_to_visible.get(yq, [])
            for j in visible_js:
                mask[q, j] = True

        return mask.astype(np.float16)

    # ===================== Core processing =====================

    def _process_single_item(self, item: dict, line_num: int = 0) -> dict | None:
        try:
            # build raw ChatML input_ids
            if item.get("mode") == "segmented_channels":
                question = item["question"]
                segmented = item["segmented_channels"]
                num_channels = item["num_channels"]
                picked = item.get("picked_channels", None)
                heads_ids = self._build_heads_from_segmented_channels(segmented, num_channels, picked_channels=picked)
                input_ids = self._build_chatml_input_ids_from_heads(question, heads_ids)
            else:
                conversation = self._format_conversation(
                    item.get("question", ""),
                    item["answers"],
                    system_message=item.get("system_message", None),  # per-sample override
                )
                input_ids = self.tokenizer.encode(conversation, add_special_tokens=False)

            raw = self._find_blocks_raw(input_ids)

            # assistant blocks check
            asst_blocks = [b for b in raw["blocks"] if b["role"] == "assistant"]
            if len(asst_blocks) != self.fixed_heads:
                return None

            # length filtering based on raw assistant content
            head_lengths = [b["content_end"] - b["content_start"] for b in asst_blocks]
            is_valid, reason = self._validate_head_lengths(head_lengths, item.get("sample_id", ""))

            if not is_valid:
                item["filter_reason"] = reason
                item["head_lengths"] = head_lengths
                print(f"⚠️  Line {line_num} filtered: {reason}")
                return None

            # ----------------------------------------------------------
            # Hard filter for max_seq_length (NO TRUNCATION)
            # If the rebuilt y-aligned sequence would exceed max_seq_length,
            # drop the sample directly.
            # ----------------------------------------------------------
            sys_block = next((b for b in raw["blocks"] if b["role"] == "system"), None)
            usr_block = next((b for b in raw["blocks"] if b["role"] == "user"), None)
            if sys_block is None or usr_block is None:
                return None

            sys_content_len = sys_block["content_end"] - sys_block["content_start"]
            usr_content_len = usr_block["content_end"] - usr_block["content_start"]

            raw_asst_lens = head_lengths  # pre-intro lengths

            if self.add_channel_intro:
                intro_lens = [len(self._encode_channel_intro(i)) for i in range(self.fixed_heads)]
            else:
                intro_lens = [0] * self.fixed_heads

            max_real_len = max((raw_asst_lens[i] + intro_lens[i] for i in range(self.fixed_heads)), default=0)

            sys_len_rebuilt = len(self.system_prefix) + sys_content_len + 1  # + im_end
            usr_len_rebuilt = len(self.user_prefix) + usr_content_len + 1  # + im_end
            per_asst_prefix_and_end = len(self.assistant_prefix) + 1  # + im_end

            total_if_max = sys_len_rebuilt + usr_len_rebuilt + self.fixed_heads * (per_asst_prefix_and_end + max_real_len)

            if total_if_max > self.max_seq_length:
                if "filtered_by_seq_len" in self.filter_stats:
                    self.filter_stats["filtered_by_seq_len"] += 1
                item["filter_reason"] = f"Exceed max_seq_length(no-trunc): total_if_max={total_if_max} > {self.max_seq_length}"
                item["head_lengths"] = head_lengths
                print(f"⚠️  Line {line_num} filtered: {item['filter_reason']}")
                return None

            self.filter_stats["passed_filter"] += 1

            # rebuild (will NOT truncate; will raise if somehow exceeds)
            new_input_ids, boundaries = self._rebuild_sequence_y_aligned(input_ids, raw)

            # optional: supervise only selected heads if provided by item
            supervise = item.get("supervise_head_indices", None)
            if isinstance(supervise, list) and all(isinstance(x, int) for x in supervise):
                boundaries["supervise_head_indices"] = supervise

            labels = self._create_labels(new_input_ids, boundaries)
            position_ids = self._create_position_ids(new_input_ids, boundaries)
            attention_mask = self._create_attention_mask(new_input_ids, boundaries, position_ids)

            return {
                "input_ids": new_input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "num_heads": len(boundaries["all_heads"]),
                "seq_length": len(new_input_ids),
                "boundaries": boundaries,
            }

        except Exception as e:
            import traceback

            print(f"Error processing single item: {e}")
            print(f"Sample ID: {item.get('sample_id', 'unknown')}")
            print(traceback.format_exc())
            return None

    # ===================== Stats helpers =====================

    def _print_filter_statistics(self):
        if not self.enable_length_filter:
            return

        print(f"\n{'=' * 60}")
        print("Length Filter Statistics")
        print(f"{'=' * 60}")
        print(f"Passed filter: {self.filter_stats['passed_filter']}")

        total_filtered = (
            self.filter_stats["filtered_by_head_length"]
            + self.filter_stats["filtered_by_length_ratio"]
            + self.filter_stats["filtered_by_cv"]
            + self.filter_stats["filtered_by_padding"]
            + self.filter_stats["filtered_by_seq_len"]
        )

        if total_filtered > 0:
            print(f"\nFiltered samples: {total_filtered}")
            print(f"  - Max head length: {self.filter_stats['filtered_by_head_length']}")
            print(f"  - Length ratio: {self.filter_stats['filtered_by_length_ratio']}")
            print(f"  - CV: {self.filter_stats['filtered_by_cv']} (not enforced)")
            print(f"  - Max padding: {self.filter_stats['filtered_by_padding']}")
            print(f"  - Max seq length: {self.filter_stats['filtered_by_seq_len']}")

            total = self.filter_stats["passed_filter"] + total_filtered
            filter_rate = total_filtered / max(total, 1) * 100
            print(f"\nFilter rate: {filter_rate:.1f}%")
        print(f"{'=' * 60}")

    def _print_statistics(self, total_samples: int, output_dir: str):
        print(f"\n{'=' * 60}")
        print("Dataset Statistics")
        print(f"{'=' * 60}")
        print(f"Total samples saved: {total_samples}")
        print(f"Assistant heads per sample: {self.fixed_heads}")
        print(f"All channels per sample (system+user+assistants): {self.fixed_heads + 2}")
        print(f"Files created: {total_samples} .npz files + 1 index.json")

        total_size = 0
        for filename in os.listdir(output_dir):
            if filename.endswith(".npz") or filename.endswith(".json"):
                file_path = os.path.join(output_dir, filename)
                total_size += os.path.getsize(file_path)

        print(f"Total cache size: {total_size / (1024 * 1024):.2f} MB")
        print(f"Average size per sample: {total_size / max(total_samples, 1) / 1024:.2f} KB")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset: one sample per .npz (NO interaction, NO wait, NO fixed user prompt) + instruction/input/output support"
    )
    parser.add_argument("--input", required=False, nargs="+", help="Input JSONL file path(s)")
    parser.add_argument("--output", required=False, help="Output cache directory")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or name")
    parser.add_argument("--fixed-heads", type=int, required=True, help="Fixed number of assistant heads")

    # processing params
    parser.add_argument("--max-seq-length", type=int, default=32678)
    parser.add_argument("--truncation-strategy", choices=["random", "balanced"], default="random")
    parser.add_argument("--max-position-embeddings", type=int, default=32678)

    # position ids: default True, allow disabling
    parser.add_argument("--no-position-ids-2d", action="store_true", help="If set, use 1D position ids instead of 2D [cid,y].")

    parser.add_argument("--system-message", default="You are a helpful assistant.")

    # length filter
    parser.add_argument("--max-head-length", type=int, default=20000)
    parser.add_argument("--max-length-ratio", type=float, default=100.0)
    parser.add_argument("--max-cv", type=float, default=0.8)
    parser.add_argument("--max-padding-percent", type=float, default=100.0)
    parser.add_argument("--disable-length-filter", action="store_true")

    # segmented alignment
    parser.add_argument("--align-first-k-segments", type=int, default=1)

    # cross-channel im_end
    parser.add_argument("--exclude-im-end-cross-channel", action="store_true", help="Exclude <|im_end|> from cross-channel eligible set")

    # isolate assistant cross-channel
    parser.add_argument(
        "--disable-assistant-cross-channel",
        action="store_true",
        help="If set, assistants will NOT be eligible keys for cross-channel attention (assistants isolated).",
    )

    # assistant intro
    parser.add_argument("--disable-intro", action="store_true")
    parser.add_argument("--channel-intro-template", type=str, default="I am channel C{idx}.")

    # pick fixed_heads from larger pool
    parser.add_argument(
        "--head-indices", type=str, default=None, help="Comma-separated indices to pick from available outputs, e.g. '0,2,5'."
    )
    parser.add_argument(
        "--head-pick-strategy",
        type=str,
        default="first",
        choices=["first", "random", "sorted_key"],
        help="Pick strategy when available > fixed_heads and --head-indices is None.",
    )

    args = parser.parse_args()

    head_indices = None
    if args.head_indices:
        try:
            head_indices = [int(x.strip()) for x in args.head_indices.split(",") if x.strip() != ""]
        except Exception:
            raise ValueError(f"Invalid --head-indices: {args.head_indices}")

    processor = SingleFileDatasetProcessor(
        tokenizer_path=args.tokenizer,
        max_seq_length=args.max_seq_length,
        fixed_heads=args.fixed_heads,
        truncation_strategy=args.truncation_strategy,
        max_position_embeddings=args.max_position_embeddings,
        position_ids_2d=(not args.no_position_ids_2d),
        system_message=args.system_message,
        max_head_length=args.max_head_length,
        max_length_ratio=args.max_length_ratio,
        max_cv=args.max_cv,
        max_padding_percent=args.max_padding_percent,
        enable_length_filter=(not args.disable_length_filter),
        align_first_k_segments=args.align_first_k_segments,
        include_im_end_in_cross_channel=(not args.exclude_im_end_cross_channel),
        allow_assistant_cross_channel=(not args.disable_assistant_cross_channel),
        add_channel_intro=(not args.disable_intro),
        channel_intro_template=args.channel_intro_template,
        head_indices=head_indices,
        head_pick_strategy=args.head_pick_strategy,
    )

    if args.input and args.output:
        out = processor.process_jsonl_files(jsonl_paths=args.input, output_dir=args.output)
        print("\n✓ Processing completed!")
        print(f"Files saved to: {out}")
    else:
        raise ValueError("You must provide --input and --output to process dataset.")


if __name__ == "__main__":
    main()
