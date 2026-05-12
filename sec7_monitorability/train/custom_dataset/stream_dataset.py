"""
StreamDataset and StreamDataCollator for multi-channel parallel stream data.

Flattening: row-by-row into 1D sequence using only active channels:
    [r0_c0, r0_c1, ..., r1_c0, r1_c1, ...]

Attention: block-causal with self-diagonal. Same-row tokens cannot see each other.
    Token i attends to j iff: row(j) < row(i) OR j == i

Loss: shift-by-C (next-row same-channel prediction).
    logits[r*C+c] predicts input_ids[(r+1)*C+c]
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset


class StreamDataset(Dataset):
    """Dataset for multi-channel parallel stream data.

    Args:
        data_path: Directory containing dataset.jsonl
        active_channels: List of original channel indices to include (e.g. [0,1] for user+output).
            Defaults to all 10 channels.
        max_seq_length: Maximum flattened sequence length (truncated to full rows).
        prefix_rows: Number of leading rows to mask from loss (conditioning-only).
            0 = compute loss on all rows. Used with both single-sample and packed modes.
        pack_samples: Max number of extra samples to pack before the target sample.
            A random count in [1, pack_samples] is chosen per item. Combined with
            prefix_rows to control which rows get loss. 0 = disabled (single sample).
        pack_trim: Where to trim when packed samples exceed max_seq_length.
            "top" (default) = trim earliest rows from prefix, keeping target intact.
            "bottom" = trim latest rows from end, keeping conversation starts intact.
    """

    def __init__(
        self,
        data_path,
        active_channels=None,
        max_seq_length=None,
        prefix_rows=0,
        pack_samples=0,
        pack_trim="top",
    ):
        self.active_channels = active_channels if active_channels is not None else list(range(10))
        self.num_channels = len(self.active_channels)
        self.max_seq_length = max_seq_length
        self.prefix_rows = prefix_rows
        self.pack_samples = pack_samples
        self.pack_trim = pack_trim
        self.epoch = 0
        self._pack_pool = None  # set via restrict_pack_pool() after train/val split

        jsonl_path = os.path.join(data_path, "dataset.jsonl")
        self.samples = []
        n_skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                if sample.get("num_rows", 0) < 1:
                    n_skipped += 1
                    continue
                self.samples.append(sample)
        if n_skipped > 0:
            print(f"StreamDataset: skipped {n_skipped} samples with num_rows < 1")

    def __len__(self):
        return len(self.samples)

    def _extract_flat_ids(self, sample, max_rows=None):
        """Extract flat token ids for active channels from a raw sample."""
        num_rows = sample["num_rows"]
        token_ids = sample["token_ids"]  # [C_original][R]
        if max_rows is not None:
            num_rows = min(num_rows, max_rows)
        flat_ids = []
        for r in range(num_rows):
            for c in self.active_channels:
                flat_ids.append(token_ids[c][r])
        return flat_ids, num_rows

    def restrict_pack_pool(self, valid_indices):
        """Restrict packing to only draw from these sample indices.

        Must be called after train/val split to prevent val samples from
        leaking into training batches via packing or prefix_rows.
        """
        self._pack_pool = list(valid_indices)

    def set_epoch(self, epoch):
        """Set current epoch for epoch-dependent packing randomization."""
        self.epoch = epoch

    def __getitem__(self, idx):
        sample = self.samples[idx]
        C = self.num_channels

        # Multi-sample packing: pack 1-N random complete samples as prefix
        if self.pack_samples > 0 and len(self.samples) > 1:
            return self._getitem_packed(idx, sample, C)

        # Reserve space for prefix when computing max rows
        prefix_tokens = self.prefix_rows * C if self.prefix_rows > 0 else 0
        if self.max_seq_length:
            max_rows = (self.max_seq_length - prefix_tokens) // C
        else:
            max_rows = None

        flat_ids, num_rows = self._extract_flat_ids(sample, max_rows)

        # Prepend terminal rows from a deterministically random other sample
        prefix_length = 0
        if self.prefix_rows > 0 and len(self.samples) > 1:
            rng = random.Random(idx + self.epoch * len(self.samples))
            if self._pack_pool is not None:
                pool = [i for i in self._pack_pool if i != idx]
                other_idx = rng.choice(pool)
            else:
                other_idx = rng.randrange(len(self.samples) - 1)
                if other_idx >= idx:
                    other_idx += 1

            other = self.samples[other_idx]
            other_num_rows = other["num_rows"]
            n_prefix = min(self.prefix_rows, other_num_rows)
            start_row = other_num_rows - n_prefix

            prefix_flat = []
            for r in range(start_row, other_num_rows):
                for c in self.active_channels:
                    prefix_flat.append(other["token_ids"][c][r])

            flat_ids = prefix_flat + flat_ids
            prefix_length = len(prefix_flat)
            num_rows += n_prefix

        input_ids = torch.tensor(flat_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "seq_length": len(flat_ids),
            "num_rows": num_rows,
            "prefix_length": prefix_length,
        }

    def _getitem_packed(self, idx, target_sample, C):
        """Pack 1-N random samples before the target sample.

        All packed samples contribute rows. prefix_rows controls the loss mask:
          prefix_rows=0  → loss on ALL rows (all packed samples)
          prefix_rows=N  → first N rows masked from loss

        If the total exceeds max_seq_length, rows are trimmed from the top
        (earliest rows dropped first), keeping the target sample intact.
        """
        rng = random.Random(idx + self.epoch * len(self.samples))
        n_extra = rng.randint(0, self.pack_samples)

        # Extract target sample (no row limit yet — we'll enforce budget after)
        target_ids, target_rows = self._extract_flat_ids(target_sample)

        # If max_seq_length set, cap target rows to fit within budget
        if self.max_seq_length:
            max_target_rows = self.max_seq_length // C
            if target_rows > max_target_rows:
                target_ids = target_ids[: max_target_rows * C]
                target_rows = max_target_rows

        # Pick n_extra random other samples (without replacement, excluding idx)
        # Use restricted pool if set (prevents val leak into training)
        if self._pack_pool is not None:
            pool = [i for i in self._pack_pool if i != idx]
        else:
            pool = list(range(len(self.samples)))
            pool.remove(idx)
        if n_extra > len(pool):
            n_extra = len(pool)
        extra_indices = rng.sample(pool, n_extra)

        # Extract all rows from each extra sample
        extra_flat = []
        extra_total_rows = 0
        for pi in extra_indices:
            p_ids, p_rows = self._extract_flat_ids(self.samples[pi])
            extra_flat.extend(p_ids)
            extra_total_rows += p_rows

        # Truncate if total exceeds max_seq_length
        total_tokens = len(extra_flat) + len(target_ids)
        if self.max_seq_length and total_tokens > self.max_seq_length:
            excess = total_tokens - self.max_seq_length
            rows_to_trim = (excess + C - 1) // C  # ceil division
            if self.pack_trim == "top":
                # Trim earliest rows from prefix — keeps target intact
                extra_flat = extra_flat[rows_to_trim * C :]
                extra_total_rows -= rows_to_trim
            else:
                # Trim latest rows from end — keeps conversation starts intact
                total_rows = extra_total_rows + target_rows
                keep_rows = total_rows - rows_to_trim
                extra_flat = extra_flat[: max(0, keep_rows - target_rows) * C]
                extra_total_rows = max(0, keep_rows - target_rows)
                # Also trim target if extras are fully consumed and still over budget
                remaining = keep_rows - extra_total_rows
                if remaining < target_rows:
                    target_ids = target_ids[: remaining * C]
                    target_rows = remaining

        flat_ids = extra_flat + target_ids
        num_rows = extra_total_rows + target_rows

        # prefix_rows controls loss masking (0 = loss on everything)
        prefix_length = min(self.prefix_rows, num_rows) * C

        input_ids = torch.tensor(flat_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "seq_length": len(flat_ids),
            "num_rows": num_rows,
            "prefix_length": prefix_length,
        }

    def get_raw_sample(self, idx):
        """Get a sample without prefix rows (for eval)."""
        sample = self.samples[idx]
        C = self.num_channels
        max_rows = self.max_seq_length // C if self.max_seq_length else None
        flat_ids, num_rows = self._extract_flat_ids(sample, max_rows)
        return {
            "input_ids": torch.tensor(flat_ids, dtype=torch.long),
            "seq_length": len(flat_ids),
            "num_rows": num_rows,
            "prefix_length": 0,
        }

    def get_context_sample(self, idx, n_context, context_pool=None):
        """Get a sample prepended with exactly *n_context* complete other samples.

        Used for eval so we can measure how well the model uses prior context.
        Context samples are chosen deterministically from idx.
        Loss should be computed only on the target sample (prefix_length marks the
        boundary).

        Args:
            context_pool: Indices to draw context from (e.g. val indices only).
                Keeps train/val completely separated during eval.
        """
        sample = self.samples[idx]
        C = self.num_channels

        # Target sample (no row limit yet)
        target_ids, target_rows = self._extract_flat_ids(sample)
        if self.max_seq_length:
            max_target = self.max_seq_length // C
            if target_rows > max_target:
                target_ids = target_ids[: max_target * C]
                target_rows = max_target

        # Pick n_context deterministic other samples from context_pool
        rng = random.Random(idx)
        if context_pool is not None:
            pool = [i for i in context_pool if i != idx]
        else:
            pool = list(range(len(self.samples)))
            pool.remove(idx)
        n = min(n_context, len(pool))
        ctx_indices = rng.sample(pool, n)

        ctx_flat = []
        ctx_rows = 0
        for ci in ctx_indices:
            c_ids, c_rows = self._extract_flat_ids(self.samples[ci])
            ctx_flat.extend(c_ids)
            ctx_rows += c_rows

        # Trim context from the top if total exceeds max_seq_length
        total = len(ctx_flat) + len(target_ids)
        if self.max_seq_length and total > self.max_seq_length:
            excess = total - self.max_seq_length
            rows_to_trim = (excess + C - 1) // C
            ctx_flat = ctx_flat[rows_to_trim * C :]
            ctx_rows -= rows_to_trim

        flat_ids = ctx_flat + target_ids
        num_rows = ctx_rows + target_rows
        prefix_length = len(ctx_flat)

        return {
            "input_ids": torch.tensor(flat_ids, dtype=torch.long),
            "seq_length": len(flat_ids),
            "num_rows": num_rows,
            "prefix_length": prefix_length,
        }


class StreamEvalDataset(Dataset):
    """Eval-only view of a StreamDataset that always returns unpacked samples.

    Supports an optional n_context parameter to prepend exactly N complete
    samples as context (for measuring context utilization).
    """

    def __init__(self, base_dataset, indices, n_context=0):
        self.base = base_dataset
        self.indices = list(indices)
        self.n_context = n_context

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        if self.n_context > 0:
            return self.base.get_context_sample(real_idx, self.n_context, context_pool=self.indices)
        return self.base.get_raw_sample(real_idx)


class StreamDataCollator:
    """Collator that builds block-causal attention masks and shift-by-C labels.

    Output dict:
        input_ids:      [B, S_max]              padded token ids
        labels:         [B, S_max]              shift-by-C targets, -100 for last row + padding + prefix
        position_ids:   [B, S_max]              row index (i // C)
        channel_ids:    [B, S_max]              channel index (i % C)
        attention_mask:  [B, 1, S_max, S_max]   block-causal additive mask
    """

    NEG_INF = -1e4

    def __init__(
        self,
        pad_token_id,
        num_channels=10,
        attention_mask_type="block_causal",
        silence_token_id=None,
    ):
        self.pad_token_id = pad_token_id
        self.num_channels = num_channels
        self.attention_mask_type = attention_mask_type
        self.silence_token_id = silence_token_id

    def __call__(self, features):
        B = len(features)
        C = self.num_channels
        S_max = max(f["seq_length"] for f in features)

        input_ids = torch.full((B, S_max), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, S_max), -100, dtype=torch.long)
        position_ids = torch.zeros((B, S_max), dtype=torch.long)
        channel_ids = torch.zeros((B, S_max), dtype=torch.long)
        mask = torch.full((B, 1, S_max, S_max), self.NEG_INF, dtype=torch.float32)

        for b in range(B):
            S = features[b]["seq_length"]
            ids = features[b]["input_ids"]
            input_ids[b, :S] = ids

            # Position IDs = row index, Channel IDs = channel index
            pos = torch.arange(S)
            position_ids[b, :S] = pos // C
            channel_ids[b, :S] = pos % C

            # Attention mask: block_causal, causal, or block_causal_skip_silence
            rows = pos // C  # [S]
            if self.attention_mask_type == "causal":
                # Standard causal: attend to all prior positions + same row + self
                can_attend = rows.unsqueeze(0) <= rows.unsqueeze(1)
            else:
                # Block-causal (default and skip_silence base): strictly prior rows + self
                # can_attend[i, j] = True iff row(j) < row(i) OR j == i
                can_attend = (rows.unsqueeze(0) < rows.unsqueeze(1)) | torch.eye(S, dtype=torch.bool)
            mask[b, 0, :S, :S] = torch.where(can_attend, 0.0, self.NEG_INF)

            # Skip-silence: mask silence positions as keys (nobody attends to them)
            # Silence tokens can still attend to non-silence keys (as queries)
            # Self-diagonal restored below for numerical stability
            if self.attention_mask_type == "block_causal_skip_silence" and self.silence_token_id is not None:
                silence_cols = ids[:S] == self.silence_token_id  # [S]
                mask[b, 0, :S, :S].masked_fill_(silence_cols.unsqueeze(0), self.NEG_INF)

            # Labels: shift by C (next-row same-channel prediction)
            if S > C:
                labels[b, : S - C] = ids[C:]

            # Mask prefix rows — from another sample, no meaningful prediction target
            prefix_len = features[b].get("prefix_length", 0)
            if prefix_len > 0:
                labels[b, :prefix_len] = -100

        # Set diagonal to 0 for ALL positions (including padding) to prevent
        # all-masked rows from causing NaN in softmax
        diag_idx = torch.arange(S_max)
        mask[:, 0, diag_idx, diag_idx] = 0.0

        attention_mask = {"full_attention": mask, "sliding_attention": mask}

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "channel_ids": channel_ids,
            "attention_mask": attention_mask,
        }
