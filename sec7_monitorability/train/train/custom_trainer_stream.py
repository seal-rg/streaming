"""
Unified stream trainer for 10-channel parallel prediction.

Computes shift-by-10 cross-entropy loss (next-row same-channel prediction)
with DDP-safe global token count reduction and per-group learning rates
(backbone vs channel_embedding).

Optionally applies LongCE importance weighting (controlled by enable_longce).
When LongCE is disabled (the default), this is a standard CE trainer.

Per-channel loss is always logged under loss/{channel_name}.
"""

import logging
import math
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
import trl
from streamweaver.evaluation.metrics import compute_all_metrics  # type: ignore
from streamweaver.parsing.table_parser import StreamTable  # type: ignore
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

logger = logging.getLogger(__name__)

CHANNEL_NAMES = [
    "user",
    "output",
    "analytical",
    "skeptical",
    "intuitive",
    "between",
    "curious",
    "void",
    "instinct",
    "synthesis",
]

SILENCE_TOKEN_ID = 481  # Qwen3 default; overridden per-tokenizer via self.silence_token_id


def _lcs_length(a, b):
    """Longest common subsequence length between two lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _rouge_l_f1(reference, hypothesis):
    """ROUGE-L F1 between two token lists."""
    if not reference or not hypothesis:
        return 0.0
    lcs = _lcs_length(reference, hypothesis)
    if lcs == 0:
        return 0.0
    p = lcs / len(hypothesis)
    r = lcs / len(reference)
    return 2 * p * r / (p + r)


def compute_generation_metrics(
    generated_rows,
    window_size=10,
    silence_token_id=SILENCE_TOKEN_ID,
    all_rows=None,
    channel_names=None,
):
    """Repetition and stability metrics from generated token ID grid.

    Args:
        generated_rows: list of lists, shape [R, C], token IDs
        window_size: rows per window for degradation tracking
        all_rows: optional full grid (prefill + generated) for mirroring computation

    Returns:
        dict of metric_name -> value
    """
    if not generated_rows:
        return {}
    R = len(generated_rows)
    C = len(generated_rows[0])
    metrics = {}

    # Distinct trigrams / total trigrams per stream (pooled)
    total_distinct, total_count = 0, 0
    for c in range(C):
        toks = [generated_rows[r][c] for r in range(R) if generated_rows[r][c] != silence_token_id]
        if len(toks) < 3:
            continue
        trigrams = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
        total_distinct += len(set(trigrams))
        total_count += len(trigrams)
    if total_count > 0:
        metrics["distinct_trigram"] = total_distinct / total_count

    # Windowed: detect if repetition gets worse over time
    n_windows = R // window_size
    if n_windows >= 2:
        window_ratios = []
        for w in range(n_windows):
            s, e = w * window_size, (w + 1) * window_size
            wd, wc = 0, 0
            for c in range(C):
                toks = [generated_rows[r][c] for r in range(s, e) if generated_rows[r][c] != silence_token_id]
                if len(toks) < 3:
                    continue
                trigrams = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
                wd += len(set(trigrams))
                wc += len(trigrams)
            if wc > 0:
                window_ratios.append(wd / wc)
        if len(window_ratios) >= 2:
            metrics["repetition_degradation"] = window_ratios[0] - window_ratios[-1]

    # Channel mirroring: ROUGE-L between user (col 0) and each other column
    # Uses all_rows (prefill + generated) to capture user input tokens
    grid = all_rows if all_rows is not None else generated_rows
    user_col = [row[0] for row in grid]
    user_active = [i for i, t in enumerate(user_col) if t != silence_token_id]
    if user_active:
        user_toks = [user_col[i] for i in user_active]
        rl_sum = 0.0
        rl_count = 0
        for c in range(1, C):
            ch_toks = [grid[i][c] for i in user_active if grid[i][c] != silence_token_id]
            rl = _rouge_l_f1(user_toks, ch_toks)
            name = channel_names[c] if channel_names and c < len(channel_names) else f"c{c}"
            metrics[f"mirror_{name}"] = rl
            rl_sum += rl
            rl_count += 1
        if rl_count > 0:
            metrics["mirror_avg"] = rl_sum / rl_count

    return metrics


class StreamTrainer(trl.SFTTrainer):
    """c-channel parallel stream trainer with optional LongCE importance weighting."""

    def __init__(self, *args, **kwargs):
        # Optimizer LRs
        self.backbone_lr = float(kwargs.pop("backbone_lr", 1e-5))
        self.channel_embedding_lr = float(kwargs.pop("channel_embedding_lr", 1e-4))
        self.cooldown_ratio = float(kwargs.pop("cooldown_ratio", 0.0))
        self.optim_8bit = bool(kwargs.pop("optim_8bit", False))

        # Regularization
        self.label_smoothing = float(kwargs.pop("label_smoothing", 0.0))
        self.z_loss_weight = float(kwargs.pop("z_loss_weight", 0.0))

        # LongCE config (disabled by default = baseline behavior)
        self.enable_longce = bool(kwargs.pop("enable_longce", False))
        self.longce_prob = float(kwargs.pop("longce_prob", 1.0))
        self.longce_gamma = float(kwargs.pop("longce_gamma", 50.0))
        self.longce_warmup_steps = int(kwargs.pop("longce_warmup_steps", 0))
        self.longce_ramp_steps = int(kwargs.pop("longce_ramp_steps", 0))
        self.longce_num_channels_per_step = int(kwargs.pop("longce_num_channels_per_step", 3))
        self.longce_uniform_mix = float(kwargs.pop("longce_uniform_mix", 0.0))
        # Selection mode — only relevant when self_only_mode == "per_channel"
        # below. "random" (default) is uniform without replacement; "cycle"
        # deterministically round-robins so each channel gets equal LongCE
        # coverage. Becomes moot in "global" self-only mode, which covers
        # every channel every step.
        self.longce_selection_mode = str(kwargs.pop("longce_selection_mode", "random"))
        self._longce_pool: list[int] = []

        # Self-only forward mode:
        #   "per_channel" (default / legacy): run one self-only forward per
        #     selected channel (expensive at high num_channels_per_step,
        #     stochastic coverage at low values).
        #   "global": run ONE self-only forward with an attention mask that
        #     blocks ALL cross-column attention simultaneously. Produces
        #     ce_self for every token at once. Every channel gets a real
        #     LSD weight every step. Strictly better coverage at strictly
        #     lower compute (2 forwards regardless of C or num_channels_per_step
        #     vs. 1 + num_channels_per_step for per_channel mode). Correct
        #     for pure-transformer AND hybrid-with-column-DeltaNet models
        #     because the attention mask handles attention layers and
        #     column-mode DeltaNet already maintains per-column state
        #     independence. NOT correct for row_boundary-DeltaNet (rare) —
        #     DeltaNet state still leaks across columns in that mode.
        self.longce_self_only_mode = str(kwargs.pop("longce_self_only_mode", "per_channel"))

        # Isoft options
        self.isoft_positive_only = bool(kwargs.pop("isoft_positive_only", False))
        self.isoft_floor_one = bool(kwargs.pop("isoft_floor_one", False))
        self.isoft_per_channel_mean1 = bool(kwargs.pop("isoft_per_channel_mean1", False))

        # Mask config
        self.num_channels = int(kwargs.pop("num_channels", 10))
        self.neg_inf = float(kwargs.pop("neg_inf", -1e4))

        # Self-only behavior
        self.self_only_drop_input_ids = bool(kwargs.pop("self_only_drop_input_ids", False))

        # Eval config
        self.eval_num_samples = int(kwargs.pop("eval_num_samples", 8))
        self.eval_prefill_frac = float(kwargs.pop("eval_prefill_frac", 0.25))
        self.eval_gen_rows = int(kwargs.pop("eval_gen_rows", 100))
        self.eval_temperature = float(kwargs.pop("eval_temperature", 0.8))
        self.eval_top_p = float(kwargs.pop("eval_top_p", 0.9))
        self.eval_top_k = int(kwargs.pop("eval_top_k", 0))

        # Context eval datasets (for measuring context utilization)
        self.eval_context_datasets = kwargs.pop("eval_context_datasets", None) or []

        # Feature flags
        self.mask_user_loss = bool(kwargs.pop("mask_user_loss", False))
        self.user_channel_idx = kwargs.pop("user_channel_idx", None)
        self.output_channel_idx = kwargs.pop("output_channel_idx", None)
        self.channel_names = list(kwargs.pop("channel_names", CHANNEL_NAMES[: self.num_channels]))
        self.silence_token_id = int(kwargs.pop("silence_token_id", SILENCE_TOKEN_ID))
        self.attention_mask_type = str(kwargs.pop("attention_mask_type", "block_causal"))

        super().__init__(*args, **kwargs)

        # Throughput tracking
        self._step_start_time = None

    # ------------------------------------------------------------------
    # LongCE schedule
    # ------------------------------------------------------------------
    def _longce_prob_now(self) -> float:
        if not self.enable_longce:
            return 0.0
        step = int(getattr(self.state, "global_step", 0))
        if step < self.longce_warmup_steps:
            return 0.0
        if self.longce_ramp_steps <= 0:
            return float(self.longce_prob)
        t = (step - self.longce_warmup_steps) / float(self.longce_ramp_steps)
        t = max(0.0, min(1.0, t))
        return float(self.longce_prob) * t

    def _select_longce_channels(self, num_ch: int) -> list[int]:
        """Select `num_ch` channels for LongCE weighting this forward pass.

        Two modes:
          - "random": `random.sample` without replacement (original
            behavior; high variance in per-channel coverage over time).
          - "cycle": deterministic round-robin. Maintains a per-instance
            shuffled pool of channels; each call consumes `num_ch` from
            the pool and reshuffles on exhaustion. Every C/num_ch
            consecutive LongCE-active forwards (one per call) visit every
            channel exactly once. This eliminates variance in coverage
            and makes the "LongCE as regularizer" interpretation actually
            hold — every channel receives importance-weighting pressure
            at the same rate.

        Cycle advances ONLY when this method is called — skip steps
        (do_longce=False) do NOT consume cycle slots, preserving the
        "each channel visited exactly once per cycle" invariant across
        warmup / probabilistic skipping.

        Per-rank independence: each rank maintains its own pool, so a
        single optimizer step with R ranks and K num_ch covers
        min(C, R*K*accum_steps) channels at a minimum, with ranks
        naturally stratified by their independently-shuffled orderings.

        Cycle state is NOT persisted across checkpoints — on resume,
        the pool rebuilds fresh. Boundary effect is at most one
        partial-cycle per resume, negligible over long runs.

        Returns:
            Sorted list of channel indices. Length == min(num_ch, C).
            Empty list if num_ch <= 0.
        """
        C = self.num_channels
        num_ch = min(num_ch, C)
        if num_ch <= 0:
            return []

        if self.longce_selection_mode == "random":
            return sorted(random.sample(range(C), num_ch))

        # Cycle mode. Drain from the pool; reshuffle when exhausted.
        # If num_ch > pool remainder, drain what's there and continue
        # from a fresh shuffle — never leaves a partial remainder
        # unvisited, never skips forward into the next cycle except as
        # needed to fulfill num_ch.
        selected: list[int] = []
        while len(selected) < num_ch:
            if not self._longce_pool:
                pool = list(range(C))
                random.shuffle(pool)
                self._longce_pool = pool
            take = min(num_ch - len(selected), len(self._longce_pool))
            selected.extend(self._longce_pool[:take])
            self._longce_pool = self._longce_pool[take:]
        return sorted(selected)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_drop_token_id(self) -> int:
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            pid = getattr(tok, "pad_token_id", None)
            if pid is not None:
                return int(pid)
            eid = getattr(tok, "eos_token_id", None)
            if eid is not None:
                return int(eid)
        return 0

    def _log_per_channel_loss(self, per_token_loss, channel_ids, valid):
        """Log per-channel CE loss, globally reduced with a single all-reduce."""
        C = self.num_channels
        device = per_token_loss.device
        # Stack per-channel sums into one tensor: [loss_ch0, tok_ch0, loss_ch1, tok_ch1, ...]
        stats = torch.zeros(C * 2, device=device, dtype=torch.float32)
        for c in range(C):
            ch_mask = (channel_ids == c) & valid
            stats[c * 2] = (per_token_loss * ch_mask.float()).sum()
            stats[c * 2 + 1] = ch_mask.sum().float()
        # Single all-reduce
        g_stats = self.accelerator.reduce(stats, reduction="sum")
        log_dict = {}
        total_loss = 0.0
        total_tok = 0.0
        for c in range(C):
            g_loss = g_stats[c * 2].item()
            g_tok = g_stats[c * 2 + 1].item()
            total_loss += g_loss
            total_tok += g_tok
            if g_tok > 0:
                name = self.channel_names[c] if c < len(self.channel_names) else f"ch{c}"
                log_dict[f"loss/{name}"] = g_loss / g_tok
        if total_tok > 0:
            log_dict["loss/avg"] = total_loss / total_tok
        self.log(log_dict)

    def _log_weight_norms(self):
        """Log RMS parameter norms for backbone and channel embedding."""
        model = self.accelerator.unwrap_model(self.model)
        bb_ss, bb_n = 0.0, 0
        ce_ss, ce_n = 0.0, 0
        for name, p in model.named_parameters():
            sq_sum = p.detach().norm().float().pow(2)
            numel = p.numel()
            if "channel_embedding" in name:
                ce_ss += sq_sum
                ce_n += numel
            else:
                bb_ss += sq_sum
                bb_n += numel
        metrics = {"norms/backbone": (bb_ss / bb_n).item() ** 0.5}  # type: ignore
        if ce_n > 0:
            metrics["norms/channel_embedding"] = (ce_ss / ce_n).item() ** 0.5  # type: ignore
        self.log(metrics)

    def _log_nonsilence_loss(self, per_token_loss, channel_ids, labels, valid):
        """Log user/output channel loss with silence target tokens masked out."""
        stats = torch.zeros(4, device=per_token_loss.device, dtype=torch.float32)
        nonsilence = labels != self.silence_token_id
        if self.user_channel_idx is not None:
            mask = (channel_ids == self.user_channel_idx) & valid & nonsilence
            stats[0] = (per_token_loss * mask.float()).sum()
            stats[1] = mask.sum().float()
        if self.output_channel_idx is not None:
            mask = (channel_ids == self.output_channel_idx) & valid & nonsilence
            stats[2] = (per_token_loss * mask.float()).sum()
            stats[3] = mask.sum().float()
        g = self.accelerator.reduce(stats, reduction="sum")
        log_dict = {}
        if self.user_channel_idx is not None and g[1] > 0:
            log_dict["loss/user_nonsilence"] = float((g[0] / g[1]).item())
        if self.output_channel_idx is not None and g[3] > 0:
            log_dict["loss/output_nonsilence"] = float((g[2] / g[3]).item())
        if log_dict:
            self.log(log_dict)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _log_throughput(self, num_tokens: int):
        """Log raw tokens/sec throughput."""
        now = time.monotonic()
        if self._step_start_time is not None:
            elapsed = now - self._step_start_time
            if elapsed > 0:
                self.log({"throughput/tokens_per_sec": num_tokens / elapsed})
        self._step_start_time = now

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = inputs["position_ids"]
        channel_ids = inputs["channel_ids"]
        labels = inputs["labels"]

        B, S = input_ids.shape
        device = input_ids.device
        C = self.num_channels

        # Decide whether to run LongCE this step
        p_now = self._longce_prob_now()
        do_longce = p_now > 0 and random.random() < p_now

        # ---- FULL forward (grad) ----
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,
            channel_ids=channel_ids,
        )
        logits = outputs.logits  # [B, S, V]
        valid = labels != -100

        if not do_longce:
            # ---- Standard CE path ----
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).view_as(labels)

            # Optionally exclude user channel from gradient
            if self.mask_user_loss and self.user_channel_idx is not None:
                train_valid = valid & (channel_ids != self.user_channel_idx)
                loss_sum = (per_token_loss * train_valid.float()).sum()
                num_tokens = train_valid.sum().float()
            else:
                loss_sum = per_token_loss.sum()
                num_tokens = valid.sum().float()

            with torch.no_grad():
                global_tokens = self.accelerator.reduce(num_tokens, reduction="sum").clamp_min(1.0)  # type: ignore
            world_size = float(getattr(self.accelerator, "num_processes", 1))
            loss = loss_sum * world_size / global_tokens

            # Z-loss: penalty on logsumexp(logits) to stabilize logit scale.
            # Chunked + checkpointed to avoid materializing the full float32
            # [B,S,V] tensor (~12 GB).  Each chunk upcasts only [B,chunk,V]
            # during forward (freed immediately) and recomputes it in backward.
            if self.z_loss_weight > 0:
                _zloss_chunk = 256
                z_parts = []
                for _i in range(0, S, _zloss_chunk):
                    z_parts.append(
                        torch.utils.checkpoint.checkpoint(  # type: ignore
                            lambda x: torch.logsumexp(x.float(), dim=-1),
                            logits[:, _i : _i + _zloss_chunk, :],
                            use_reentrant=False,
                        )
                    )
                z = torch.cat(z_parts, dim=1)
                z_loss = self.z_loss_weight * (z[valid] ** 2).mean()
                loss = loss + z_loss

            with torch.no_grad():
                self._log_per_channel_loss(per_token_loss.detach(), channel_ids, valid)
                self._log_nonsilence_loss(per_token_loss.detach(), channel_ids, labels, valid)
                self._log_weight_norms()
                self._log_throughput(int(valid.sum().item()) * int(getattr(self.accelerator, "num_processes", 1)))

            return (loss, outputs) if return_outputs else loss

        # ---- LongCE path (memory-efficient) ----
        # Instead of materializing the full [B,S,V] log_softmax (~6 GB for 8B),
        # use F.cross_entropy which fuses log_softmax+nll_loss in a single kernel,
        # producing only per-token scalars [B,S].
        gamma = float(self.longce_gamma)
        tgt = labels.clamp(min=0)

        # Per-token CE with gradient: [B, S], fused — no [B,S,V] intermediate
        ce_full = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt.view(-1),
            reduction="none",
        ).view(B, S)
        ce_full = ce_full * valid.float()  # zero out invalid positions

        # Extract the raw mask tensor from dict
        if isinstance(attention_mask, dict):
            base_mask = attention_mask["full_attention"]
        else:
            base_mask = attention_mask

        col_channels = torch.arange(S, device=device) % C
        diag_idx = torch.arange(S, device=device)
        drop_token_id = self._get_drop_token_id()

        global_mode = (self.longce_self_only_mode == "global")

        # Compute self-only cross-entropy per token. Two branches:
        #   global: one forward with a mask that blocks ALL cross-column
        #     attention. Produces ce_self valid for every channel at once.
        #     ce_self_by_ch[c] aliases the same [B,S] tensor for every c.
        #   per_channel: N forwards for the N selected channels, each with
        #     its own mask. Legacy behavior.
        if global_mode:
            # All channels covered this forward — no selection needed.
            selected_channels = list(range(C))
            selected_set = set(selected_channels)
            with torch.no_grad():
                # Block every query→key edge where key column != query column.
                # Broadcasts [S]→[S,S] then to [B, 1, S, S].
                diff_col = col_channels.unsqueeze(0) != col_channels.unsqueeze(1)
                self_only_mask = base_mask.clone()
                # Use the last-dim index into a [B, 1, S, S] mask. diff_col is
                # [S, S] — broadcasting collapses to all batch/head entries.
                self_only_mask[..., diff_col] = self.neg_inf
                # Defensive diagonal unmask (same policy as per-channel path).
                self_only_mask[:, 0, diag_idx, diag_idx] = 0.0

                # input_ids unchanged — attention is the sole cross-token path.
                # DeltaNet layers with architecture.deltanet_block_causal="column"
                # maintain per-column state independence natively, so this mask
                # is sufficient for both pure-transformer and column-hybrid models.
                self_only_attn = {
                    "full_attention": self_only_mask,
                    "sliding_attention": self_only_mask,
                }
                out_s = model(
                    input_ids=input_ids,
                    attention_mask=self_only_attn,
                    position_ids=position_ids,
                    labels=None,
                    channel_ids=channel_ids,
                )
                ce_self_global = F.cross_entropy(
                    out_s.logits.view(-1, out_s.logits.size(-1)),
                    tgt.view(-1),
                    reduction="none",
                ).view(B, S)
            # Sentinel: in global mode the "per-channel" ce_self tensor is
            # the same for every channel. Use a defaultdict-like lambda so
            # the accumulator loop below can access ce_self_by_ch[c].
            ce_self_by_ch = {c: ce_self_global for c in range(C)}
        else:
            num_ch = min(self.longce_num_channels_per_step, C)
            # Pick channels via the selection helper (cycle or random). The
            # cycle advances only on this path — i.e., only when do_longce
            # is True — so skip steps preserve the cycle invariant.
            selected_channels = self._select_longce_channels(num_ch)
            selected_set = set(selected_channels)
            ce_self_by_ch = {}
            with torch.no_grad():
                for c in selected_channels:
                    self_only_mask = base_mask.clone()
                    other_cols = col_channels != c
                    self_only_mask[:, :, :, other_cols] = self.neg_inf
                    self_only_mask[:, 0, diag_idx, diag_idx] = 0.0

                    if self.self_only_drop_input_ids:
                        ids = input_ids.clone()
                        ids[:, other_cols] = drop_token_id
                    else:
                        ids = input_ids

                    self_only_attn = {
                        "full_attention": self_only_mask,
                        "sliding_attention": self_only_mask,
                    }
                    out_s = model(
                        input_ids=ids,
                        attention_mask=self_only_attn,
                        position_ids=position_ids,
                        labels=None,
                        channel_ids=channel_ids,
                    )
                    ce_self_by_ch[c] = F.cross_entropy(
                        out_s.logits.view(-1, out_s.logits.size(-1)),
                        tgt.view(-1),
                        reduction="none",
                    ).view(B, S)

        # Per-channel accumulators
        weighted_sum = torch.zeros((), device=device, dtype=torch.float32)
        tok_den = torch.zeros((), device=device, dtype=torch.float32)
        ch_loss_sum = torch.zeros((C,), device=device, dtype=torch.float32)
        ch_tok_den = torch.zeros((C,), device=device, dtype=torch.float32)

        lsd_sum_ch = torch.zeros((C,), device=device, dtype=torch.float32)
        clip_cnt_total = torch.zeros((), device=device, dtype=torch.float32)
        tok_cnt_total = torch.zeros((), device=device, dtype=torch.float32)

        is_user_masked = self.mask_user_loss and self.user_channel_idx is not None

        for c in selected_channels:
            ch_mask = (channel_ids == c) & valid

            for b in range(B):
                tok_idx = ch_mask[b].nonzero(as_tuple=True)[0]
                n_tok = tok_idx.numel()
                if n_tok == 0:
                    continue

                ce_f = ce_full[b, tok_idx]  # [n_tok], with gradient
                ce_s = ce_self_by_ch[c][b, tok_idx]  # [n_tok], detached

                with torch.no_grad():
                    # LSD = logp_full - logp_self = ce_self - ce_full
                    # (CE = -logp, so logp_full - logp_self = ce_self - ce_full)
                    lsd = ce_s - ce_f.detach()
                    if self.isoft_positive_only:
                        lsd = lsd.clamp_min(0.0)
                    w = torch.exp(lsd).clamp(max=gamma)
                    if self.isoft_floor_one:
                        w = w.clamp_min(1.0)
                    if self.isoft_per_channel_mean1:
                        w = w / w.mean().clamp_min(1e-6)
                    if self.longce_uniform_mix > 0.0:
                        w = self.longce_uniform_mix + (1.0 - self.longce_uniform_mix) * w

                    clipped = (lsd >= math.log(gamma)).float()
                    clip_cnt_total += clipped.sum()
                    tok_cnt_total += float(n_tok)
                    lsd_sum_ch[c] += lsd.sum()

                num_add = (w * ce_f).sum()
                den_add = ce_f.new_tensor(n_tok, dtype=torch.float32)

                ch_loss_sum[c] = ch_loss_sum[c] + num_add.detach()
                ch_tok_den[c] = ch_tok_den[c] + den_add
                if not (is_user_masked and c == self.user_channel_idx):
                    weighted_sum = weighted_sum + num_add
                    tok_den = tok_den + den_add

        # Unselected channels: weight=1 (standard CE)
        for c in range(C):
            if c in selected_set:
                continue
            ch_mask = (channel_ids == c) & valid
            for b in range(B):
                tok_idx = ch_mask[b].nonzero(as_tuple=True)[0]
                n_tok = tok_idx.numel()
                if n_tok == 0:
                    continue
                ce = ce_full[b, tok_idx]
                num_add = ce.sum()
                den_add = ce.new_tensor(n_tok, dtype=torch.float32)

                ch_loss_sum[c] = ch_loss_sum[c] + num_add.detach()
                ch_tok_den[c] = ch_tok_den[c] + den_add
                if not (is_user_masked and c == self.user_channel_idx):
                    weighted_sum = weighted_sum + num_add
                    tok_den = tok_den + den_add

        # Global loss + per-channel logging via single all-reduce
        # Pack: [tok_den, clip_cnt, tok_cnt, lsd_sum, ch_loss_0, ch_tok_0, ..., ch_loss_9, ch_tok_9]
        with torch.no_grad():
            stats = torch.cat(
                [
                    tok_den.unsqueeze(0),
                    clip_cnt_total.unsqueeze(0),
                    tok_cnt_total.unsqueeze(0),
                    lsd_sum_ch.sum().unsqueeze(0),
                    ch_loss_sum,
                    ch_tok_den,
                ]
            )
            g = self.accelerator.reduce(stats, reduction="sum")
            global_den = g[0].clamp_min(1.0)

        world_size = float(getattr(self.accelerator, "num_processes", 1))
        loss = weighted_sum * world_size / global_den

        with torch.no_grad():
            g_clip = g[1]
            g_tok_cnt = g[2].clamp_min(1.0)
            g_lsd_sum = g[3]
            g_ch_loss = g[4 : 4 + C]
            g_ch_tok = g[4 + C : 4 + 2 * C].clamp_min(1.0)

            log_dict: dict[str, Any] = {
                "longce/active": 1,
                "longce/clip_frac": float((g_clip / g_tok_cnt).item()),
                "longce/lsd_mean": float((g_lsd_sum / global_den).item()),
                # Cycle telemetry: remaining pool size on THIS rank after
                # this forward. Zero means the next call reshuffles. In
                # random mode this is always 0. Rank-0-only is fine —
                # the value is the same modulo per-rank independent RNG.
                "longce/pool_remaining": int(len(self._longce_pool)),
            }
            for c in range(C):
                name = self.channel_names[c] if c < len(self.channel_names) else f"ch{c}"
                log_dict[f"loss/{name}"] = float((g_ch_loss[c] / g_ch_tok[c]).item())
            self.log(log_dict)

            self._log_nonsilence_loss(ce_full.detach(), channel_ids, labels, valid)
            self._log_weight_norms()
            self._log_throughput(int(valid.sum().item()) * int(getattr(self.accelerator, "num_processes", 1)))

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Eval: generate from prefill + compute StreamWeaver metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_top_pk(logits, temperature=0.8, top_p=0.9, top_k=0):
        if temperature > 0:
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

    @torch.no_grad()
    def _eval_generate_sample(self, model, sample):
        """Generate from a prefilled training sample.

        Returns (all_rows, n_prefill) where all_rows is a list of
        [C] token-id lists (prefill rows + generated rows).
        """
        device = next(model.parameters()).device
        C = self.num_channels
        temperature = self.eval_temperature
        top_p = self.eval_top_p
        top_k = self.eval_top_k

        # Reshape flat input_ids to [num_rows, C]
        input_ids = sample["input_ids"]
        num_rows = sample["num_rows"]
        grid = input_ids[: num_rows * C].view(num_rows, C)

        n_prefill = max(1, int(num_rows * self.eval_prefill_frac))
        prefill = grid[:n_prefill]  # [n_prefill, C]

        # Build prefill input
        flat = prefill.reshape(-1).unsqueeze(0).to(device)  # [1, N]
        N = n_prefill * C
        position_ids = (torch.arange(N, device=device) // C).unsqueeze(0)
        channel_ids = (torch.arange(N, device=device) % C).unsqueeze(0)

        # Block-causal mask for prefill
        rows_idx = torch.arange(N, device=device) // C
        can_attend = (rows_idx.unsqueeze(0) < rows_idx.unsqueeze(1)) | torch.eye(N, dtype=torch.bool, device=device)
        mask = torch.where(can_attend, 0.0, -1e4).to(torch.bfloat16)[None, None]

        # Skip-silence: mask silence key positions (nobody attends TO them)
        # but restore self-diagonal so silence tokens can still attend to themselves
        if self.attention_mask_type == "block_causal_skip_silence":
            sil_cols = flat[0] == self.silence_token_id  # [N]
            mask[0, 0, :, :].masked_fill_(sil_cols.unsqueeze(0), -1e4)
            mask[0, 0].diagonal().clamp_(min=0.0)  # restore self-attention

        outputs = model(
            input_ids=flat,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=position_ids,
            channel_ids=channel_ids,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        logits = outputs.logits[0]  # [N, V]

        all_rows = prefill.tolist()

        # Track all token IDs fed to KV cache (for skip-silence masking)
        all_cached_ids = flat[0].clone()  # [N]

        # First generated row from last prefill row's logits
        last_logits = logits[(n_prefill - 1) * C : n_prefill * C]
        row = [self._sample_top_pk(last_logits[c], temperature, top_p, top_k) for c in range(C)]
        all_rows.append(row)

        # Peer mask template: block same-row peers, allow self
        peer_mask = torch.full((C, C), -1e4, device=device, dtype=torch.bfloat16)
        peer_mask.diagonal().fill_(0.0)

        skip_silence = self.attention_mask_type == "block_causal_skip_silence"

        current_row_idx = n_prefill
        for _ in range(self.eval_gen_rows - 1):
            row_tensor = torch.tensor([row], device=device, dtype=torch.long)
            pos_ids = torch.full((1, C), current_row_idx, device=device, dtype=torch.long)
            ch_ids = torch.arange(C, device=device, dtype=torch.long).unsqueeze(0)

            cached_len = past_kv.get_seq_length()
            attn_mask = torch.cat(
                [
                    torch.zeros(1, 1, C, cached_len, device=device, dtype=torch.bfloat16),
                    peer_mask[None, None],
                ],
                dim=-1,
            )

            # Skip-silence: mask silence key positions in both cache and peer sections
            if skip_silence:
                sil_cols = torch.cat(
                    [
                        all_cached_ids == self.silence_token_id,
                        row_tensor[0] == self.silence_token_id,
                    ]
                )  # [cached_len + C]
                attn_mask[0, 0].masked_fill_(sil_cols.unsqueeze(0), -1e4)
                # Restore self-diagonal for numerical stability
                for i in range(C):
                    attn_mask[0, 0, i, cached_len + i] = 0.0

            outputs = model(
                input_ids=row_tensor,
                attention_mask={
                    "full_attention": attn_mask,
                    "sliding_attention": attn_mask,
                },
                position_ids=pos_ids,
                channel_ids=ch_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            all_cached_ids = torch.cat([all_cached_ids, row_tensor[0]])
            logits_row = outputs.logits[0]  # [C, V]

            row = [self._sample_top_pk(logits_row[c], temperature, top_p, top_k) for c in range(C)]
            all_rows.append(row)
            current_row_idx += 1

        return all_rows, n_prefill

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        old_use_cache: bool = model.config.use_cache  # type: ignore
        metrics = {}
        C = self.num_channels

        # --- Validation CE loss (all ranks) ---
        if self.eval_dataset is not None:
            model.config.use_cache: bool = False  # type: ignore
            device = next(model.parameters()).device

            is_user_masked = self.mask_user_loss and self.user_channel_idx is not None

            def _eval_ce_loss(ds, prefix="eval"):
                """Compute per-channel CE loss on a dataset, return metrics dict."""
                dl = self.get_eval_dataloader(ds)
                st = torch.zeros(2 + C * 2, device=device)
                with torch.no_grad():
                    for batch in dl:
                        batch = self._prepare_inputs(batch)
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            position_ids=batch["position_ids"],
                            channel_ids=batch["channel_ids"],
                        )
                        labels = batch["labels"]
                        ch_ids = batch["channel_ids"]
                        valid = labels != -100
                        per_token = F.cross_entropy(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                            reduction="none",
                        ).view_as(labels)
                        # Headline loss excludes user channel (matching training)
                        if is_user_masked:
                            headline_mask = valid & (ch_ids != self.user_channel_idx)
                        else:
                            headline_mask = valid
                        st[0] += (per_token * headline_mask.float()).sum()
                        st[1] += headline_mask.sum().float()
                        for c in range(C):
                            ch_mask = (ch_ids == c) & valid
                            st[2 + c * 2] += (per_token * ch_mask.float()).sum()
                            st[2 + c * 2 + 1] += ch_mask.sum().float()
                g = self.accelerator.reduce(st, reduction="sum")
                m = {}
                if g[1] > 0:
                    m[f"{prefix}/loss"] = float((g[0] / g[1]).item())
                for c in range(C):
                    if g[2 + c * 2 + 1] > 0:
                        name = self.channel_names[c] if c < len(self.channel_names) else f"ch{c}"
                        m[f"{prefix}/loss_{name}"] = float((g[2 + c * 2] / g[2 + c * 2 + 1]).item())
                return m

            # Primary eval: unpacked (fair comparison)
            metrics.update(_eval_ce_loss(self.eval_dataset, "eval"))

            # Context eval: measure how well model uses prior context
            for i, ctx_ds in enumerate(self.eval_context_datasets, 1):
                if ctx_ds is not None:
                    metrics.update(_eval_ce_loss(ctx_ds, f"eval_ctx{i}"))

        # --- Generation metrics (all ranks generate to avoid NCCL timeout) ---
        # With ZeRO-2 the full model is replicated on each rank. Running
        # generation on all ranks keeps every GPU busy so no rank idles at
        # wait_for_everyone() long enough to trigger the NCCL watchdog.
        # Only rank 0 aggregates and logs the resulting metrics.
        model.config.use_cache = True  # type: ignore
        tokenizer = getattr(self, "processing_class", None) or getattr(  # type: ignore
            self, "tokenizer", None
        )
        # Unwrap Subset to access get_raw_sample on the base dataset
        ds = self.train_dataset
        if isinstance(ds, torch.utils.data.Subset):
            base_ds = ds.dataset
            pool = list(ds.indices)
        else:
            base_ds = ds
            pool = list(range(len(ds)))  # type: ignore

        n = min(self.eval_num_samples, len(pool))
        chosen = random.sample(pool, n)

        headers = [name.capitalize() for name in self.channel_names]
        all_metric_results = []
        all_gen_metrics = []
        for idx in chosen:
            sample = base_ds.get_raw_sample(idx)  # type: ignore
            try:
                rows, n_prefill = self._eval_generate_sample(model, sample)
                generated_rows = rows[n_prefill:]

                gen_m = compute_generation_metrics(
                    generated_rows,
                    silence_token_id=self.silence_token_id,
                    all_rows=rows,
                    channel_names=self.channel_names,
                )
                all_gen_metrics.append(gen_m)

                decoded_rows = [
                    [
                        "-" if t == self.silence_token_id else (tokenizer.decode([t]).strip() or "-")  # type: ignore
                        for t in row
                    ]
                    for row in generated_rows
                ]
                table = StreamTable(headers=headers, rows=decoded_rows)
                result = compute_all_metrics(table)
                del result["format"], result["silence"]
                all_metric_results.append(result)
            except Exception as e:
                logger.warning(f"Eval sample {idx} failed: {e}")

        # Only main process aggregates and logs generation metrics
        if self.accelerator.is_main_process:
            RAW_DETAILS = {
                "fill_rate": ["overall_fill_rate", "thinking_fill_rate"],
                "interaction": [
                    "avg_overlap",
                    "interaction_word_ratio",
                    "response_pattern_ratio",
                    "stream_reference_ratio",
                ],
                "diversity": ["avg_distance"],
                "word_repetition": ["issue_ratio"],
            }
            if all_metric_results:
                for metric_name, detail_keys in RAW_DETAILS.items():
                    for dk in detail_keys:
                        vals = [r[metric_name].details.get(dk, 0) for r in all_metric_results if metric_name in r]
                        if vals:
                            metrics[f"eval/{dk}"] = sum(vals) / len(vals)

            if all_gen_metrics:
                for key in all_gen_metrics[0]:
                    vals = [m[key] for m in all_gen_metrics if key in m]
                    if vals:
                        metrics[f"eval/{key}"] = sum(vals) / len(vals)

            logger.info(f"Eval metrics: {metrics}")

        model.config.use_cache: bool = old_use_cache  # type: ignore
        model.train()
        self.accelerator.wait_for_everyone()

        if metrics:
            self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Optimizer with per-group LRs
    # ------------------------------------------------------------------
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model

        decay_names = get_parameter_names(model, [nn.LayerNorm])
        decay_names = [n for n in decay_names if not n.endswith("bias")]
        wd = float(self.args.weight_decay)

        bb_decay, bb_nodecay = [], []
        ce_decay, ce_nodecay = [], []

        for name, p in model.named_parameters():  # type: ignore
            if not p.requires_grad:
                continue

            is_ce = "channel_embedding" in name
            is_decay = (name in decay_names) and ("embedding" not in name.lower())

            if is_ce:
                (ce_decay if is_decay else ce_nodecay).append(p)
            else:
                (bb_decay if is_decay else bb_nodecay).append(p)

        groups = []
        if bb_decay:
            groups.append({"params": bb_decay, "weight_decay": wd, "lr": self.backbone_lr})
        if bb_nodecay:
            groups.append({"params": bb_nodecay, "weight_decay": 0.0, "lr": self.backbone_lr})
        if ce_decay:
            groups.append(
                {
                    "params": ce_decay,
                    "weight_decay": wd,
                    "lr": self.channel_embedding_lr,
                }
            )
        if ce_nodecay:
            groups.append(
                {
                    "params": ce_nodecay,
                    "weight_decay": 0.0,
                    "lr": self.channel_embedding_lr,
                }
            )

        if self.optim_8bit:
            try:
                import bitsandbytes as bnb  # type: ignore
            except ImportError:
                raise ImportError(  # noqa: B904
                    "optim.adam_8bit requires bitsandbytes: uv pip install bitsandbytes"
                )
            self.optimizer = bnb.optim.AdamW8bit(
                groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        else:
            from torch.optim import AdamW

            self.optimizer = AdamW(
                groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer

    # ------------------------------------------------------------------
    # Scheduler with optional cooldown
    # ------------------------------------------------------------------
    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        if self.cooldown_ratio <= 0:
            return super().create_scheduler(num_training_steps, optimizer)

        optimizer = optimizer or self.optimizer
        warmup = self.args.warmup_steps
        cooldown = int(num_training_steps * self.cooldown_ratio)
        constant_end = num_training_steps - cooldown

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            if step < constant_end:
                return 1.0
            return max(0.0, (num_training_steps - step) / max(1, cooldown))

        from torch.optim.lr_scheduler import LambdaLR

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler
