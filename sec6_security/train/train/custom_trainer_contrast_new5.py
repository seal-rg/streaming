#!/usr/bin/env python3

"""
customized_trainer.py

TRL SFTTrainer customized for multi-head training with:

- Optimizer groups (backbone vs medusa_head/channel_embedding)
- LongCE-channel self-only:
    FULL forward (grad) => logp_full [B,S,V]
    SELF-only forward (no_grad) per head:
        - mask USER content entirely (as KEYS) using additive NxN mask
        - mask other assistant head spans (as KEYS)
        - (optional) replace input_ids in those spans with drop token id
    LSD = logP_full - logP_self
    Isoft = clamp(exp(LSD), gamma)
    optional per-head mean=1 normalization

- DDP safe denominator reduction: reduce only denominators under no_grad
"""

import logging
import math
import random
from typing import Any

import torch
import torch.nn.functional as F
import trl
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

logger = logging.getLogger(__name__)


class CustomizedTrainer(trl.SFTTrainer):
    """Multi-head trainer with per-group LRs + self-only LongCE-channel + per-head mean=1 Isoft."""

    def __init__(self, *args, **kwargs):
        # -------------------------
        # Optimizer LRs
        # -------------------------
        self.backbone_lr = float(kwargs.pop("backbone_lr", 1e-5))
        self.head_lr = float(kwargs.pop("head_lr", 2e-5))
        self.head_param_keywords = tuple(
            kwargs.pop(
                "head_param_keywords",
                ("medusa_head", "channel_embedding", "lm_head"),
            )
        )

        # -------------------------
        # Label conventions
        # -------------------------
        self.IGNORE_TOKEN_ID = int(kwargs.pop("ignore_token_id", -100))
        self.label_shift = int(kwargs.pop("label_shift", 1))

        # -------------------------
        # LongCE-channel config
        # -------------------------
        self.enable_longce_channel = bool(kwargs.pop("enable_longce_channel", True))
        self.longce_prob = float(kwargs.pop("longce_prob", 1.0))
        self.longce_gamma = float(kwargs.pop("longce_gamma", 3.0))
        self.longce_warmup_steps = int(kwargs.pop("longce_warmup_steps", 0))
        self.longce_ramp_steps = int(kwargs.pop("longce_ramp_steps", 0))

        self.isoft_positive_only = bool(kwargs.pop("isoft_positive_only", False))
        self.isoft_floor_one = bool(kwargs.pop("isoft_floor_one", False))
        self.isoft_per_head_mean1 = bool(kwargs.pop("isoft_per_head_mean1", True))

        # ✅ additive mask NEG_INF (must match collator)
        self.neg_inf = float(kwargs.pop("neg_inf", -1e4))

        # ✅ self-only behavior toggles
        self.self_only_mask_user = bool(kwargs.pop("self_only_mask_user", True))
        self.self_only_drop_input_ids = bool(kwargs.pop("self_only_drop_input_ids", True))  # also replace ids in masked spans

        # -------------------------
        # Logging
        # -------------------------
        self.log_every_steps = int(kwargs.pop("log_every_steps", 1))

        super().__init__(*args, **kwargs)

    # -------------------------
    # Optimizer grouping
    # -------------------------
    def get_optimizer_grouped_parameters(self, model: nn.Module):
        decay_names = get_parameter_names(model, [nn.LayerNorm])
        decay_names = [n for n in decay_names if not n.endswith("bias")]
        wd = float(self.args.weight_decay)

        bb_decay, bb_nodecay = [], []
        hd_decay, hd_nodecay = [], []

        bb_n, hd_n = 0, 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            is_head = any(k in name for k in self.head_param_keywords)

            if "embedding" in name.lower():
                is_decay = False
            else:
                is_decay = name in decay_names

            if is_head:
                hd_n += p.numel()
                (hd_decay if is_decay else hd_nodecay).append(p)
            else:
                bb_n += p.numel()
                (bb_decay if is_decay else bb_nodecay).append(p)

        groups = []
        if bb_decay:
            groups.append({"params": bb_decay, "weight_decay": wd, "lr": self.backbone_lr})
        if bb_nodecay:
            groups.append({"params": bb_nodecay, "weight_decay": 0.0, "lr": self.backbone_lr})
        if hd_decay:
            groups.append({"params": hd_decay, "weight_decay": wd, "lr": self.head_lr})
        if hd_nodecay:
            groups.append({"params": hd_nodecay, "weight_decay": 0.0, "lr": self.head_lr})
        self._group_numel = {
            "backbone": int(bb_n),
            "head": int(hd_n),
        }
        return groups

    def create_optimizer(self):
        if self.optimizer is None:
            model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
            param_groups = self.get_optimizer_grouped_parameters(model)
            from torch.optim import AdamW

            self.optimizer = AdamW(
                param_groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            g = getattr(self, "_group_numel", {"backbone": 0, "head": 0})
            logger.warning(
                "optimizer groups: backbone=%d params (lr=%.3e), head=%d params (lr=%.3e), keys=%s",
                int(g.get("backbone", 0)),
                float(self.backbone_lr),
                int(g.get("head", 0)),
                float(self.head_lr),
                list(self.head_param_keywords),
            )
            if int(g.get("head", 0)) == 0:
                logger.warning("head param group is empty; split LR is ineffective with current head_param_keywords.")
        return self.optimizer

    # -------------------------
    # Logging helper
    # -------------------------
    def _maybe_log(self, log_dict: dict[str, Any]):
        step = int(getattr(self.state, "global_step", 0))
        if self.log_every_steps <= 1 or (step % self.log_every_steps == 0):
            self.log(log_dict)

    # -------------------------
    # LongCE schedule
    # -------------------------
    def _longce_prob_now(self) -> float:
        step = int(getattr(self.state, "global_step", 0))
        if not self.enable_longce_channel:
            return 0.0
        if step < self.longce_warmup_steps:
            return 0.0
        if self.longce_ramp_steps <= 0:
            return float(self.longce_prob)
        t = (step - self.longce_warmup_steps) / float(self.longce_ramp_steps)
        t = max(0.0, min(1.0, t))
        return float(self.longce_prob) * t

    # -------------------------
    # Safer drop token id
    # -------------------------
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

    # -------------------------
    # Model forward helper (logits [B,S,V])
    # -------------------------
    def _forward_get_logits(self, model, _inputs) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, Any, dict[str, Any]]:
        """
        Returns:
          sum_loss, num_tokens (if model returns them),
          logits_bsv [B,S,V],
          raw_output,
          log_dict
        """
        ret = model(
            input_ids=_inputs["input_ids"],
            attention_mask=_inputs.get("attention_mask", None),
            labels=_inputs.get("labels", None),
            position_ids=_inputs.get("position_ids", None),
            boundaries=_inputs.get("boundaries", None),
            head_start=_inputs.get("head_start", None),
            head_end=_inputs.get("head_end", None),
            head_ok=_inputs.get("head_ok", None),
            return_dict=False,
        )

        if isinstance(ret, tuple) and len(ret) == 4 and torch.is_tensor(ret[0]):
            sum_loss, num_tokens, output, log_dict = ret
        else:
            sum_loss, num_tokens, output, log_dict = None, None, ret, {}

        logits_bsv = output[0]  # [B,S,V]
        return sum_loss, num_tokens, logits_bsv, output, log_dict

    # -------------------------
    # helpers: additive NxN mask key-column blocking
    # -------------------------
    def _block_key_span_in_additive_mask(self, attn: torch.Tensor, st: int, ed: int, b: int):
        """
        attn: [B,1,S,S] additive. block keys in [st,ed)
        """
        if ed <= st:
            return
        attn[b, 0, :, st:ed] = float(self.neg_inf)

    # -------------------------
    # Loss
    # -------------------------
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs: bool = False):
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", None)
        attention_mask = inputs.get("attention_mask", None)  # [B,1,S,S] additive
        position_ids = inputs.get("position_ids", None)

        head_start = inputs.get("head_start", None)  # [B,H]
        head_end = inputs.get("head_end", None)  # [B,H]
        head_ok = inputs.get("head_ok", None)  # [B,H] bool

        # ✅ user spans (content only)
        user_start = inputs.get("user_start", None)  # [B]
        user_end = inputs.get("user_end", None)  # [B]

        device = input_ids.device
        B, S = input_ids.shape

        # Decide whether to run LongCE-channel this step
        do_longce = (
            labels is not None
            and head_start is not None
            and head_end is not None
            and head_ok is not None
            and attention_mask is not None
            and self.enable_longce_channel
        )
        p_now = 0.0
        if do_longce:
            p_now = float(self._longce_prob_now())
            if random.random() >= p_now:
                do_longce = False

        # -------------------------
        # Non-LongCE path (use model's provided sum_loss/num_tokens)
        # -------------------------
        if not do_longce:
            sum_loss, num_tokens, _, output, log_dict = self._forward_get_logits(model, inputs)

            if labels is not None:
                assert sum_loss is not None and num_tokens is not None, (
                    "Expected (sum_loss, num_tokens, output, log_dict) when labels are provided."
                )

            if sum_loss is None:
                sum_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            if not torch.is_tensor(sum_loss):
                sum_loss = torch.tensor(float(sum_loss), device=device, dtype=torch.float32)

            if num_tokens is None:
                num_tokens = torch.tensor(0.0, device=device, dtype=torch.float32)
            if not torch.is_tensor(num_tokens):
                num_tokens = torch.tensor(float(num_tokens), device=device, dtype=torch.float32)

            with torch.no_grad():
                global_tokens = self.accelerator.reduce(num_tokens.float(), reduction="sum").clamp_min(1.0)
            world_size = float(getattr(self.accelerator, "num_processes", 1))
            loss = sum_loss * world_size / global_tokens

            safe_log: dict[str, Any] = {}
            if isinstance(log_dict, dict):
                safe_log.update(dict(log_dict))
            safe_log.update(
                {
                    "longce_channel_on": 0,
                    "longce_prob_now": float(self._longce_prob_now()),
                    "loss_global": float(loss.detach().item()),
                    "tokden_global": float(global_tokens.detach().item()),
                }
            )
            self._maybe_log(safe_log)

            return (loss, output) if return_outputs else loss

        # -------------------------
        # LongCE-channel path
        # -------------------------
        H = int(head_start.shape[1]) - 2
        shift = int(self.label_shift)
        gamma = float(self.longce_gamma)
        drop_token_id = self._get_drop_token_id()

        base_attn = attention_mask  # [B,1,S,S] additive

        # FULL forward (grad): logits only
        inputs_full = dict(inputs)
        inputs_full["labels"] = None
        _, _, logits_full, output_full, _ = self._forward_get_logits(model, inputs_full)  # [B,S,V]
        logp_full = F.log_softmax(logits_full, dim=-1)  # [B,S,V]

        # SELF-ONLY forwards (no_grad): one per head
        logp_self = [None] * H
        with torch.no_grad():
            for keep_h in range(H):
                ids = input_ids.clone()
                attn2 = base_attn.clone()

                # ✅ 1) 屏蔽 user 全部内容（作为 KEY 列），让任何 query 看不到 user 内容
                if self.self_only_mask_user and (user_start is not None) and (user_end is not None):
                    u_st = user_start.to(device).long()
                    u_ed = user_end.to(device).long()
                    for b in range(B):
                        st_u = int(u_st[b].item())
                        ed_u = int(u_ed[b].item())
                        st_u = max(0, min(st_u, S))
                        ed_u = max(0, min(ed_u, S))
                        if ed_u > st_u:
                            self._block_key_span_in_additive_mask(attn2, st_u, ed_u, b)
                            if self.self_only_drop_input_ids:
                                ids[b, st_u:ed_u] = drop_token_id

                # ✅ 2) 屏蔽其它 head span（作为 KEY 列），只保留 keep_h
                for b in range(B):
                    for h in range(H):
                        if h == keep_h:
                            continue
                        if not bool(head_ok[b, h].item()):
                            continue
                        st = int(head_start[b, h].item())
                        ed = int(head_end[b, h].item())
                        st = max(0, min(st, S))
                        ed = max(0, min(ed, S))
                        if ed <= st:
                            continue

                        # block as keys
                        self._block_key_span_in_additive_mask(attn2, st, ed, b)
                        if self.self_only_drop_input_ids:
                            ids[b, st:ed] = drop_token_id

                inputs_self = dict(inputs)
                inputs_self["input_ids"] = ids
                inputs_self["attention_mask"] = attn2
                inputs_self["labels"] = None

                _, _, logits_s, _, _ = self._forward_get_logits(model, inputs_self)  # [B,S,V]
                logp_self[keep_h] = F.log_softmax(logits_s, dim=-1)

        # Accumulators
        weighted_sum = torch.zeros((), device=device, dtype=torch.float32)  # HAS grad
        tok_den = torch.zeros((), device=device, dtype=torch.float32)  # NO grad

        weighted_sum_h = torch.zeros((H,), device=device, dtype=torch.float32)
        tok_den_h = torch.zeros((H,), device=device, dtype=torch.float32)

        lsd_sum_h = torch.zeros((H,), device=device, dtype=torch.float32)
        isoft_std_sum_h = torch.zeros((H,), device=device, dtype=torch.float32)
        clip_cnt_h = torch.zeros((H,), device=device, dtype=torch.float32)
        tok_cnt_h = torch.zeros((H,), device=device, dtype=torch.float32)

        clip_cnt_total = torch.zeros((), device=device, dtype=torch.float32)
        tok_cnt_total = torch.zeros((), device=device, dtype=torch.float32)

        for b in range(B):
            for h in range(H):
                if not bool(head_ok[b, h].item()):
                    continue

                st = int(head_start[b, h].item())
                ed = int(head_end[b, h].item())
                st = max(0, min(st, S))
                ed = max(0, min(ed, S))
                if ed <= st:
                    continue
                if st - shift < 0:
                    continue

                tgt = labels[b, st:ed]
                valid = tgt != self.IGNORE_TOKEN_ID
                if not valid.any():
                    continue

                pred_all = torch.arange(st - shift, ed - shift, device=device)
                pred_idx = pred_all[valid]
                tgt_ids = tgt[valid].to(device)
                n_tok = tgt_ids.numel()
                if n_tok <= 0:
                    continue

                # FULL logprob (grad)
                lp_f = logp_full[b, pred_idx, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # [N]
                ce_full = -lp_f

                # SELF-only logprob (no grad)
                lp_s = logp_self[h][b, pred_idx, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # [N]

                with torch.no_grad():
                    lsd = lp_f.detach() - lp_s
                    if self.isoft_positive_only:
                        lsd = lsd.clamp_min(0.0)

                    w_pre = torch.exp(lsd).clamp(max=gamma)
                    if self.isoft_floor_one:
                        w_pre = w_pre.clamp_min(1.0)

                    clipped = (lsd >= math.log(gamma)).float()
                    clip_cnt = clipped.sum()

                    if self.isoft_per_head_mean1:
                        w = w_pre / w_pre.mean().clamp_min(1e-6)
                    else:
                        w = w_pre

                    lsd_sum_h[h] += lsd.sum()
                    isoft_std_sum_h[h] += w.std(unbiased=False) * w.new_tensor(n_tok, dtype=torch.float32)
                    clip_cnt_h[h] += clip_cnt
                    tok_cnt_h[h] += w.new_tensor(n_tok, dtype=torch.float32)

                    clip_cnt_total += clip_cnt
                    tok_cnt_total += w.new_tensor(n_tok, dtype=torch.float32)

                num_add = (w * ce_full).sum()
                den_add = ce_full.new_tensor(n_tok, dtype=torch.float32)

                weighted_sum = weighted_sum + num_add
                tok_den = tok_den + den_add

                weighted_sum_h[h] = weighted_sum_h[h] + num_add
                tok_den_h[h] = tok_den_h[h] + den_add

        # DDP-consistent global mean WITHOUT reducing grad tensors
        with torch.no_grad():
            global_den = self.accelerator.reduce(tok_den, reduction="sum").clamp_min(1.0)
        world_size = float(getattr(self.accelerator, "num_processes", 1))
        loss = weighted_sum * world_size / global_den

        # ---- Globalize per-head stats for logging ----
        with torch.no_grad():
            g_tok_den_h = self.accelerator.reduce(tok_den_h, reduction="sum").clamp_min(1.0)
            g_num_h = self.accelerator.reduce(weighted_sum_h.detach(), reduction="sum")
            head_loss_global = g_num_h / g_tok_den_h

            g_lsd_sum_h = self.accelerator.reduce(lsd_sum_h, reduction="sum")
            head_lsd_mean = g_lsd_sum_h / g_tok_den_h

            g_isoft_std_sum_h = self.accelerator.reduce(isoft_std_sum_h, reduction="sum")
            head_isoft_std = g_isoft_std_sum_h / g_tok_den_h

            g_clip_cnt_h = self.accelerator.reduce(clip_cnt_h, reduction="sum")
            head_clip_frac = g_clip_cnt_h / g_tok_den_h

            g_clip_cnt_total = self.accelerator.reduce(clip_cnt_total, reduction="sum")
            g_tok_cnt_total = self.accelerator.reduce(tok_cnt_total, reduction="sum").clamp_min(1.0)
            clip_frac_overall = float((g_clip_cnt_total / g_tok_cnt_total).item())

        safe_log: dict[str, Any] = {
            "longce_channel_on": 1,
            "longce_prob_now": float(p_now),
            "loss_global": float(loss.detach().item()),
            "tokden_global": float(global_den.detach().item()),
            "gamma": float(gamma),
            "label_shift": int(shift),
            "lsd_mean": float((g_lsd_sum_h.sum() / global_den).item()),
            "isoft_pos_only": int(self.isoft_positive_only),
            "isoft_floor1": int(self.isoft_floor_one),
            "isoft_head_mean1": int(self.isoft_per_head_mean1),
            "clip_frac_overall": float(clip_frac_overall),
            "self_only_mask_user": int(self.self_only_mask_user),
            "self_only_drop_input_ids": int(self.self_only_drop_input_ids),
            "neg_inf": float(self.neg_inf),
        }
        for h in range(H):
            safe_log[f"head{h}_tokden_global"] = float(g_tok_den_h[h].item())
            safe_log[f"head{h}_loss_global"] = float(head_loss_global[h].item())
            safe_log[f"head{h}_lsd_mean"] = float(head_lsd_mean[h].item())
            safe_log[f"head{h}_isoft_std"] = float(head_isoft_std[h].item())
            safe_log[f"head{h}_clip_frac"] = float(head_clip_frac[h].item())

        self._maybe_log(safe_log)

        return (loss, output_full) if return_outputs else loss
