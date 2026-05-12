#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import random
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import trl
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

logger = logging.getLogger(__name__)


class CustomizedTrainer(trl.SFTTrainer):
    """
    Optimized multi-head trainer:
    - Real optimizer param groups (backbone/head LR split)
    - Optional LongCE-channel reweighting
    - Optional head-balanced aggregation
    """

    def __init__(self, *args, **kwargs):
        # Optimizer groups
        self.backbone_lr = float(kwargs.pop("backbone_lr", 1e-5))
        self.head_lr = float(kwargs.pop("head_lr", 2e-5))
        self.head_param_keywords = tuple(kwargs.pop(
            "head_param_keywords",
            ("medusa_head", "channel_embedding", "lm_head"),
        ))

        # Label config
        self.IGNORE_TOKEN_ID = int(kwargs.pop("ignore_token_id", -100))
        self.label_shift = int(kwargs.pop("label_shift", 1))

        # LongCE controls
        self.enable_longce_channel = bool(kwargs.pop("enable_longce_channel", True))
        self.longce_prob = float(kwargs.pop("longce_prob", 0.5))
        self.longce_gamma = float(kwargs.pop("longce_gamma", 2.0))
        self.longce_warmup_steps = int(kwargs.pop("longce_warmup_steps", 100))
        self.longce_ramp_steps = int(kwargs.pop("longce_ramp_steps", 2000))

        # Isoft shaping
        self.isoft_positive_only = bool(kwargs.pop("isoft_positive_only", True))
        self.isoft_floor_one = bool(kwargs.pop("isoft_floor_one", False))
        self.isoft_per_head_mean1 = bool(kwargs.pop("isoft_per_head_mean1", True))

        # Self-only masking behavior
        self.neg_inf = float(kwargs.pop("neg_inf", -1e4))
        self.self_only_mask_user = bool(kwargs.pop("self_only_mask_user", True))
        self.self_only_drop_input_ids = bool(kwargs.pop("self_only_drop_input_ids", True))

        # Aggregation mode: "token" (global token mean) or "head" (mean over head means)
        self.balance_mode = str(kwargs.pop("balance_mode", "head")).strip().lower()
        if self.balance_mode not in {"token", "head"}:
            self.balance_mode = "head"

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

            # Robust to wrappers like model.xxx, module.xxx, base_model.model.xxx
            is_head = any(k in name for k in self.head_param_keywords)
            is_decay = (name in decay_names) and ("embedding" not in name.lower())

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
            groups = self.get_optimizer_grouped_parameters(model)
            from torch.optim import AdamW
            self.optimizer = AdamW(
                groups,
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
    # Helpers
    # -------------------------
    def _maybe_log(self, log_dict: Dict[str, Any]):
        step = int(getattr(self.state, "global_step", 0))
        if self.log_every_steps <= 1 or (step % self.log_every_steps == 0):
            self.log(log_dict)

    def _longce_prob_now(self) -> float:
        if not self.enable_longce_channel:
            return 0.0
        step = int(getattr(self.state, "global_step", 0))
        if step < self.longce_warmup_steps:
            return 0.0
        if self.longce_ramp_steps <= 0:
            return float(self.longce_prob)
        t = (step - self.longce_warmup_steps) / float(self.longce_ramp_steps)
        t = max(0.0, min(1.0, t))
        return float(self.longce_prob) * t

    def _get_drop_token_id(self) -> int:
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            if getattr(tok, "pad_token_id", None) is not None:
                return int(tok.pad_token_id)
            if getattr(tok, "eos_token_id", None) is not None:
                return int(tok.eos_token_id)
        return 0

    def _block_key_span(self, attn: torch.Tensor, b: int, st: int, ed: int):
        if ed > st:
            attn[b, 0, :, st:ed] = float(self.neg_inf)

    def _forward_get_logits(
        self, model, _inputs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Any, Dict[str, Any]]:
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
        logits_bsv = output[0]
        return sum_loss, num_tokens, logits_bsv, output, log_dict

    # -------------------------
    # Base CE (non-LongCE)
    # -------------------------
    def _compute_base_loss(self, model, inputs, device):
        sum_loss, num_tokens, _, output, log_dict = self._forward_get_logits(model, inputs)

        if sum_loss is None:
            sum_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        elif not torch.is_tensor(sum_loss):
            sum_loss = torch.tensor(float(sum_loss), device=device, dtype=torch.float32)
        else:
            sum_loss = sum_loss.float()

        if num_tokens is None:
            num_tokens = torch.tensor(0.0, device=device, dtype=torch.float32)
        elif not torch.is_tensor(num_tokens):
            num_tokens = torch.tensor(float(num_tokens), device=device, dtype=torch.float32)
        else:
            num_tokens = num_tokens.float()

        # Keep gradient local; reduce denominator only, then compensate DDP mean.
        with torch.no_grad():
            global_tokens = self.accelerator.reduce(num_tokens, reduction="sum").clamp_min(1.0)
        world_size = float(getattr(self.accelerator, "num_processes", 1))
        loss = sum_loss * world_size / global_tokens

        safe_log = dict(log_dict) if isinstance(log_dict, dict) else {}
        safe_log.update({
            "longce_channel_on": 0,
            "longce_prob_now": float(self._longce_prob_now()),
            "loss_global": float(loss.detach().item()),
            "tokden_global": float(global_tokens.detach().item()),
        })
        self._maybe_log(safe_log)
        return loss, output

    # -------------------------
    # Main loss
    # -------------------------
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", None)
        attention_mask = inputs.get("attention_mask", None)  # [B,1,S,S] additive
        head_start = inputs.get("head_start", None)          # [B,H]
        head_end = inputs.get("head_end", None)              # [B,H]
        head_ok = inputs.get("head_ok", None)                # [B,H]
        user_start = inputs.get("user_start", None)          # [B]
        user_end = inputs.get("user_end", None)              # [B]

        device = input_ids.device
        B, S = input_ids.shape

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

        if not do_longce:
            loss, output = self._compute_base_loss(model, inputs, device)
            return (loss, output) if return_outputs else loss

        hs = head_start.to(device).long()
        he = head_end.to(device).long()
        hok = head_ok.to(device).bool()
        H = int(hs.shape[1])
        if H <= 0:
            loss, output = self._compute_base_loss(model, inputs, device)
            return (loss, output) if return_outputs else loss

        # FULL forward (grad)
        full_inputs = dict(inputs)
        full_inputs["labels"] = None
        _, _, logits_full, output_full, _ = self._forward_get_logits(model, full_inputs)
        logp_full = F.log_softmax(logits_full, dim=-1)

        # SELF-only forward per head (no grad)
        drop_token_id = self._get_drop_token_id()
        logp_self = [None] * H
        with torch.no_grad():
            for keep_h in range(H):
                ids = input_ids.clone()
                attn2 = attention_mask.clone()

                if self.self_only_mask_user and (user_start is not None) and (user_end is not None):
                    u_st = user_start.to(device).long()
                    u_ed = user_end.to(device).long()
                    for b in range(B):
                        st_u = max(0, min(int(u_st[b].item()), S))
                        ed_u = max(0, min(int(u_ed[b].item()), S))
                        if ed_u > st_u:
                            self._block_key_span(attn2, b, st_u, ed_u)
                            if self.self_only_drop_input_ids:
                                ids[b, st_u:ed_u] = drop_token_id

                for b in range(B):
                    for h in range(H):
                        if h == keep_h:
                            continue
                        if not bool(hok[b, h].item()):
                            continue
                        st = max(0, min(int(hs[b, h].item()), S))
                        ed = max(0, min(int(he[b, h].item()), S))
                        if ed <= st:
                            continue
                        self._block_key_span(attn2, b, st, ed)
                        if self.self_only_drop_input_ids:
                            ids[b, st:ed] = drop_token_id

                self_inputs = dict(inputs)
                self_inputs["input_ids"] = ids
                self_inputs["attention_mask"] = attn2
                self_inputs["labels"] = None
                _, _, logits_s, _, _ = self._forward_get_logits(model, self_inputs)
                logp_self[keep_h] = F.log_softmax(logits_s, dim=-1)

        shift = int(self.label_shift)
        gamma = float(self.longce_gamma)

        token_weighted_sum = torch.zeros((), device=device, dtype=torch.float32)
        token_den = torch.zeros((), device=device, dtype=torch.float32)
        head_num = torch.zeros((H,), device=device, dtype=torch.float32)
        head_den = torch.zeros((H,), device=device, dtype=torch.float32)

        lsd_sum_h = torch.zeros((H,), device=device, dtype=torch.float32)
        clip_cnt_h = torch.zeros((H,), device=device, dtype=torch.float32)
        tok_cnt_h = torch.zeros((H,), device=device, dtype=torch.float32)

        for b in range(B):
            for h in range(H):
                if not bool(hok[b, h].item()):
                    continue
                st = max(0, min(int(hs[b, h].item()), S))
                ed = max(0, min(int(he[b, h].item()), S))
                if ed <= st or (st - shift < 0):
                    continue

                tgt = labels[b, st:ed]
                valid = (tgt != self.IGNORE_TOKEN_ID)
                if not valid.any():
                    continue

                pred_all = torch.arange(st - shift, ed - shift, device=device)
                pred_idx = pred_all[valid]
                tgt_ids = tgt[valid].to(device)
                n_tok = int(tgt_ids.numel())
                if n_tok <= 0:
                    continue

                lp_f = logp_full[b, pred_idx, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
                ce_full = -lp_f
                lp_s = logp_self[h][b, pred_idx, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    lsd = lp_f.detach() - lp_s
                    if self.isoft_positive_only:
                        lsd = lsd.clamp_min(0.0)
                    w = torch.exp(lsd).clamp(max=gamma)
                    if self.isoft_floor_one:
                        w = w.clamp_min(1.0)
                    if self.isoft_per_head_mean1:
                        w = w / w.mean().clamp_min(1e-6)
                    clipped = (lsd >= math.log(max(gamma, 1e-6))).float()

                num_add = (w * ce_full).sum()
                den_add = ce_full.new_tensor(float(n_tok), dtype=torch.float32)

                token_weighted_sum = token_weighted_sum + num_add
                token_den = token_den + den_add
                head_num[h] = head_num[h] + num_add
                head_den[h] = head_den[h] + den_add

                lsd_sum_h[h] += lsd.sum()
                clip_cnt_h[h] += clipped.sum()
                tok_cnt_h[h] += den_add

        if self.balance_mode == "head":
            valid_h = (head_den > 0)
            if valid_h.any():
                head_loss_local = head_num[valid_h] / head_den[valid_h].clamp_min(1.0)
                loss = head_loss_local.mean()
            else:
                loss = torch.zeros((), device=device, dtype=torch.float32)
            with torch.no_grad():
                tokden_global = self.accelerator.reduce(token_den.detach(), reduction="sum").clamp_min(1.0)
        else:
            with torch.no_grad():
                tokden_global = self.accelerator.reduce(token_den.detach(), reduction="sum").clamp_min(1.0)
            world_size = float(getattr(self.accelerator, "num_processes", 1))
            loss = token_weighted_sum * world_size / tokden_global

        with torch.no_grad():
            g_head_den = self.accelerator.reduce(head_den.detach(), reduction="sum").clamp_min(1.0)
            g_head_num = self.accelerator.reduce(head_num.detach(), reduction="sum")
            g_lsd_sum = self.accelerator.reduce(lsd_sum_h.detach(), reduction="sum")
            g_clip_cnt = self.accelerator.reduce(clip_cnt_h.detach(), reduction="sum")
            g_tok_cnt = self.accelerator.reduce(tok_cnt_h.detach(), reduction="sum").clamp_min(1.0)

            head_loss_global = g_head_num / g_head_den
            head_lsd_mean = g_lsd_sum / g_head_den
            head_clip_frac = g_clip_cnt / g_tok_cnt
            clip_frac_overall = float((g_clip_cnt.sum() / g_tok_cnt.sum().clamp_min(1.0)).item())

        safe_log: Dict[str, Any] = {
            "longce_channel_on": 1,
            "longce_prob_now": float(p_now),
            "balance_mode": self.balance_mode,
            "loss_global": float(loss.detach().item()),
            "tokden_global": float(tokden_global.detach().item()),
            "gamma": float(gamma),
            "label_shift": int(shift),
            "lsd_mean": float((g_lsd_sum.sum() / g_tok_cnt.sum().clamp_min(1.0)).item()),
            "isoft_pos_only": int(self.isoft_positive_only),
            "isoft_floor1": int(self.isoft_floor_one),
            "isoft_head_mean1": int(self.isoft_per_head_mean1),
            "clip_frac_overall": clip_frac_overall,
            "self_only_mask_user": int(self.self_only_mask_user),
            "self_only_drop_input_ids": int(self.self_only_drop_input_ids),
            "neg_inf": float(self.neg_inf),
        }
        for h in range(H):
            safe_log[f"head{h}_tokden_global"] = float(g_head_den[h].item())
            safe_log[f"head{h}_loss_global"] = float(head_loss_global[h].item())
            safe_log[f"head{h}_lsd_mean"] = float(head_lsd_mean[h].item())
            safe_log[f"head{h}_clip_frac"] = float(head_clip_frac[h].item())
        self._maybe_log(safe_log)

        return (loss, output_full) if return_outputs else loss
