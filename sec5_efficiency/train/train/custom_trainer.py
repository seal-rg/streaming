#!/usr/bin/env python3
import logging
import math
import random
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import trl
from torch import nn
from torch.optim import AdamW
from transformers.trainer_pt_utils import get_parameter_names

logger = logging.getLogger(__name__)


class CustomizedTrainer(trl.SFTTrainer):
    """Multi-stream SFT trainer with backbone/head LR split and optional LongCE.

    LongCE reweights each token's CE loss by how much the full-context model
    outperforms a self-only baseline — tokens that benefit from cross-stream
    attention get higher weight.
    """

    def __init__(self, *args, **kwargs):
        self.backbone_lr = float(kwargs.pop("backbone_lr", 1e-5))
        self.head_lr = float(kwargs.pop("head_lr", 2e-5))
        self.head_param_keywords = tuple(kwargs.pop(
            "head_param_keywords",
            ("channel_embedding", "lm_head"),
        ))

        self.IGNORE_TOKEN_ID = int(kwargs.pop("ignore_token_id", -100))
        self.label_shift = int(kwargs.pop("label_shift", 1))

        # LongCE schedule
        self.enable_longce = bool(kwargs.pop("enable_longce_channel", True))
        self.longce_prob = float(kwargs.pop("longce_prob", 0.5))
        self.longce_gamma = float(kwargs.pop("longce_gamma", 2.0))
        self.longce_warmup_steps = int(kwargs.pop("longce_warmup_steps", 100))
        self.longce_ramp_steps = int(kwargs.pop("longce_ramp_steps", 2000))

        # LongCE weight shaping
        self.longce_positive_only = bool(kwargs.pop("isoft_positive_only", True))
        self.longce_floor_one = bool(kwargs.pop("isoft_floor_one", False))
        self.longce_normalize_weights = bool(kwargs.pop("isoft_per_head_mean1", True))

        # LongCE self-only masking
        self.longce_mask_user = bool(kwargs.pop("self_only_mask_user", True))
        self.longce_drop_tokens = bool(kwargs.pop("self_only_drop_input_ids", True))
        self.attn_neg_inf = float(kwargs.pop("neg_inf", -1e4))

        self.balance_mode = str(kwargs.pop("balance_mode", "head")).strip().lower()
        if self.balance_mode not in {"token", "head"}:
            self.balance_mode = "head"

        self.log_every_steps = int(kwargs.pop("log_every_steps", 1))
        super().__init__(*args, **kwargs)

    # ── Optimizer ────────────────────────────────────────────────────────────

    def get_optimizer_grouped_parameters(self, model: nn.Module):
        decay_names = get_parameter_names(model, [nn.LayerNorm])
        decay_names = [n for n in decay_names if not n.endswith("bias")]
        wd = float(self.args.weight_decay)

        bb_decay, bb_nodecay, hd_decay, hd_nodecay = [], [], [], []
        bb_n = hd_n = 0

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
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
            groups.append({"params": bb_decay,   "weight_decay": wd,  "lr": self.backbone_lr})
        if bb_nodecay:
            groups.append({"params": bb_nodecay, "weight_decay": 0.0, "lr": self.backbone_lr})
        if hd_decay:
            groups.append({"params": hd_decay,   "weight_decay": wd,  "lr": self.head_lr})
        if hd_nodecay:
            groups.append({"params": hd_nodecay, "weight_decay": 0.0, "lr": self.head_lr})

        self._group_numel = {"backbone": bb_n, "head": hd_n}
        return groups

    def create_optimizer(self):
        if self.optimizer is None:
            model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
            groups = self.get_optimizer_grouped_parameters(model)
            self.optimizer = AdamW(
                groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            g = getattr(self, "_group_numel", {"backbone": 0, "head": 0})
            logger.warning(
                "optimizer groups: backbone=%d params (lr=%.3e), head=%d params (lr=%.3e), keys=%s",
                g["backbone"], self.backbone_lr, g["head"], self.head_lr,
                list(self.head_param_keywords),
            )
            if g["head"] == 0:
                logger.warning("head param group is empty — check head_param_keywords.")
        return self.optimizer

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _maybe_log(self, log_dict: Dict[str, Any]):
        step = int(getattr(self.state, "global_step", 0))
        if self.log_every_steps <= 1 or step % self.log_every_steps == 0:
            self.log(log_dict)

    def _longce_prob_now(self) -> float:
        if not self.enable_longce:
            return 0.0
        step = int(getattr(self.state, "global_step", 0))
        if step < self.longce_warmup_steps:
            return 0.0
        if self.longce_ramp_steps <= 0:
            return self.longce_prob
        t = max(0.0, min(1.0, (step - self.longce_warmup_steps) / self.longce_ramp_steps))
        return self.longce_prob * t

    def _forward_get_logits(
        self, model, inputs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Any, Dict[str, Any]]:
        ret = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            position_ids=inputs.get("position_ids"),
            head_start=inputs.get("head_start"),
            head_end=inputs.get("head_end"),
            head_ok=inputs.get("head_ok"),
            return_dict=False,
        )
        if isinstance(ret, tuple) and len(ret) == 4 and torch.is_tensor(ret[0]):
            sum_loss, num_tokens, output, log_dict = ret
        else:
            sum_loss, num_tokens, output, log_dict = None, None, ret, {}
        return sum_loss, num_tokens, output[0], output, log_dict

    def _compute_base_loss(self, model, inputs, device):
        sum_loss, num_tokens, _, output, log_dict = self._forward_get_logits(model, inputs)

        sum_loss = (
            torch.tensor(0.0, device=device, dtype=torch.float32) if sum_loss is None
            else (torch.tensor(float(sum_loss), device=device, dtype=torch.float32) if not torch.is_tensor(sum_loss)
                  else sum_loss.float())
        )
        num_tokens = (
            torch.tensor(0.0, device=device, dtype=torch.float32) if num_tokens is None
            else (torch.tensor(float(num_tokens), device=device, dtype=torch.float32) if not torch.is_tensor(num_tokens)
                  else num_tokens.float())
        )

        with torch.no_grad():
            global_tokens = self.accelerator.reduce(num_tokens, reduction="sum").clamp_min(1.0)
        world_size = float(getattr(self.accelerator, "num_processes", 1))
        loss = sum_loss * world_size / global_tokens

        safe_log = dict(log_dict) if isinstance(log_dict, dict) else {}
        safe_log.update({
            "longce_on": 0,
            "longce_prob_now": float(self._longce_prob_now()),
            "loss_global": float(loss.detach().item()),
            "tokden_global": float(global_tokens.detach().item()),
        })
        self._maybe_log(safe_log)
        return loss, output

    # ── Main loss ─────────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        input_ids = inputs["input_ids"]
        labels    = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        head_start = inputs.get("head_start")
        head_end   = inputs.get("head_end")
        head_ok    = inputs.get("head_ok")
        user_start = inputs.get("user_start")
        user_end   = inputs.get("user_end")

        device = input_ids.device
        B, S = input_ids.shape

        do_longce = (
            labels is not None
            and head_start is not None
            and head_end is not None
            and head_ok is not None
            and attention_mask is not None
            and self.enable_longce
        )
        p_now = 0.0
        if do_longce:
            p_now = self._longce_prob_now()
            if random.random() >= p_now:
                do_longce = False

        if not do_longce:
            loss, output = self._compute_base_loss(model, inputs, device)
            return (loss, output) if return_outputs else loss

        hs  = head_start.to(device).long()
        he  = head_end.to(device).long()
        hok = head_ok.to(device).bool()
        H   = int(hs.shape[1])
        if H <= 0:
            loss, output = self._compute_base_loss(model, inputs, device)
            return (loss, output) if return_outputs else loss

        # Full-context forward (with grad)
        full_inputs = {**inputs, "labels": None}
        _, _, _, output_full, _ = self._forward_get_logits(model, full_inputs)
        logits_full = output_full[0]
        logp_full = F.log_softmax(logits_full, dim=-1)

        # Per-head self-only forward (no grad): mask out all other heads' keys
        pad_id = (
            getattr(getattr(self, "tokenizer", None), "pad_token_id", None)
            or getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
            or 0
        )
        logp_self = [None] * H
        with torch.no_grad():
            for keep_h in range(H):
                ids  = input_ids.clone()
                attn = attention_mask.clone()

                if self.longce_mask_user and user_start is not None and user_end is not None:
                    u_st = user_start.to(device).long()
                    u_ed = user_end.to(device).long()
                    for b in range(B):
                        st_u = max(0, min(int(u_st[b]), S))
                        ed_u = max(0, min(int(u_ed[b]), S))
                        if ed_u > st_u:
                            attn[b, 0, :, st_u:ed_u] = self.attn_neg_inf
                            if self.longce_drop_tokens:
                                ids[b, st_u:ed_u] = pad_id

                for b in range(B):
                    for h in range(H):
                        if h == keep_h or not bool(hok[b, h]):
                            continue
                        st = max(0, min(int(hs[b, h]), S))
                        ed = max(0, min(int(he[b, h]), S))
                        if ed > st:
                            attn[b, 0, :, st:ed] = self.attn_neg_inf
                            if self.longce_drop_tokens:
                                ids[b, st:ed] = pad_id

                self_inputs = {**inputs, "input_ids": ids, "attention_mask": attn, "labels": None}
                _, _, logits_s, _, _ = self._forward_get_logits(model, self_inputs)
                logp_self[keep_h] = F.log_softmax(logits_s, dim=-1)

        shift = self.label_shift
        gamma = self.longce_gamma

        token_weighted_sum = torch.zeros((), device=device, dtype=torch.float32)
        token_den = torch.zeros((), device=device, dtype=torch.float32)
        head_num  = torch.zeros((H,), device=device, dtype=torch.float32)
        head_den  = torch.zeros((H,), device=device, dtype=torch.float32)
        lsd_sum_h  = torch.zeros((H,), device=device, dtype=torch.float32)
        clip_cnt_h = torch.zeros((H,), device=device, dtype=torch.float32)
        tok_cnt_h  = torch.zeros((H,), device=device, dtype=torch.float32)

        for b in range(B):
            for h in range(H):
                if not bool(hok[b, h]):
                    continue
                st = max(0, min(int(hs[b, h]), S))
                ed = max(0, min(int(he[b, h]), S))
                if ed <= st or (st - shift) < 0:
                    continue

                tgt   = labels[b, st:ed]
                valid = tgt != self.IGNORE_TOKEN_ID
                if not valid.any():
                    continue

                pred_idx = torch.arange(st - shift, ed - shift, device=device)[valid]
                tgt_ids  = tgt[valid].to(device)
                n_tok    = int(tgt_ids.numel())
                if n_tok <= 0:
                    continue

                lp_f  = logp_full[b, pred_idx].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
                ce_f  = -lp_f
                lp_s  = logp_self[h][b, pred_idx].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    lsd = lp_f.detach() - lp_s
                    if self.longce_positive_only:
                        lsd = lsd.clamp_min(0.0)
                    w = torch.exp(lsd).clamp(max=gamma)
                    if self.longce_floor_one:
                        w = w.clamp_min(1.0)
                    if self.longce_normalize_weights:
                        w = w / w.mean().clamp_min(1e-6)
                    clipped = (lsd >= math.log(max(gamma, 1e-6))).float()

                num_add = (w * ce_f).sum()
                den_add = ce_f.new_tensor(float(n_tok), dtype=torch.float32)

                token_weighted_sum += num_add
                token_den  += den_add
                head_num[h] += num_add
                head_den[h] += den_add
                lsd_sum_h[h]  += lsd.sum()
                clip_cnt_h[h] += clipped.sum()
                tok_cnt_h[h]  += den_add

        with torch.no_grad():
            g_head_den  = self.accelerator.reduce(head_den.detach(),  reduction="sum").clamp_min(1.0)
            g_head_num  = self.accelerator.reduce(head_num.detach(),  reduction="sum")
            g_lsd_sum   = self.accelerator.reduce(lsd_sum_h.detach(), reduction="sum")
            g_clip_cnt  = self.accelerator.reduce(clip_cnt_h.detach(), reduction="sum")
            g_tok_cnt   = self.accelerator.reduce(tok_cnt_h.detach(), reduction="sum").clamp_min(1.0)
            tokden_global = self.accelerator.reduce(token_den.detach(), reduction="sum").clamp_min(1.0)

        if self.balance_mode == "head":
            valid_h = g_head_den > 0
            loss = (g_head_num[valid_h] / g_head_den[valid_h]).mean() if valid_h.any() else torch.zeros((), device=device)
        else:
            world_size = float(getattr(self.accelerator, "num_processes", 1))
            loss = token_weighted_sum * world_size / tokden_global

        head_loss_global = g_head_num / g_head_den
        safe_log: Dict[str, Any] = {
            "longce_on": 1,
            "longce_prob_now": float(p_now),
            "balance_mode": self.balance_mode,
            "loss_global": float(loss.detach().item()),
            "tokden_global": float(tokden_global.item()),
            "gamma": float(gamma),
            "lsd_mean": float((g_lsd_sum.sum() / g_tok_cnt.sum().clamp_min(1.0)).item()),
            "clip_frac": float((g_clip_cnt.sum() / g_tok_cnt.sum().clamp_min(1.0)).item()),
        }
        for h in range(H):
            safe_log[f"head{h}_tokden"] = float(g_head_den[h].item())
            safe_log[f"head{h}_loss"]   = float(head_loss_global[h].item())
            safe_log[f"head{h}_lsd"]    = float((g_lsd_sum[h] / g_tok_cnt[h].clamp_min(1.0)).item())
            safe_log[f"head{h}_clip"]   = float((g_clip_cnt[h] / g_tok_cnt[h].clamp_min(1.0)).item())
        self._maybe_log(safe_log)

        return (loss, output_full) if return_outputs else loss
