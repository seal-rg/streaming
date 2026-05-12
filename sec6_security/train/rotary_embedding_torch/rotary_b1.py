#!/usr/bin/env python3

"""
channel_phase_rope.py

A trainable, HF/Qwen2-compatible RoPE module implementing B1/E1:
Channel-conditioned phase bias (a.k.a. per-channel SO(2)^m rotation).

Key points:
- NO @torch.no_grad() on forward (so delta_phi is trainable).
- Outputs (cos, sin) exactly like Qwen2RotaryEmbedding: [B, N, head_dim].
- Accepts position_ids as:
    - [B, N, 2] : (cid, y)  (recommended)
    - [B, N]    : y only (fallback, cid=0)
- Uses bounded phase bias for inference stability:
    phase_c = alpha * tanh(delta_phi[cid])
  where alpha can be scalar or per-frequency (recommended).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelPhaseRoPE(nn.Module):
    """
    B1/E1: channel-conditioned phase bias RoPE

    phase_k(y,c) = (inv_freq_k * y) + alpha_k * tanh(delta_phi_k(c))

    Returns:
      cos, sin: [B, N, head_dim] (pair-wise interleaved, same as HF/Qwen2)
    """

    def __init__(
        self,
        head_dim: int,
        num_channels: int,
        base: float = 10000.0,
        alpha_init: float = 0.1,
        per_frequency_alpha: bool = True,  # ✅ recommended True
        learnable_alpha: bool = False,  # ✅ start False for stability; switch True later
        alpha_max: float = 0.2,  # cap if learnable_alpha=True
        channel_init: str = "zeros",  # "zeros" or "randn"
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")

        self.head_dim = int(head_dim)
        self.half_dim = head_dim // 2
        self.num_channels = int(num_channels)
        self.base = float(base)

        # ---- inv_freq (HF/Qwen2-style): [half_dim] ----
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.half_dim, dtype=dtype, device=device) / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # ---- trainable per-channel phase bias table: [C, half_dim] ----
        if channel_init == "zeros":
            table = torch.zeros(self.num_channels, self.half_dim, dtype=dtype, device=device)
        elif channel_init == "randn":
            table = torch.randn(self.num_channels, self.half_dim, dtype=dtype, device=device) * 0.01
        else:
            raise ValueError(f"Unknown channel_init={channel_init}")
        self.delta_phi = nn.Parameter(table)  # ✅ trainable

        # ---- alpha (scalar or per-frequency) ----
        self.per_frequency_alpha = bool(per_frequency_alpha)
        self.learnable_alpha = bool(learnable_alpha)
        self.alpha_max = float(alpha_max)

        if self.per_frequency_alpha:
            # alpha: [half_dim]
            if self.learnable_alpha:
                # raw_alpha -> bounded (0, alpha_max) via sigmoid
                self.raw_alpha = nn.Parameter(torch.zeros(self.half_dim, dtype=dtype, device=device))
                self.register_buffer("alpha", torch.empty(0, dtype=dtype, device=device), persistent=False)
            else:
                self.raw_alpha = None
                self.register_buffer(
                    "alpha",
                    torch.full((self.half_dim,), float(alpha_init), dtype=dtype, device=device),
                    persistent=False,
                )
        else:
            # scalar alpha
            if self.learnable_alpha:
                self.raw_alpha = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))
                self.register_buffer("alpha", torch.empty(0, dtype=dtype, device=device), persistent=False)
            else:
                self.raw_alpha = None
                self.register_buffer("alpha", torch.tensor(float(alpha_init), dtype=dtype, device=device), persistent=False)

    def _get_alpha(self) -> torch.Tensor:
        """
        Returns:
          - scalar tensor if per_frequency_alpha=False
          - [half_dim] tensor if per_frequency_alpha=True
        """
        if self.learnable_alpha:
            # bounded in (0, alpha_max)
            return self.alpha_max * torch.sigmoid(self.raw_alpha)
        return self.alpha

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: only used for device/dtype context (like HF Qwen2RotaryEmbedding)
        position_ids:
          - [B, N, 2] where (cid, y)
          - or [B, N] where y only (cid assumed 0)
        Returns:
          cos, sin: [B, N, head_dim] in x.dtype
        """
        if position_ids.ndim == 2:
            # fallback: y only
            y = position_ids
            cid = torch.zeros_like(y, dtype=torch.long)
        elif position_ids.ndim == 3 and position_ids.size(-1) == 2:
            cid = position_ids[..., 0]
            y = position_ids[..., 1]
        else:
            raise ValueError(f"bad position_ids shape {tuple(position_ids.shape)}; expected [B,N] or [B,N,2]")

        # [B,N]
        cid = cid.long().clamp(0, self.num_channels - 1)
        y = y.to(torch.float32)

        # HF-style: inv_freq_expanded [B, half_dim, 1], y_expanded [B,1,N]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(y.shape[0], -1, 1)
        y_expanded = y[:, None, :].float()

        # match HF: disable autocast inside rope math for numerical stability
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs_y: [B,N,half_dim]
            freqs_y = (inv_freq_expanded @ y_expanded).transpose(1, 2)

            # delta: [B,N,half_dim], bounded
            delta = torch.tanh(self.delta_phi[cid])

            alpha = self._get_alpha()
            if self.per_frequency_alpha:
                # alpha: [half_dim] -> broadcast
                freqs_c = delta * alpha.view(1, 1, -1)
            else:
                # scalar alpha
                freqs_c = delta * alpha

            phase = freqs_y + freqs_c  # [B,N,half_dim]

            # HF/Qwen2: emb = cat((freqs,freqs)) -> [B,N,head_dim]
            emb = torch.cat((phase, phase), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# -------------------------
# Minimal integration hints
# -------------------------
#
# 1) In Qwen2Model.__init__ (or your model init):
#
#   head_dim = config.hidden_size // config.num_attention_heads
#   self.rotary_emb = ChannelPhaseRotaryEmbedding(
#       head_dim=head_dim,
#       num_channels=getattr(config, "num_channels", 4),
#       base=getattr(config, "rope_theta", 10000.0),
#       alpha_init=0.1,
#       per_frequency_alpha=True,     # recommended
#       learnable_alpha=False,        # start False
#       channel_init="zeros",
#   )
#
# 2) In Qwen2Model.forward, compute and pass position_embeddings=(cos,sin):
#
#   cos, sin = self.rotary_emb(hidden_states, position_ids)   # position_ids: [B,N,2]
#   position_embeddings = (cos, sin)
#
# 3) In Qwen2Attention / Qwen2SdpaAttention:
#   if position_embeddings is not None:
#       cos, sin = position_embeddings
#       query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#
# This keeps cache.update kwargs {"sin":sin,"cos":cos,...} compatible.
#
