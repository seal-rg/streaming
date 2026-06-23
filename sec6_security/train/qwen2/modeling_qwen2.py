# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2 model."""

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from .configuration_qwen2 import Qwen2StreamConfig as Qwen2Config

if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B"
_CONFIG_FOR_DOC = "Qwen2Config"


from typing import Any


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Qwen2Config | None = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len, **self.rope_kwargs)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ============================================================
# RoleGatingMLP
# ============================================================
class RoleGatingMLP(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_roles: int, granularity: str = "layer", mode: str = "query", mlp_hidden: int = 0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_roles = num_roles
        self.granularity = granularity
        self.mode = mode

        in_dim = hidden_size if mode == "query" else (hidden_size * 2)
        out_dim = num_roles if granularity == "layer" else (num_heads * num_roles)

        if mlp_hidden and mlp_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden, bias=True),
                nn.SiLU(),
                nn.Linear(mlp_hidden, out_dim, bias=True),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, q_hidden: torch.Tensor, ctx: torch.Tensor | None = None):
        if self.mode == "query_ctx":
            assert ctx is not None
            x = torch.cat([q_hidden, ctx], dim=-1)
        else:
            x = q_hidden

        y = self.net(x)
        B, Q, _ = y.shape

        if self.granularity == "layer":
            return y.view(B, Q, self.num_roles)
        else:
            return y.view(B, Q, self.num_heads, self.num_roles).permute(0, 2, 1, 3).contiguous()


# ============================================================
# Qwen2Attention with role gating
# ============================================================
class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Role gating config
        self.role_gating_enabled = bool(getattr(config, "role_gating_enabled", False))
        if self.role_gating_enabled:
            self.gate_in_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.num_roles = int(getattr(config, "num_channels", 3))
            self.role_gating_granularity = str(getattr(config, "role_gating_granularity", "layer"))
            self.role_gating_mode = str(getattr(config, "role_gating_mode", "query"))
            self.role_gating_mlp_hidden = int(getattr(config, "role_gating_mlp_hidden", 0))
            self.role_gating_tau = float(getattr(config, "role_gating_tau", 2.0))
            self.role_gating_beta_max = float(getattr(config, "role_gating_beta_max", 0.8))
            self.role_gating_log_eps = float(getattr(config, "role_gating_log_eps", 1e-4))
            self.role_gating_log_clip_min = float(getattr(config, "role_gating_log_clip_min", -6.0))
            self.role_gating_uniform_mix = float(getattr(config, "role_gating_uniform_mix", 0.05))

            self.role_gating = RoleGatingMLP(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_roles=self.num_roles,
                granularity=self.role_gating_granularity,
                mode=self.role_gating_mode,
                mlp_hidden=self.role_gating_mlp_hidden,
            )

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

    def _prefix_ctx_summary(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,S,D] -> ctx: [B,S,D], ctx[t] = mean_{<=t}(x)"""
        B, S, D = x.shape
        csum = x.cumsum(dim=1)
        denom = torch.arange(1, S + 1, device=x.device, dtype=x.dtype)[None, :, None]
        return csum / denom

    def _apply_role_gating_bias(
        self,
        attn_weights: torch.Tensor,
        g: torch.Tensor,
        channel_ids_k: torch.LongTensor,
        beta: float,
    ) -> torch.Tensor:
        """Apply role gating bias using log-prior method."""
        B, H, Q, K = attn_weights.shape

        if g.dim() == 3:
            idx = channel_ids_k[:, None, :].expand(B, Q, K)
            g_qk = torch.gather(g, dim=-1, index=idx)
            log_bias = torch.log(g_qk + self.role_gating_log_eps).clamp(min=self.role_gating_log_clip_min, max=0.0)
            attn_weights = attn_weights + beta * log_bias[:, None, :, :]
        else:
            idx = channel_ids_k[:, None, :].expand(B, Q, K)
            idx = idx[:, None, :, :].expand(B, H, Q, K)
            g_qk = torch.gather(g, dim=-1, index=idx)
            log_bias = torch.log(g_qk + self.role_gating_log_eps).clamp(min=self.role_gating_log_clip_min, max=0.0)
            attn_weights = attn_weights + beta * log_bias

        return attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        channel_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Compute role gating prior
        g = None
        beta = None

        if self.role_gating_enabled and channel_ids is not None:
            hidden_states = self.gate_in_norm(hidden_states)
            if self.role_gating_mode == "query_ctx":
                ctx = self._prefix_ctx_summary(hidden_states)
                g_logits = self.role_gating(hidden_states, ctx=ctx)
            else:
                g_logits = self.role_gating(hidden_states, ctx=None)

            g = torch.softmax(g_logits / self.role_gating_tau, dim=-1)

            if self.role_gating_uniform_mix > 0:
                lam = self.role_gating_uniform_mix
                uni = torch.full_like(g, 1.0 / float(self.num_roles))
                g = (1.0 - lam) * g + lam * uni

            beta = self.role_gating_beta_max

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply role gating bias before causal mask
        if g is not None:
            attn_weights = self._apply_role_gating_bias(attn_weights, g, channel_ids, beta)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2FlashAttention2(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        channel_ids: torch.LongTensor | None = None,
        **kwargs,
    ):
        # Fallback to eager if role gating is active
        if self.role_gating_enabled and channel_ids is not None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                channel_ids=channel_ids,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2SdpaAttention(Qwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        channel_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:

        # Fallback to eager if output_attentions or role gating is active
        if output_attentions or (self.role_gating_enabled and channel_ids is not None):
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `output_attentions=True` or role gating is active. "
                "Falling back to the manual attention implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                channel_ids=channel_ids,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        if causal_mask is not None and causal_mask.dtype != query_states.dtype:
            causal_mask = causal_mask.to(query_states.dtype)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_type = "ori"  # "rope_axial" #"ori"

        self.num_channels = getattr(config, "num_channels", 3)  # system, user, assistant0-5
        self.channel_embedding = nn.Embedding(self.num_channels, config.hidden_size)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # self.rotary_emb = LierePositionEmbedding(
        #     head_dim= config.hidden_size // config.num_attention_heads,
        #     generator_dim=getattr(config, 'liere_generator_dim', 2),
        #     phase_type=getattr(config, 'liere_phase_type', 'learned'),
        #     rotary_embedding_per_head=getattr(config, 'liere_per_head', False),
        #     num_heads=config.num_attention_heads,
        #     max_position=getattr(config, 'max_position_embeddings', 32768),
        # )

        # self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            raise RuntimeError("rope_axial: position_ids is None (will become 1D and break 2D RoPE)")

            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        #######
        hidden_states = inputs_embeds

        channel_ids = position_ids[..., 0]  # [B, N]
        local_y = position_ids[..., 1]  # [B, N]

        channel_emb = self.channel_embedding(channel_ids)  # [B, N, hidden_size]
        hidden_states = inputs_embeds + channel_emb
        #######

        position_embeddings = self.rotary_emb(inputs_embeds, local_y)  # position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    channel_ids=channel_ids,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not (using_static_cache or using_sliding_window_cache) and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Qwen2
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Qwen2 Model transformer with a sequence classification head on top (linear layer).

    [`Qwen2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    QWEN2_START_DOCSTRING,
)

# -----------------------------
# Small utilities
# -----------------------------
def _as_list(x: torch.Tensor | list[int]) -> list[int]:
    if isinstance(x, list):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)


def _minf(dtype: torch.dtype) -> float:
    # Use a stable "very negative" value compatible with additive masks
    # (avoid -inf in some kernels)
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


# -----------------------------
# -----------------------------
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


def compute_span_ce_sum_single_head_fast(
    logits: torch.Tensor,  # [B,S,V]
    labels: torch.Tensor,  # [B,S]
    span_start: torch.Tensor,  # [B]    0-based inclusive
    span_end: torch.Tensor,  # [B]    0-based exclusive
) -> dict[str, Any]:
    """
    Compute sum CE loss for a single span per sample (no mean here).
    Uses full-sequence shift + mask; span [start, end) in label space
    maps to [start-1, end-1) in prediction space.
    """
    B, S, V = logits.shape
    device = logits.device

    # next-token alignment
    pred = logits[:, :-1, :]  # [B,S-1,V]
    tgt = labels[:, 1:].to(device)  # [B,S-1]

    # shift span to prediction space: [start-1, end-1)
    st = (span_start.to(device) - 1).clamp(0, S - 1)
    ed = (span_end.to(device) - 1).clamp(0, S - 1)

    pos = torch.arange(S - 1, device=device)[None, :]  # [1,S-1]
    mask = (pos >= st[:, None]) & (pos < ed[:, None]) & (tgt != -100)  # [B,S-1]

    n_tokens = int(mask.sum().item())
    if n_tokens == 0:
        return {
            "sum_loss": torch.zeros((), device=device, dtype=torch.float32),
            "num_tokens": 0,
        }

    # single CE call over all valid tokens
    sum_loss = F.cross_entropy(pred[mask], tgt[mask], reduction="sum").float()

    return {
        "sum_loss": sum_loss,
        "num_tokens": n_tokens,
    }


def get_span_from_boundaries_single(
    bd: dict,
    S: int,
    supervise_im_end: bool = True,
) -> tuple[int, int] | None:
    """
    Extract a single assistant span from a boundaries dict.
    Returns 0-based [start, end) or None if unavailable.
    """
    if not isinstance(bd, dict):
        return None

    all_heads = bd.get("all_heads", [])
    ahidx = bd.get("assistant_head_indices", [])
    if not isinstance(all_heads, list) or not isinstance(ahidx, list) or len(ahidx) < 1:
        return None

    # single head: take assistant_head_indices[0]
    try:
        hid = int(ahidx[0])
    except Exception:
        return None
    if hid < 0 or hid >= len(all_heads):
        return None

    h = all_heads[hid]
    if not isinstance(h, dict):
        return None

    start = int(h.get("assistant_solution_start", h.get("content_start", -1)))
    end = int(h.get("real_content_end", h.get("content_end", -1)))

    # clamp to valid 0-based range
    start = max(0, start)
    end = min(max(end, 0), S)

    if supervise_im_end:
        ie = int(h.get("im_end_pos", -1))
        if 0 <= ie < S:
            end = min(max(end, ie + 1), S)

    if end <= start:
        return None
    return start, end


class Qwen2ForMultiStream(Qwen2PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)

        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # You hard-coded path in your snippet; keep it, but you can pass via env/arg too.
        self.tokenizer = AutoTokenizer.from_pretrained(
            # "${MODELS_ROOT}/Qwen--Qwen2.5-7B-Instruct-medusa-inserts"
            "${MODELS_ROOT}/Qwen--Qwen2.5-7B/snapshots/medusa"
            # "${MODELS_ROOT}/Qwen--Qwen2.5-7B-Instruct/snapshots/medusa"
            # "${MODELS_ROOT}/Qwen--Qwen2.5-32B/snapshots/medusa"
        )

        self.num_streams = config.num_streams
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
        self.newline_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.max_position_embeddings = config.max_position_embeddings

        # role tokens
        self.system_tokens = self.tokenizer.encode("system", add_special_tokens=False)
        self.user_tokens = self.tokenizer.encode("user", add_special_tokens=False)
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
        self.post_init()

    # ---- embedding plumbing ----
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # NOTE: you disabled lm_head; so these are not used.
    # Kept for HF API compatibility if needed.
    def get_output_embeddings(self):
        return getattr(self, "lm_head", None)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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

        logits_bsv = output[0]
        return sum_loss, num_tokens, logits_bsv, output, log_dict

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_start: torch.Tensor | None = None,
        head_end: torch.Tensor | None = None,
        head_ok: torch.Tensor | None = None,
        boundaries: list[dict] | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **loss_kwargs,
    ) -> tuple | CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]  # [B,S,H]
        logits = self.lm_head(hidden_states)

        sum_loss_total = torch.zeros((), device=logits.device, dtype=torch.float32)
        num_tokens_total = torch.zeros((), device=logits.device, dtype=torch.float32)
        log_dict: dict[str, Any] = {}

        if labels is not None:
            B, S = labels.shape
            device = logits.device

            # ============================================================
            # CASE A: tensor spans are [B,H]  -> multi-head supervised loss
            # ============================================================
            if head_start is not None and head_end is not None and head_start.dim() == 2 and head_end.dim() == 2:
                hs = head_start.to(device).long()  # [B,H]
                he = head_end.to(device).long()  # [B,H]
                H = hs.size(1)

                if head_ok is None:
                    hok = torch.ones((B, H), device=device, dtype=torch.bool)
                else:
                    hok = head_ok.to(device).bool()
                    if hok.dim() == 1:
                        hok = hok[:, None].expand(B, H)

                hs = hs.clamp(0, S)
                he = he.clamp(0, S)

                # next-token alignment
                pred = logits[:, :-1, :]  # [B,S-1,V]
                tgt = labels[:, 1:].to(device)  # [B,S-1]
                S1 = S - 1
                V = pred.size(-1)

                st = (hs - 1).clamp(0, S1)  # [B,H]
                ed = (he - 1).clamp(0, S1)  # [B,H]

                pos = torch.arange(S1, device=device)[None, None, :]  # [1,1,S-1]
                mask = (pos >= st[:, :, None]) & (pos < ed[:, :, None])  # [B,H,S-1]
                mask = mask & hok[:, :, None]
                mask = mask & (tgt[:, None, :] != -100)

                n_tokens = int(mask.sum().item())
                if n_tokens > 0:
                    pred_bh = pred[:, None, :, :].expand(B, H, S1, V)  # [B,H,S-1,V]
                    pred_sel = pred_bh[mask]  # [N,V]
                    tgt_sel = tgt[:, None, :].expand(B, H, S1)[mask]  # [N]
                    sum_loss_total = F.cross_entropy(pred_sel, tgt_sel, reduction="sum").float()
                    num_tokens_total = torch.tensor(float(n_tokens), device=device, dtype=torch.float32)

                    denom = max(1, n_tokens)
                    log_dict["loss_local_mean"] = float(sum_loss_total.detach().item() / denom)
                    log_dict["tokens_local_total"] = float(n_tokens)
                    log_dict["sum_loss_local_total"] = float(sum_loss_total.detach().item())
                else:
                    sum_loss_total = torch.zeros((), device=device, dtype=torch.float32)
                    num_tokens_total = torch.zeros((), device=device, dtype=torch.float32)
                    log_dict["loss_local_mean"] = 0.0
                    log_dict["tokens_local_total"] = 0.0
                    log_dict["sum_loss_local_total"] = 0.0

            # ============================================================
            # CASE B: tensor spans are [B] or [B,1] -> single-span loss
            # ============================================================
            else:
                span_start = None
                span_end = None

                if head_start is not None and head_end is not None:
                    hs = head_start
                    he = head_end
                    if hs.dim() == 2 and hs.size(1) == 1:
                        hs = hs[:, 0]
                    if he.dim() == 2 and he.size(1) == 1:
                        he = he[:, 0]
                    span_start = hs.to(device).long()
                    span_end = he.to(device).long()

                    if head_ok is not None:
                        ok = head_ok.to(device).bool()
                        if ok.dim() == 2 and ok.size(1) == 1:
                            ok = ok[:, 0]
                        span_start = torch.where(ok, span_start, torch.zeros_like(span_start))
                        span_end = torch.where(ok, span_end, torch.zeros_like(span_end))

                elif boundaries is not None:
                    st_list = []
                    ed_list = []
                    for b in range(B):
                        sp = get_span_from_boundaries_single(boundaries[b], S, supervise_im_end=True)
                        if sp is None:
                            st_list.append(0)
                            ed_list.append(0)
                        else:
                            st_list.append(int(sp[0]))
                            ed_list.append(int(sp[1]))
                    span_start = torch.tensor(st_list, device=device, dtype=torch.long)
                    span_end = torch.tensor(ed_list, device=device, dtype=torch.long)

                if span_start is not None and span_end is not None:
                    out_loss = compute_span_ce_sum_single_head_fast(
                        logits=logits,
                        labels=labels,
                        span_start=span_start,
                        span_end=span_end,
                    )
                    sum_loss_total = out_loss["sum_loss"]
                    n_tok = int(out_loss["num_tokens"])
                    num_tokens_total = torch.tensor(float(n_tok), device=device, dtype=torch.float32)

                    denom = max(1, n_tok)
                    log_dict["loss_local_mean"] = float(sum_loss_total.detach().item() / denom) if n_tok > 0 else 0.0
                    log_dict["tokens_local_total"] = float(num_tokens_total.detach().item())
                    log_dict["sum_loss_local_total"] = float(sum_loss_total.detach().item())

        if not return_dict:
            output = (logits,) + outputs[1:]
            if labels is not None:
                return sum_loss_total, num_tokens_total, output, log_dict
            return output

        mean_loss = None
        if labels is not None:
            mean_loss = sum_loss_total / num_tokens_total.clamp_min(1.0)

        return CausalLMOutputWithPast(
            loss=mean_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ----------------------------
    # Token selection
    # ----------------------------
    def _select_next_token(
        self,
        logits_1d: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> int:
        """
        Robust next-token sampling.

        Fixes CUDA device-side assert by guaranteeing multinomial() receives:
        - finite probs
        - non-negative probs
        - sum(probs) > 0

        Fallback strategy:
        If anything becomes invalid (all -inf, NaN, sum<=0), return argmax(original logits).
        """
        assert logits_1d.dim() == 1, "logits_1d must be 1D [vocab]"

        # Always keep a safe greedy fallback index from ORIGINAL logits
        # (use nan_to_num to avoid argmax on all-NaN)
        safe_logits_for_argmax = torch.nan_to_num(logits_1d, nan=-1e30, posinf=1e30, neginf=-1e30)
        fallback_id = int(torch.argmax(safe_logits_for_argmax, dim=-1).item())

        if not do_sample:
            return fallback_id

        # ---- temperature (do in float32 for numerical stability) ----
        if temperature is None or temperature <= 0:
            temperature = 1.0

        logits = logits_1d.float()
        logits = torch.nan_to_num(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
        logits = logits / float(temperature)

        # If logits are all -inf-like (very negative), softmax may underflow badly.
        # Still ok, but if everything is effectively invalid, fallback.
        if not torch.isfinite(logits).any():
            return fallback_id

        # ---- top-k ----
        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), logits.size(-1))
            kth_vals, _ = torch.topk(logits, k)
            cutoff = kth_vals[-1]
            logits = torch.where(
                logits < cutoff,
                torch.full_like(logits, -float("inf")),
                logits,
            )

        # ---- top-p (nucleus) ----
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)

            # softmax on sorted logits (may become NaN if all -inf)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            sorted_probs = torch.nan_to_num(sorted_probs, nan=0.0, posinf=0.0, neginf=0.0)

            cumprobs = torch.cumsum(sorted_probs, dim=-1)

            mask = cumprobs > float(top_p)
            mask[..., 0] = False  # keep at least one token

            sorted_logits = torch.where(
                mask,
                torch.full_like(sorted_logits, -float("inf")),
                sorted_logits,
            )

            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(0, sorted_idx, sorted_logits)

        # ---- probs ----
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)

        # If everything is filtered out / invalid, fallback
        s = probs.sum()
        if (not torch.isfinite(s)) or float(s.item()) <= 0.0:
            return fallback_id

        # ---- min-p ----
        if min_p is not None and float(min_p) > 0.0:
            mp = float(min_p)
            probs = torch.where(probs < mp, torch.zeros_like(probs), probs)

            s = probs.sum()
            if (not torch.isfinite(s)) or float(s.item()) <= 0.0:
                return fallback_id
            probs = probs / s
        else:
            probs = probs / s

        # Final guard (multinomial requires probs finite, >=0, sum>0)
        if (not torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0.0:
            return fallback_id

        token = torch.multinomial(probs, num_samples=1)
        return int(token.item())

    # ============================================================
    # Extract outputs (truncate at <|im_end|>)
    # ============================================================
    def _extract_outputs(
        self,
        head_token_positions: list[list[int]],
        generated_ids: torch.Tensor,  # [1, S_total]
        stop_at_im_end: bool = True,
        stop_at_eos: bool = False,
    ) -> dict[int, torch.Tensor]:
        assert generated_ids.dim() == 2 and generated_ids.size(0) == 1
        seq = generated_ids[0]
        S = seq.numel()
        device = seq.device

        eos_id = getattr(self.config, "eos_token_id", None)
        im_end = self.im_end

        outputs: dict[int, torch.Tensor] = {}
        for h, pos_list in enumerate(head_token_positions):
            valid_pos = [p for p in pos_list if 0 <= int(p) < S]
            if not valid_pos:
                outputs[h] = torch.empty((0,), dtype=torch.long, device=device)
                continue

            idx = torch.tensor(valid_pos, device=device, dtype=torch.long)
            toks = seq.index_select(0, idx)

            if stop_at_im_end:
                m = toks == im_end
                if torch.any(m):
                    first = int(torch.nonzero(m, as_tuple=False)[0].item())
                    toks = toks[: first + 1]

            if stop_at_eos and eos_id is not None:
                m = toks == eos_id
                if torch.any(m):
                    first = int(torch.nonzero(m, as_tuple=False)[0].item())
                    toks = toks[: first + 1]

            outputs[h] = toks

        return outputs

    @torch.no_grad()
    def generate_stream(
        self,
        question_text: str,
        assistant_prefix_text: str = "",
        max_new_tokens: int = 1024,
        max_steps: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 0,
        do_sample: bool = False,
        stop_on_im_end: bool = True,
        include_im_end_in_cross_channel: bool = True,
        allow_same_step_visible: bool = False,
        disable_assistant_cross_channel: bool = False,
    ) -> dict[int, torch.Tensor]:
        """
        2-stream interleaved generation: user tokens arrive one per step while
        the assistant stream generates in parallel. Returns {0: token_tensor}.
        """

        device = next(self.parameters()).device
        tokenizer = self.tokenizer

        # ---------- ids ----------
        im_start = int(self.im_start)
        im_end = int(self.im_end)
        nl = int(self.newline_token)

        sys_tokens = list(self.system_tokens)
        user_tokens = list(self.user_tokens)
        asst_tokens = list(self.assistant_tokens)

        system_prefix = [im_start] + sys_tokens + [nl]
        user_prefix = [im_start] + user_tokens + [nl]
        asst_prefix = [im_start] + asst_tokens + [nl]

        sys_msg = getattr(self, "system_message", "You are a helpful assistant.")
        sys_ids = tokenizer.encode(sys_msg, add_special_tokens=False)

        # user_stream: question tokens + im_end
        user_content_ids = tokenizer.encode(question_text, add_special_tokens=False)
        user_stream = user_content_ids + [im_end]

        # optional assistant prefill text prepended before generation
        asst_prefill_ids = tokenizer.encode(assistant_prefix_text, add_special_tokens=False) if assistant_prefix_text else []

        # Step0: prefill system (closed) + user prefix only + assistant prefix + optional prefill
        # step0_ids: system block closed; user block just opened (prefix only, no content, no im_end);
        #            assistant block opened (prefix + optional prefill)
        step0_ids: list[int] = []
        # system (closed)
        step0_ids += system_prefix + sys_ids + [im_end]
        # user (prefix only)
        step0_ids += user_prefix
        # assistant (open)
        step0_ids += asst_prefix + asst_prefill_ids

        input_ids0 = torch.tensor([step0_ids], device=device, dtype=torch.long)
        S0 = input_ids0.size(1)

        # ---------- lengths / offsets ----------
        # sys_len includes im_end
        sys_len = len(system_prefix) + len(sys_ids) + 1
        y0 = sys_len  # v2: user & assistants offset = sys_len

        user_prefix_len = len(user_prefix)  # local y within user prefix
        asst_prefix_len = len(asst_prefix)  # local y within assistant prefix
        asst_prefill_len = len(asst_prefill_ids)

        # ---------- Step0 position_ids (2D) ----------
        # token spans in step0:
        # [0, sys_len)             -> system cid=0, y=0..sys_len-1
        # [sys_len, sys_len+user_prefix_len) -> user cid=1, y=y0 + 0..user_prefix_len-1
        # [sys_len+user_prefix_len, end) -> assistant cid=2, y=y0 + 0..(asst_prefix_len+asst_prefill_len-1)
        pos0 = torch.zeros((S0, 2), device=device, dtype=torch.long)

        # system positions
        for i in range(0, sys_len):
            pos0[i, 0] = 0
            pos0[i, 1] = i

        # user prefix positions
        user_bs = sys_len
        user_pe = sys_len + user_prefix_len
        for j, i in enumerate(range(user_bs, user_pe)):
            pos0[i, 0] = 1
            pos0[i, 1] = y0 + j

        # assistant positions (prefix + optional prefill)
        asst_bs = user_pe
        for j, i in enumerate(range(asst_bs, S0)):
            pos0[i, 0] = 2  # only 1 assistant head => cid=2
            pos0[i, 1] = y0 + j

        token_y0 = pos0[:, 1].tolist()

        # ---------- cache owner bookkeeping ----------
        # owner: 0 system, 1 user, 2 assistant
        cache_owner0 = [0] * S0
        for i in range(user_bs, user_pe):
            cache_owner0[i] = 1
        for i in range(asst_bs, S0):
            cache_owner0[i] = 2

        # ---------- Step0 attention mask ----------
        def _minf(dtype: torch.dtype) -> float:
            return -1e9 if dtype == torch.float32 else -1e4

        eligible = set(range(S0))

        attn0 = torch.full((S0, S0), _minf(torch.float32), device=device, dtype=torch.float32)
        diag = torch.arange(S0, device=device)
        attn0[diag, diag] = 0.0

        # intra-block causal
        # system block: [0, sys_len)
        for q in range(0, sys_len):
            attn0[q, 0 : q + 1] = 0.0
        # user prefix block: [user_bs, user_pe)
        for q in range(user_bs, user_pe):
            attn0[q, user_bs : q + 1] = 0.0
        # assistant block: [asst_bs, S0)
        for q in range(asst_bs, S0):
            attn0[q, asst_bs : q + 1] = 0.0

        # cross-channel y-rule: allow y_key < y_query
        y_to_keys: dict[int, list[int]] = {}
        for j in eligible:
            y_to_keys.setdefault(int(token_y0[j]), []).append(j)
        ys_sorted = sorted(y_to_keys.keys())
        unique_yq = sorted({int(token_y0[q]) for q in eligible})
        cum: list[int] = []
        ptr = 0
        yq_to_vis: dict[int, list[int]] = {}
        for yq in unique_yq:
            while ptr < len(ys_sorted) and ys_sorted[ptr] < yq:
                cum.extend(y_to_keys[ys_sorted[ptr]])
                ptr += 1
            yq_to_vis[yq] = list(cum)

        for q in eligible:
            yq = int(token_y0[q])
            for j in yq_to_vis.get(yq, []):
                if disable_assistant_cross_channel and cache_owner0[q] == 2:
                    # assistant query: only see system(owner0) + user(owner1) + self(owner2) history
                    if cache_owner0[j] in (0, 1, 2):
                        attn0[q, j] = 0.0
                else:
                    attn0[q, j] = 0.0

        # Step0 forward
        out0 = self.model(
            input_ids=input_ids0,
            attention_mask=attn0.unsqueeze(0).unsqueeze(0),
            position_ids=pos0.unsqueeze(0),
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out0.past_key_values

        # ---------- generation bookkeeping ----------
        generated = input_ids0.clone()  # [1, S0]
        cache_len = S0
        cache_token_y: list[int] = token_y0[:]  # y for cached tokens
        cache_owner: list[int] = cache_owner0[:]  # owner for cached tokens

        # user streaming pointer
        user_ptr = 0  # next user_stream token index to feed

        # assistant generation
        asst_ctx_pos = S0 - 1  # last token in step0 is in assistant block
        hidden = out0.last_hidden_state[0, asst_ctx_pos]
        logits = self.lm_head(hidden)
        first_tok = int(
            self._select_next_token(
                logits_1d=logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        )

        head_token_positions: list[list[int]] = [[]]  # only head0
        pending_asst: int | None = first_tok
        asst_gen_fed = 0  # how many assistant generated tokens have been FED to cache
        asst_gen_count = 0  # how many assistant generated tokens have been SAMPLED (== appended to generated buffer)

        # append first sampled token to generated (NOT yet in cache until fed)
        generated = torch.cat([generated, torch.tensor([[first_tok]], device=device, dtype=torch.long)], dim=1)
        head_token_positions[0].append(generated.size(1) - 1)
        asst_gen_count += 1
        if stop_on_im_end and first_tok == im_end:
            pending_asst = None

        # ---------- main loop ----------
        for _ in range(max_steps):
            if asst_gen_count >= max_new_tokens:
                break
            if (user_ptr >= len(user_stream)) and (pending_asst is None):
                break

            active_kinds: list[str] = []
            feed_tokens: list[int] = []
            yq_list: list[int] = []
            cid_list: list[int] = []
            owner_list: list[int] = []

            # 1) feed one user token if any remaining
            if user_ptr < len(user_stream):
                tok_u = int(user_stream[user_ptr])
                active_kinds.append("user")
                feed_tokens.append(tok_u)

                # user y: y0 + user_prefix_len + user_ptr
                y_u = int(y0 + user_prefix_len + user_ptr)
                y_u = max(0, min(y_u, int(self.max_position_embeddings) - 1))
                yq_list.append(y_u)
                cid_list.append(1)
                owner_list.append(1)

            # 2) feed assistant pending token if exists
            if pending_asst is not None and asst_gen_count <= max_new_tokens:
                tok_a = int(pending_asst)
                active_kinds.append("asst")
                feed_tokens.append(tok_a)

                # assistant y: y0 + asst_prefix_len + asst_prefill_len + asst_gen_fed
                y_a = int(y0 + asst_prefix_len + asst_prefill_len + asst_gen_fed)
                y_a = max(0, min(y_a, int(self.max_position_embeddings) - 1))
                yq_list.append(y_a)
                cid_list.append(2)
                owner_list.append(2)

            if not feed_tokens:
                break

            Q = len(feed_tokens)
            inp = torch.tensor([feed_tokens], device=device, dtype=torch.long)  # [1,Q]

            pos_q = torch.tensor(list(zip(cid_list, yq_list)), device=device, dtype=torch.long).unsqueeze(0)  # [1,Q,2]

            # attention rows: [Q, cache_len+Q]
            key_len = cache_len + Q
            rows = torch.full((Q, key_len), _minf(torch.float32), device=device, dtype=torch.float32)

            for qi in range(Q):
                yq = int(yq_list[qi])
                q_owner = int(owner_list[qi])

                # cached keys visible if y_key < yq
                for j in range(cache_len):
                    if int(cache_token_y[j]) < yq:
                        if disable_assistant_cross_channel and q_owner == 2:
                            # assistant query: only see system/user + self history
                            if cache_owner[j] in (0, 1, 2):
                                rows[qi, j] = 0.0
                        else:
                            rows[qi, j] = 0.0

                # within-call
                for kj in range(Q):
                    if kj == qi:
                        rows[qi, cache_len + kj] = 0.0
                    elif allow_same_step_visible and (kj < qi):
                        rows[qi, cache_len + kj] = 0.0

            out = self.model(
                input_ids=inp,
                attention_mask=rows.unsqueeze(0).unsqueeze(0),
                position_ids=pos_q,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values

            # update cache bookkeeping (FED tokens are now in cache)
            cache_token_y.extend([int(y) for y in yq_list])
            cache_owner.extend([int(o) for o in owner_list])
            cache_len += Q

            # update pointers and sample next assistant token from the assistant-fed row (if present)
            # out.last_hidden_state: [1,Q,hidden]
            for i, kind in enumerate(active_kinds):
                if kind == "user":
                    # user token has been fed
                    user_ptr += 1
                else:
                    # assistant token has been fed; sample next
                    asst_gen_fed += 1
                    hidden_i = out.last_hidden_state[0, i]
                    logits_i = self.lm_head(hidden_i)
                    tok_next = int(
                        self._select_next_token(
                            logits_1d=logits_i,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                    )

                    # append sampled token to generated buffer; set pending
                    generated = torch.cat([generated, torch.tensor([[tok_next]], device=device, dtype=torch.long)], dim=1)
                    head_token_positions[0].append(generated.size(1) - 1)
                    asst_gen_count += 1

                    if stop_on_im_end and tok_next == im_end:
                        pending_asst = None
                    elif asst_gen_count >= max_new_tokens:
                        pending_asst = None
                    else:
                        pending_asst = tok_next

        return self._extract_outputs(
            head_token_positions=head_token_positions,
            generated_ids=generated,
            stop_at_im_end=True,
            stop_at_eos=False,
        )
