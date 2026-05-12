# type: ignore
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from stream_arch.role_gating import RoleGatingMLP
from torch import nn
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_qwen3 import Qwen3MedusaConfig
from .sdpa_attention import sdpa_attention_forward


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
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


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ---- role gating config ----
        self.role_gating_enabled = bool(getattr(config, "role_gating_enabled", False))
        self.num_roles = int(getattr(config, "num_channels", 3))
        self.role_gating_granularity = str(getattr(config, "role_gating_granularity", "layer"))  # "layer"|"head"
        self.role_gating_mode = str(getattr(config, "role_gating_mode", "query"))  # "query"|"query_ctx"
        self.role_gating_mlp_hidden = int(getattr(config, "role_gating_mlp_hidden", 0))

        self.role_gating_tau = float(getattr(config, "role_gating_tau", 2.0))
        self.role_gating_beta_max = float(getattr(config, "role_gating_beta_max", 0.8))
        self.role_gating_log_eps = float(getattr(config, "role_gating_log_eps", 1e-4))
        self.role_gating_log_clip_min = float(getattr(config, "role_gating_log_clip_min", -6.0))
        self.role_gating_uniform_mix = float(getattr(config, "role_gating_uniform_mix", 0.05))  # optional, e.g. 0.05

        self.gate_in_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.role_gating_enabled:
            self.role_gating = RoleGatingMLP(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_roles=self.num_roles,
                granularity=self.role_gating_granularity,
                mode=self.role_gating_mode,
                mlp_hidden=self.role_gating_mlp_hidden,
            )

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def _prefix_ctx_summary(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,S,D] -> ctx: [B,S,D], ctx[t] = mean_{<=t}(x)
        """
        B, S, D = x.shape
        csum = x.cumsum(dim=1)
        denom = torch.arange(1, S + 1, device=x.device, dtype=x.dtype)[None, :, None]
        return csum / denom

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position=None,
        channel_ids: torch.LongTensor | None = None,
        channel_ids_kv: torch.LongTensor | None = None,
        **kwargs,
    ):
        B, S, _ = hidden_states.shape
        hidden_shape = (B, S, -1, self.head_dim)

        # ✅ gating input uses RMSNorm'ed hidden
        gate_in = self.gate_in_norm(hidden_states)  # [B,S,D]

        # ---- standard attention ----
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B,H,S,d]
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B,kvH,S,d]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,kvH,S,d]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ---- gating prior ----
        g = None
        beta = None
        channel_ids_k = None

        if self.role_gating_enabled and (channel_ids is not None):
            if channel_ids_kv is not None:
                channel_ids_k = channel_ids_kv  # [B,K] in cache mode
            else:
                # no-cache fallback only
                if past_key_value is None:
                    channel_ids_k = channel_ids  # [B,S] == [B,K]
                else:
                    raise ValueError()

            # compute g (query-side): use gate_in (RMSNormed)
            if self.role_gating_mode == "query_ctx":
                ctx = self._prefix_ctx_summary(gate_in)
                g_logits = self.role_gating(gate_in, ctx=ctx)
            else:
                g_logits = self.role_gating(gate_in, ctx=None)

            g = torch.softmax(g_logits / self.role_gating_tau, dim=-1)

            lam = float(self.role_gating_uniform_mix)
            if lam > 0:
                uni = torch.full_like(g, 1.0 / float(self.num_roles))
                g = (1.0 - lam) * g + lam * uni

            beta = hidden_states.new_tensor(self.role_gating_beta_max)

        # ---- choose attention impl ----
        attention_interface = sdpa_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            g=g,
            channel_ids_k=channel_ids_k,
            beta=beta,
            log_eps=self.role_gating_log_eps,
            log_clip_min=self.role_gating_log_clip_min,
            **kwargs,
        )

        attn_output = attn_output.reshape(B, S, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3MedusaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]  # type: ignore

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention

        if attention_mask is not None and attention_mask.dtype.is_floating_point:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,  # type: ignores
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3MedusaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3MedusaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        if self.rope_type == "default":
            inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config, device)
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]  # type: ignore
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(config, device=None):
        base = getattr(config, "rope_theta", None)
        if base is None and hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            base = config.rope_scaling.get("rope_theta", 1000000)
        if base is None:
            base = 1000000
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3MedusaConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.num_channels = getattr(config, "num_channels", 3)  # system, user, assistant0-5
        self.channel_embedding_method = getattr(config, "channel_embedding_method", "additive")
        if self.channel_embedding_method != "none":
            self.channel_embedding = nn.Embedding(self.num_channels, config.hidden_size)

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # print(self.config.layer_types)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        channel_ids = kwargs.pop("channel_ids", None)  # allow explicit channel_ids override

        if position_ids is not None and position_ids.dim() == 3 and position_ids.size(-1) == 2:
            # your custom format: [B,S,2] = [cid, y]
            channel_ids = position_ids[..., 0].long()
            local_y = position_ids[..., 1].long()
        else:
            # HF default: [B,S] local positions
            if position_ids is None:
                local_y = cache_position.unsqueeze(0).long()
            else:
                local_y = position_ids.long()

            if channel_ids is None:
                channel_ids = torch.zeros_like(local_y, dtype=torch.long)

        # clamp channel ids into embedding range
        # channel_ids = channel_ids.clamp_(0, self.num_channels - 1)
        channel_ids = channel_ids.contiguous().clamp(0, self.num_channels - 1)

        # ------------------------------------------------------------
        # masks
        # ------------------------------------------------------------
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": local_y,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # Add channel embedding using channel_ids
        hidden_states = inputs_embeds
        if self.channel_embedding_method != "none":
            channel_emb = self.channel_embedding(channel_ids.to(self.channel_embedding.weight.device))  # [B,S,D]
            hidden_states = hidden_states + channel_emb.to(hidden_states.device)

        # RoPE uses local_y
        position_embeddings = self.rotary_emb(hidden_states, local_y)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=local_y,  #  only y
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                channel_ids=channel_ids,  #  pass channel ids down
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compute_span_ce_sum_single_head_fast(
    logits: torch.Tensor,  # [B,S,V]
    labels: torch.Tensor,  # [B,S]
    span_start: torch.Tensor,  # [B]    0-based inclusive
    span_end: torch.Tensor,  # [B]    0-based exclusive
) -> dict[str, Any]:

    B, S, V = logits.shape
    device = logits.device

    pred = logits[:, :-1, :]  # [B,S-1,V]
    tgt = labels[:, 1:].to(device)  # [B,S-1]

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

    if not isinstance(bd, dict):
        return None

    all_heads = bd.get("all_heads", [])
    ahidx = bd.get("assistant_head_indices", [])
    if not isinstance(all_heads, list) or not isinstance(ahidx, list) or len(ahidx) < 1:
        return None

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

    start = max(0, start)
    end = min(max(end, 0), S)

    if supervise_im_end:
        ie = int(h.get("im_end_pos", -1))

        if 0 <= ie < S:
            end = min(max(end, ie + 1), S)

    if end <= start:
        return None
    return start, end


class Qwen3ForMedusa(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        tokenizer_path = getattr(config, "tokenizer_path", None) or getattr(config, "_name_or_path", None) or "Qwen/Qwen3-4B"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.medusa_num_heads = config.medusa_num_heads
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
        self.newline_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.max_position_embeddings = config.max_position_embeddings

        # role tokens
        self.system_tokens = self.tokenizer.encode("system", add_special_tokens=False)
        self.user_tokens = self.tokenizer.encode("user", add_special_tokens=False)
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _forward_get_logits(
        self, model, _inputs
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        Any,
        dict[str, Any],
    ]:
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
        logits = self.lm_head(hidden_states)  # [B,S,V]

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

    def _select_next_token(
        self,
        logits_1d: torch.Tensor,  # [vocab]
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 0,
        min_p: float = 0.0,
        # ===== NEW =====
        presence_penalty: float = 0.0,
        past_tokens: Iterable[int] | None = None,
    ) -> int:

        assert logits_1d.dim() == 1, "logits_1d must be 1D [vocab]"

        # ---- safe greedy fallback (original logits) ----
        safe_logits = torch.nan_to_num(logits_1d, nan=-1e30, posinf=1e30, neginf=-1e30)
        fallback_id = int(torch.argmax(safe_logits).item())

        if not do_sample:
            return fallback_id

        if temperature is None or temperature <= 0:
            temperature = 1.0

        logits = logits_1d.float()
        logits = torch.nan_to_num(logits, nan=-1e30, posinf=1e30, neginf=-1e30)

        if presence_penalty and past_tokens:
            # unique token ids only (presence ≠ frequency)
            uniq = set(int(t) for t in past_tokens)
            if uniq:
                ids = torch.tensor(list(uniq), device=logits.device, dtype=torch.long)
                logits.index_put_(
                    (ids,),
                    logits.index_select(0, ids) - float(presence_penalty),
                )

        # ---- temperature ----
        logits = logits / float(temperature)

        if not torch.isfinite(logits).any():
            return fallback_id

        # ---- top-k ----
        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            kth_vals, _ = torch.topk(logits, k)
            cutoff = kth_vals[-1]
            logits = torch.where(
                logits < cutoff,
                torch.full_like(logits, -float("inf")),
                logits,
            )

        # ---- top-p ----
        if top_p and 0.0 < float(top_p) < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            sorted_probs = torch.nan_to_num(sorted_probs, nan=0.0)

            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumprobs > float(top_p)
            mask[..., 0] = False

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

        s = probs.sum()
        if not torch.isfinite(s) or s.item() <= 0:
            return fallback_id

        # ---- min-p ----
        if min_p and min_p > 0.0:
            probs = torch.where(probs < min_p, torch.zeros_like(probs), probs)
            s = probs.sum()
            if not torch.isfinite(s) or s.item() <= 0:
                return fallback_id
            probs = probs / s
        else:
            probs = probs / s

        if not torch.isfinite(probs).all():
            return fallback_id

        token = torch.multinomial(probs, 1)
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
    def medusa_generate_interleaved_v2_stream_user(
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
        # ===== NEW =====
        presence_penalty: float = 0.0,
    ) -> dict[int, torch.Tensor]:
        device = next(self.parameters()).device
        tokenizer = self.tokenizer

        # ===== presence history (assistant only) =====
        assistant_history: list[int] = []

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

        # user_stream: question tokens + im_end（把 im_end 也当成 user 的最后一个“到达 token”）
        user_content_ids = tokenizer.encode(question_text, add_special_tokens=False)
        user_stream = user_content_ids + [im_end]

        asst_prefill_ids = tokenizer.encode(assistant_prefix_text, add_special_tokens=False) if assistant_prefix_text else []

        # ---------- Step0: cache 里只放 system(full) + user_prefix_only + assistant_prefix_only(+asst_prefill) ----------
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

        # intra-block causal（按“块”来）
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
                min_p=0.0,
                # ===== presence penalty hook =====
                presence_penalty=presence_penalty,
                past_tokens=assistant_history,
            )
        )

        head_token_positions: list[list[int]] = [[]]  # only head0
        pending_asst: int | None = first_tok
        asst_gen_fed = 0  # how many assistant generated tokens have been FED to cache
        asst_gen_count = 0  # how many assistant generated tokens have been SAMPLED

        # append first sampled token
        assistant_history.append(first_tok)
        generated = torch.cat(
            [generated, torch.tensor([[first_tok]], device=device, dtype=torch.long)],
            dim=1,
        )
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
            for i, kind in enumerate(active_kinds):
                if kind == "user":
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
                            min_p=0.0,
                            # ===== presence penalty hook =====
                            presence_penalty=presence_penalty,
                            past_tokens=assistant_history,
                        )
                    )

                    # record as assistant history (presence semantics)
                    assistant_history.append(tok_next)

                    # append sampled token to generated buffer; set pending
                    generated = torch.cat(
                        [
                            generated,
                            torch.tensor([[tok_next]], device=device, dtype=torch.long),
                        ],
                        dim=1,
                    )
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
