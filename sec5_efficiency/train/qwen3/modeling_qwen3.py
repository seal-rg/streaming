# type: ignore
# coding=utf-8
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs

from .configuration_qwen3 import Qwen3StreamConfig


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

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        B, S, _ = hidden_states.shape
        hidden_shape = (B, S, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA: expand KV heads to match query heads if needed
        if self.num_key_value_groups > 1:
            key_states   = key_states[:, :, None, :, :].expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(
                key_states.shape[0], -1, key_states.shape[2], key_states.shape[3])
            value_states = value_states[:, :, None, :, :].expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(
                value_states.shape[0], -1, value_states.shape[2], value_states.shape[3])

        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        attn_output = F.scaled_dot_product_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
        ).transpose(1, 2)
        attn_weights = None

        attn_output = attn_output.reshape(B, S, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3StreamConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3StreamConfig
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
    def __init__(self, config: Qwen3StreamConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3StreamConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.num_channels = getattr(config, "num_channels", 5)
        self.channel_embedding = nn.Embedding(self.num_channels, config.hidden_size)

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # channel_ids can be provided explicitly or packed into position_ids as [B,S,2]=[cid, y]
        channel_ids = kwargs.pop("channel_ids", None)

        if position_ids is not None and position_ids.dim() == 3 and position_ids.size(-1) == 2:
            channel_ids = position_ids[..., 0].long()
            local_y = position_ids[..., 1].long()
        else:
            if position_ids is None:
                local_y = cache_position.unsqueeze(0).long()
            else:
                local_y = position_ids.long()
            if channel_ids is None:
                channel_ids = torch.zeros_like(local_y, dtype=torch.long)

        channel_ids = channel_ids.contiguous().clamp(0, self.num_channels - 1)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": local_y,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        channel_emb = self.channel_embedding(channel_ids)
        hidden_states = hidden_states + channel_emb

        position_embeddings = self.rotary_emb(hidden_states, local_y)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=local_y,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                channel_ids=channel_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ── Span CE helpers (used by Qwen3ForMultiStream training forward) ──────────────────


def compute_span_ce_sum_single_head_fast(
    logits: torch.Tensor,   # [B,S,V]
    labels: torch.Tensor,   # [B,S]
    span_start: torch.Tensor,  # [B] 0-based inclusive
    span_end: torch.Tensor,    # [B] 0-based exclusive
) -> Dict[str, Any]:
    B, S, V = logits.shape
    device = logits.device

    pred = logits[:, :-1, :]        # [B,S-1,V]
    tgt  = labels[:, 1:].to(device) # [B,S-1]

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
    return {"sum_loss": sum_loss, "num_tokens": n_tokens}


def get_span_from_boundaries_single(
    bd: dict,
    S: int,
    supervise_im_end: bool = True,
) -> Optional[Tuple[int, int]]:
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
    end   = int(h.get("real_content_end", h.get("content_end", -1)))

    start = max(0, start)
    end = min(max(end, 0), S)

    if supervise_im_end:
        ie = int(h.get("im_end_pos", -1))
        if 0 <= ie < S:
            end = min(max(end, ie + 1), S)

    if end <= start:
        return None
    return start, end


# ── Multi-stream model (training + inference) ──────────────────────────────────


class Qwen3ForMultiStream(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        tokenizer_path = (
            getattr(config, "tokenizer_path", None)
            or getattr(config, "_name_or_path", None)
            or "Qwen/Qwen3-4B"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.num_streams = config.num_streams
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.newline_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.max_position_embeddings = config.max_position_embeddings

        self.system_tokens   = self.tokenizer.encode("system",    add_special_tokens=False)
        self.user_tokens     = self.tokenizer.encode("user",      add_special_tokens=False)
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)

        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_start: Optional[torch.Tensor] = None,   # [B,H] or [B] or [B,1]
        head_end: Optional[torch.Tensor] = None,
        head_ok: Optional[torch.Tensor] = None,
        boundaries: Optional[List[dict]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions   = output_attentions   if output_attentions   is not None else self.config.output_attentions
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

        hidden_states = outputs[0]           # [B,S,H]
        logits = self.lm_head(hidden_states) # [B,S,V]

        sum_loss_total = torch.zeros((), device=logits.device, dtype=torch.float32)
        num_tokens_total = torch.zeros((), device=logits.device, dtype=torch.float32)
        log_dict: Dict[str, Any] = {}

        if labels is not None:
            B, S = labels.shape
            device = logits.device

            # ── Case A: [B,H] spans → multi-head supervised loss ──────────────
            if head_start is not None and head_end is not None and head_start.dim() == 2 and head_end.dim() == 2:
                hs = head_start.to(device).long()  # [B,H]
                he = head_end.to(device).long()
                H = hs.size(1)

                if head_ok is None:
                    hok = torch.ones((B, H), device=device, dtype=torch.bool)
                else:
                    hok = head_ok.to(device).bool()
                    if hok.dim() == 1:
                        hok = hok[:, None].expand(B, H)

                hs = hs.clamp(0, S)
                he = he.clamp(0, S)

                pred = logits[:, :-1, :]             # [B,S-1,V]
                tgt  = labels[:, 1:].to(device)      # [B,S-1]
                S1 = S - 1
                V = pred.size(-1)

                st = (hs - 1).clamp(0, S1)
                ed = (he - 1).clamp(0, S1)

                pos = torch.arange(S1, device=device)[None, None, :]  # [1,1,S-1]
                mask = (pos >= st[:, :, None]) & (pos < ed[:, :, None])
                mask = mask & hok[:, :, None]
                mask = mask & (tgt[:, None, :] != -100)

                n_tokens = int(mask.sum().item())
                if n_tokens > 0:
                    pred_bh  = pred[:, None, :, :].expand(B, H, S1, V)
                    pred_sel = pred_bh[mask]
                    tgt_sel  = tgt[:, None, :].expand(B, H, S1)[mask]
                    sum_loss_total = F.cross_entropy(pred_sel, tgt_sel, reduction="sum").float()
                    num_tokens_total = torch.tensor(float(n_tokens), device=device, dtype=torch.float32)

                    denom = max(1, n_tokens)
                    log_dict["loss_local_mean"] = float(sum_loss_total.detach().item() / denom)
                    log_dict["tokens_local_total"] = float(n_tokens)
                    log_dict["sum_loss_local_total"] = float(sum_loss_total.detach().item())
                else:
                    sum_loss_total = torch.zeros((), device=device, dtype=torch.float32)
                    num_tokens_total = torch.zeros((), device=device, dtype=torch.float32)
                    log_dict.update({"loss_local_mean": 0.0, "tokens_local_total": 0.0, "sum_loss_local_total": 0.0})

            # ── Case B: [B] or [B,1] spans → single-span loss ────────────────
            else:
                span_start = span_end = None

                if head_start is not None and head_end is not None:
                    hs = head_start
                    he = head_end
                    if hs.dim() == 2 and hs.size(1) == 1:
                        hs = hs[:, 0]
                    if he.dim() == 2 and he.size(1) == 1:
                        he = he[:, 0]
                    span_start = hs.to(device).long()
                    span_end   = he.to(device).long()

                    if head_ok is not None:
                        ok = head_ok.to(device).bool()
                        if ok.dim() == 2 and ok.size(1) == 1:
                            ok = ok[:, 0]
                        span_start = torch.where(ok, span_start, torch.zeros_like(span_start))
                        span_end   = torch.where(ok, span_end,   torch.zeros_like(span_end))

                elif boundaries is not None:
                    st_list, ed_list = [], []
                    for b in range(B):
                        sp = get_span_from_boundaries_single(boundaries[b], S, supervise_im_end=True)
                        if sp is None:
                            st_list.append(0); ed_list.append(0)
                        else:
                            st_list.append(int(sp[0])); ed_list.append(int(sp[1]))
                    span_start = torch.tensor(st_list, device=device, dtype=torch.long)
                    span_end   = torch.tensor(ed_list, device=device, dtype=torch.long)

                if span_start is not None and span_end is not None:
                    out_loss = compute_span_ce_sum_single_head_fast(
                        logits=logits, labels=labels, span_start=span_start, span_end=span_end
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

    # ── Sampling ────────────────────────────────────────────────────────────────

    def _select_next_token(
        self,
        logits_1d: torch.Tensor,  # [vocab]
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        past_tokens: Optional[Iterable[int]] = None,
    ) -> int:
        assert logits_1d.dim() == 1, "logits_1d must be 1D [vocab]"

        safe_logits = torch.nan_to_num(logits_1d, nan=-1e30, posinf=1e30, neginf=-1e30)
        fallback_id = int(torch.argmax(safe_logits).item())

        if not do_sample:
            return fallback_id

        if temperature is None or temperature <= 0:
            temperature = 1.0

        logits = logits_1d.float()
        logits = torch.nan_to_num(logits, nan=-1e30, posinf=1e30, neginf=-1e30)

        if presence_penalty and past_tokens:
            uniq = set(int(t) for t in past_tokens)
            if uniq:
                ids = torch.tensor(list(uniq), device=logits.device, dtype=torch.long)
                logits.index_put_((ids,), logits.index_select(0, ids) - float(presence_penalty))

        logits = logits / float(temperature)

        if not torch.isfinite(logits).any():
            return fallback_id

        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            kth_vals, _ = torch.topk(logits, k)
            cutoff = kth_vals[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

        if top_p and 0.0 < float(top_p) < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            sorted_probs = torch.nan_to_num(sorted_probs, nan=0.0)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumprobs > float(top_p)
            mask[..., 0] = False
            sorted_logits = torch.where(mask, torch.full_like(sorted_logits, -float("inf")), sorted_logits)
            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)

        s = probs.sum()
        if not torch.isfinite(s) or s.item() <= 0:
            return fallback_id

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

    # ── Output extraction ────────────────────────────────────────────────────────

    def _extract_outputs(
        self,
        head_token_positions: List[List[int]],
        generated_ids: torch.Tensor,  # [1, S_total]
        stop_at_im_end: bool = True,
        stop_at_eos: bool = False,
    ) -> Dict[int, torch.Tensor]:
        assert generated_ids.dim() == 2 and generated_ids.size(0) == 1
        seq = generated_ids[0]
        S = seq.numel()
        device = seq.device

        eos_id = getattr(self.config, "eos_token_id", None)
        im_end = self.im_end

        outputs: Dict[int, torch.Tensor] = {}
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

    # ── 2-stream inference: one user + one assistant ─────────────────────────────

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
        presence_penalty: float = 0.0,
    ) -> Dict[int, torch.Tensor]:
        device = next(self.parameters()).device
        tokenizer = self.tokenizer

        assistant_history: List[int] = []

        im_start = int(self.im_start)
        im_end   = int(self.im_end)
        nl       = int(self.newline_token)

        system_prefix = [im_start] + list(self.system_tokens) + [nl]
        user_prefix   = [im_start] + list(self.user_tokens)   + [nl]
        asst_prefix   = [im_start] + list(self.assistant_tokens) + [nl]

        sys_msg = getattr(self, "system_message", "You are a helpful assistant.")
        sys_ids = tokenizer.encode(sys_msg, add_special_tokens=False)

        user_content_ids = tokenizer.encode(question_text, add_special_tokens=False)
        user_stream = user_content_ids + [im_end]

        asst_prefill_ids = tokenizer.encode(assistant_prefix_text, add_special_tokens=False) if assistant_prefix_text else []

        step0_ids: List[int] = system_prefix + sys_ids + [im_end] + user_prefix + asst_prefix + asst_prefill_ids

        input_ids0 = torch.tensor([step0_ids], device=device, dtype=torch.long)
        S0 = input_ids0.size(1)

        sys_len        = len(system_prefix) + len(sys_ids) + 1
        y0             = sys_len
        user_prefix_len = len(user_prefix)
        asst_prefix_len = len(asst_prefix)
        asst_prefill_len = len(asst_prefill_ids)

        pos0 = torch.zeros((S0, 2), device=device, dtype=torch.long)
        for i in range(sys_len):
            pos0[i, 0] = 0; pos0[i, 1] = i
        user_bs = sys_len; user_pe = sys_len + user_prefix_len
        for j, i in enumerate(range(user_bs, user_pe)):
            pos0[i, 0] = 1; pos0[i, 1] = y0 + j
        asst_bs = user_pe
        for j, i in enumerate(range(asst_bs, S0)):
            pos0[i, 0] = 2; pos0[i, 1] = y0 + j

        token_y0    = pos0[:, 1].tolist()
        cache_owner0 = [0] * S0
        for i in range(user_bs, user_pe):
            cache_owner0[i] = 1
        for i in range(asst_bs, S0):
            cache_owner0[i] = 2

        def _minf(dtype: torch.dtype) -> float:
            return -1e9 if dtype == torch.float32 else -1e4

        attn0 = torch.full((S0, S0), _minf(torch.float32), device=device, dtype=torch.float32)
        diag  = torch.arange(S0, device=device)
        attn0[diag, diag] = 0.0

        for q in range(sys_len):
            attn0[q, 0:q+1] = 0.0
        for q in range(user_bs, user_pe):
            attn0[q, user_bs:q+1] = 0.0
        for q in range(asst_bs, S0):
            attn0[q, asst_bs:q+1] = 0.0

        y_to_keys: Dict[int, List[int]] = {}
        for j in range(S0):
            y_to_keys.setdefault(int(token_y0[j]), []).append(j)
        ys_sorted = sorted(y_to_keys.keys())
        unique_yq = sorted({int(token_y0[q]) for q in range(S0)})
        cum: List[int] = []; ptr = 0
        yq_to_vis: Dict[int, List[int]] = {}
        for yq in unique_yq:
            while ptr < len(ys_sorted) and ys_sorted[ptr] < yq:
                cum.extend(y_to_keys[ys_sorted[ptr]]); ptr += 1
            yq_to_vis[yq] = list(cum)

        for q in range(S0):
            yq = int(token_y0[q])
            for j in yq_to_vis.get(yq, []):
                if disable_assistant_cross_channel and cache_owner0[q] == 2:
                    if cache_owner0[j] in (0, 1, 2):
                        attn0[q, j] = 0.0
                else:
                    attn0[q, j] = 0.0

        out0 = self.model(
            input_ids=input_ids0,
            attention_mask=attn0.unsqueeze(0).unsqueeze(0),
            position_ids=pos0.unsqueeze(0),
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out0.past_key_values

        generated   = input_ids0.clone()
        cache_len   = S0
        cache_token_y: List[int] = token_y0[:]
        cache_owner:   List[int] = cache_owner0[:]
        user_ptr = 0

        asst_ctx_pos = S0 - 1
        hidden = out0.last_hidden_state[0, asst_ctx_pos]
        logits = self.lm_head(hidden)

        first_tok = int(self._select_next_token(
            logits_1d=logits, do_sample=do_sample, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=0.0,
            presence_penalty=presence_penalty, past_tokens=assistant_history,
        ))

        head_token_positions: List[List[int]] = [[]]
        pending_asst: Optional[int] = first_tok
        asst_gen_fed = 0
        asst_gen_count = 0

        assistant_history.append(first_tok)
        generated = torch.cat([generated, torch.tensor([[first_tok]], device=device, dtype=torch.long)], dim=1)
        head_token_positions[0].append(generated.size(1) - 1)
        asst_gen_count += 1
        if stop_on_im_end and first_tok == im_end:
            pending_asst = None

        for _ in range(max_steps):
            if asst_gen_count >= max_new_tokens:
                break
            if (user_ptr >= len(user_stream)) and (pending_asst is None):
                break

            active_kinds: List[str] = []
            feed_tokens: List[int] = []
            yq_list: List[int] = []
            cid_list: List[int] = []
            owner_list: List[int] = []

            if user_ptr < len(user_stream):
                tok_u = int(user_stream[user_ptr])
                active_kinds.append("user"); feed_tokens.append(tok_u)
                y_u = max(0, min(int(y0 + user_prefix_len + user_ptr), int(self.max_position_embeddings) - 1))
                yq_list.append(y_u); cid_list.append(1); owner_list.append(1)

            if pending_asst is not None and asst_gen_count <= max_new_tokens:
                tok_a = int(pending_asst)
                active_kinds.append("asst"); feed_tokens.append(tok_a)
                y_a = max(0, min(int(y0 + asst_prefix_len + asst_prefill_len + asst_gen_fed), int(self.max_position_embeddings) - 1))
                yq_list.append(y_a); cid_list.append(2); owner_list.append(2)

            if not feed_tokens:
                break

            Q = len(feed_tokens)
            inp   = torch.tensor([feed_tokens], device=device, dtype=torch.long)
            pos_q = torch.tensor(list(zip(cid_list, yq_list)), device=device, dtype=torch.long).unsqueeze(0)

            key_len = cache_len + Q
            rows = torch.full((Q, key_len), _minf(torch.float32), device=device, dtype=torch.float32)

            for qi in range(Q):
                yq = int(yq_list[qi]); q_owner = int(owner_list[qi])
                for j in range(cache_len):
                    if int(cache_token_y[j]) < yq:
                        if disable_assistant_cross_channel and q_owner == 2:
                            if cache_owner[j] in (0, 1, 2):
                                rows[qi, j] = 0.0
                        else:
                            rows[qi, j] = 0.0
                for kj in range(Q):
                    if kj == qi:
                        rows[qi, cache_len + kj] = 0.0
                    elif allow_same_step_visible and kj < qi:
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
            cache_token_y.extend([int(y) for y in yq_list])
            cache_owner.extend([int(o) for o in owner_list])
            cache_len += Q

            for i, kind in enumerate(active_kinds):
                if kind == "user":
                    user_ptr += 1
                else:
                    asst_gen_fed += 1
                    hidden_i = out.last_hidden_state[0, i]
                    logits_i = self.lm_head(hidden_i)

                    tok_next = int(self._select_next_token(
                        logits_1d=logits_i, do_sample=do_sample, temperature=temperature,
                        top_p=top_p, top_k=top_k, min_p=0.0,
                        presence_penalty=presence_penalty, past_tokens=assistant_history,
                    ))
                    assistant_history.append(tok_next)
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

    # ── Multi-head inference: one user + H assistant heads ───────────────────────

    @torch.no_grad()
    def generate_multistream(
        self,
        question_text: str,
        assistant_heads: int = 4,
        assistant_prefix_texts=None,
        assistant_prefill_texts=None,
        max_new_tokens: int = 1024,
        max_steps: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 0,
        do_sample: bool = False,
        stop_on_im_end: bool = True,
        allow_same_step_visible: bool = False,
        presence_penalty: float = 0.0,
    ) -> Dict[int, torch.Tensor]:
        """Multi-head streaming user + interleaved generation.

        All assistant heads share the same y0 origin (y axes aligned).
        Cross-head visibility: strict y_key < y_query.
        Termination: when the last head (h=H-1) emits im_end, all other heads
        are immediately stopped and the main loop exits.

        Optimizations vs. a naive loop:
        - pre-allocated step buffers (feed, pos, rows) to avoid per-step allocs
        - per-owner sorted-y cache with searchsorted for O(log n) visibility
        - batched lm_head over all active assistant rows per step
        - masked_fill_ instead of torch.where + zeros_like
        """
        device = next(self.parameters()).device
        tokenizer = self.tokenizer
        H = int(assistant_heads)
        assert H >= 1
        last_head = H - 1

        def _norm_texts(x, default=""):
            if x is None:
                return [default] * H
            if isinstance(x, str):
                return [x] * H
            xs = list(x)
            if len(xs) == 1 and H > 1:
                return xs * H
            assert len(xs) == H, f"Expected len={H}, got len={len(xs)}"
            return xs

        assistant_prefix_texts  = _norm_texts(assistant_prefix_texts,  default="")
        assistant_prefill_texts = _norm_texts(assistant_prefill_texts, default="")

        im_start = int(self.im_start)
        im_end   = int(self.im_end)
        nl       = int(self.newline_token)

        system_prefix = [im_start] + list(self.system_tokens) + [nl]
        user_prefix   = [im_start] + list(self.user_tokens)   + [nl]
        asst_prefix   = [im_start] + list(self.assistant_tokens) + [nl]

        sys_msg = getattr(self, "system_message", "You are a helpful assistant.")
        sys_ids = tokenizer.encode(sys_msg, add_special_tokens=False)
        user_content_ids = tokenizer.encode(question_text, add_special_tokens=False)
        user_stream = user_content_ids + [im_end]

        head_prefix_ids:  List[List[int]] = []
        head_prefill_ids: List[List[int]] = []
        for h in range(H):
            ptxt = assistant_prefix_texts[h]
            ctxt = assistant_prefill_texts[h]
            head_prefix_ids.append(tokenizer.encode(ptxt, add_special_tokens=False) if ptxt else [])
            head_prefill_ids.append(tokenizer.encode(ctxt, add_special_tokens=False) if ctxt else [])

        step0_ids: List[int] = system_prefix + sys_ids + [im_end] + user_prefix
        head_block_ranges: List[Tuple[int, int]] = []
        for h in range(H):
            bs = len(step0_ids)
            step0_ids += asst_prefix + head_prefix_ids[h] + head_prefill_ids[h]
            head_block_ranges.append((bs, len(step0_ids)))

        input_ids0 = torch.tensor([step0_ids], device=device, dtype=torch.long)
        S0 = input_ids0.size(1)

        sys_len         = len(system_prefix) + len(sys_ids) + 1
        y0              = sys_len
        user_prefix_len = len(user_prefix)
        user_bs         = sys_len
        user_pe         = sys_len + user_prefix_len
        asst_prefix_len = len(asst_prefix)
        head_prefix_len  = [len(x) for x in head_prefix_ids]
        head_prefill_len = [len(x) for x in head_prefill_ids]

        pos0 = torch.zeros((S0, 2), device=device, dtype=torch.long)
        pos0[:sys_len, 0] = 0
        pos0[:sys_len, 1] = torch.arange(sys_len, device=device, dtype=torch.long)
        pos0[user_bs:user_pe, 0] = 1
        pos0[user_bs:user_pe, 1] = y0 + torch.arange(user_prefix_len, device=device, dtype=torch.long)
        for h in range(H):
            owner = 2 + h
            bs, be = head_block_ranges[h]
            block_len = be - bs
            if block_len <= 0:
                continue
            pos0[bs:be, 0] = owner
            pos0[bs:be, 1] = y0 + torch.arange(block_len, device=device, dtype=torch.long)

        token_y0 = pos0[:, 1].clone()

        def _minf(dtype: torch.dtype) -> float:
            return -1e9 if dtype == torch.float32 else -1e4

        mask_dtype = next(self.parameters()).dtype
        minf = _minf(mask_dtype)
        attn0 = torch.full((S0, S0), minf, device=device, dtype=mask_dtype)
        idx0     = torch.arange(S0, device=device)
        row_idx0 = idx0.unsqueeze(1)
        col_idx0 = idx0.unsqueeze(0)
        # cross-channel visibility: y_key < y_query
        attn0.masked_fill_(token_y0.unsqueeze(0) < token_y0.unsqueeze(1), 0.0)
        # intra-block causal
        attn0.masked_fill_((row_idx0 < sys_len) & (col_idx0 <= row_idx0), 0.0)
        attn0.masked_fill_((row_idx0 >= user_bs) & (row_idx0 < user_pe) & (col_idx0 >= user_bs) & (col_idx0 <= row_idx0), 0.0)
        for h in range(H):
            bs, be = head_block_ranges[h]
            if be <= bs:
                continue
            attn0.masked_fill_((row_idx0 >= bs) & (row_idx0 < be) & (col_idx0 >= bs) & (col_idx0 <= row_idx0), 0.0)
        attn0[idx0, idx0] = 0.0

        out0 = self.model(
            input_ids=input_ids0,
            attention_mask=attn0.unsqueeze(0).unsqueeze(0),
            position_ids=pos0.unsqueeze(0),
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        past_kv = out0.past_key_values

        # ── Pre-allocated buffers ─────────────────────────────────────────────
        max_cache_len = S0 + len(user_stream) + H * max_new_tokens
        num_owners    = H + 2

        # per-owner sorted-y cache for O(log n) visibility via searchsorted
        owner_pos_buf = torch.empty((num_owners, max_cache_len), device=device, dtype=torch.long)
        owner_y_buf   = torch.empty((num_owners, max_cache_len), device=device, dtype=torch.long)
        owner_lens    = [0] * num_owners
        owner_lens_t  = torch.zeros((num_owners,), device=device, dtype=torch.long)

        owner_ids0 = pos0[:, 0]
        for idx in range(S0):
            owner = int(owner_ids0[idx].item())
            off   = owner_lens[owner]
            owner_pos_buf[owner, off] = idx
            owner_y_buf[owner, off]   = token_y0[idx]
            owner_lens[owner]   = off + 1
            owner_lens_t[owner] = off + 1

        cache_len = S0
        generated_prefix: List[int] = step0_ids[:]
        generated_tail:   List[int] = []
        gen_len  = len(generated_prefix)
        user_ptr = 0

        assistant_histories:  List[List[int]]    = [[] for _ in range(H)]
        head_token_positions: List[List[int]]    = [[] for _ in range(H)]
        pending_asst:         List[Optional[int]] = [None] * H
        asst_gen_fed:         List[int]           = [0] * H
        asst_gen_count:       List[int]           = [0] * H
        local_prefix_total = [asst_prefix_len + head_prefix_len[h] + head_prefill_len[h] for h in range(H)]
        global_stop = False

        qmax     = 1 + H
        diag_idx = torch.arange(qmax, device=device)
        tri_prev = None
        if allow_same_step_visible and qmax > 1:
            tri_prev = torch.tril(torch.ones((qmax, qmax), device=device, dtype=torch.bool), diagonal=-1)

        # reusable per-step tensors
        feed_buf              = torch.empty((1, qmax), device=device, dtype=torch.long)
        pos_q_buf             = torch.empty((1, qmax, 2), device=device, dtype=torch.long)
        rows_buf              = torch.empty((qmax, max_cache_len + qmax), device=device, dtype=mask_dtype)
        assistant_row_idx_buf = torch.empty((H,), device=device, dtype=torch.long)
        head_ids_buf          = torch.empty((qmax,), device=device, dtype=torch.long)
        prefix_idx_buf        = torch.arange(max_cache_len, device=device, dtype=torch.long).unsqueeze(0)

        # ── Init: sample first token per head (batched lm_head) ──────────────
        ctx_positions = torch.tensor(
            [(be - 1) if be > bs else (S0 - 1) for bs, be in head_block_ranges],
            device=device, dtype=torch.long,
        )
        init_hidden = out0.last_hidden_state[0, ctx_positions]   # [H, D]
        init_logits = self.lm_head(init_hidden)                   # [H, V]

        for h in range(H):
            first_tok = int(self._select_next_token(
                logits_1d=init_logits[h], do_sample=do_sample,
                temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0,
                presence_penalty=presence_penalty, past_tokens=assistant_histories[h],
            ))
            assistant_histories[h].append(first_tok)
            generated_tail.append(first_tok)
            head_token_positions[h].append(gen_len)
            gen_len += 1
            asst_gen_count[h] += 1
            if stop_on_im_end and first_tok == im_end:
                pending_asst[h] = None
                if h == last_head:
                    global_stop = True
            else:
                pending_asst[h] = first_tok

        if global_stop:
            for hh in range(H):
                pending_asst[hh] = None

        # ── Main loop ─────────────────────────────────────────────────────────
        for _ in range(max_steps):
            if global_stop:
                break
            if user_ptr >= len(user_stream):
                if all((p is None or asst_gen_count[h] >= max_new_tokens) for h, p in enumerate(pending_asst)):
                    break

            q        = 0
            has_user = False

            if user_ptr < len(user_stream):
                has_user = True
                feed_buf[0, q] = int(user_stream[user_ptr])
                pos_q_buf[0, q, 0] = 1
                pos_q_buf[0, q, 1] = max(0, min(int(y0 + user_prefix_len + user_ptr), int(self.max_position_embeddings) - 1))
                head_ids_buf[q] = -1
                q += 1

            for h in range(H):
                if pending_asst[h] is None:
                    continue
                if asst_gen_count[h] >= max_new_tokens:
                    pending_asst[h] = None
                    continue
                feed_buf[0, q] = int(pending_asst[h])
                pos_q_buf[0, q, 0] = 2 + h
                pos_q_buf[0, q, 1] = max(
                    0, min(int(y0 + local_prefix_total[h] + asst_gen_fed[h]), int(self.max_position_embeddings) - 1)
                )
                head_ids_buf[q] = h
                q += 1

            if q == 0:
                break

            key_len = cache_len + q
            inp   = feed_buf[:, :q]
            pos_q = pos_q_buf[:, :q, :]
            yq_t  = pos_q[0, :, 1]

            # fast visibility via searchsorted per owner
            rows = rows_buf[:q, :key_len]
            rows.fill_(minf)
            for owner in range(num_owners):
                owner_len = int(owner_lens_t[owner].item())
                if owner_len <= 0:
                    continue
                owner_y  = owner_y_buf[owner, :owner_len]
                vis_lens = torch.searchsorted(owner_y, yq_t, right=False)
                max_vis  = int(vis_lens.max().item())
                if max_vis <= 0:
                    continue
                owner_pos   = owner_pos_buf[owner, :max_vis]
                prefix_mask = prefix_idx_buf[:, :max_vis] < vis_lens.unsqueeze(1)
                rows[:, owner_pos].masked_fill_(prefix_mask, 0.0)

            di = diag_idx[:q]
            rows[di, cache_len + di] = 0.0
            if allow_same_step_visible and q > 1:
                rows[:, cache_len:cache_len+q].masked_fill_(tri_prev[:q, :q], 0.0)

            out = self.model(
                input_ids=inp,
                attention_mask=rows.unsqueeze(0).unsqueeze(0),
                position_ids=pos_q,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values

            if cache_len + q > max_cache_len:
                raise RuntimeError(f"cache buffer overflow: {cache_len=}, {q=}, {max_cache_len=}")
            for i in range(q):
                owner = int(pos_q[0, i, 0].item())
                off   = owner_lens[owner]
                owner_pos_buf[owner, off] = cache_len + i
                owner_y_buf[owner, off]   = yq_t[i]
                owner_lens[owner]   = off + 1
                owner_lens_t[owner] = off + 1
            cache_len += q

            if has_user:
                user_ptr += 1

            # batch lm_head over all active assistant rows
            num_asst_rows = 0
            for i in range(q):
                h = int(head_ids_buf[i].item())
                if h < 0:
                    continue
                assistant_row_idx_buf[num_asst_rows] = i
                num_asst_rows += 1
                asst_gen_fed[h] += 1

            stop_triggered = False
            if num_asst_rows:
                row_idx_t    = assistant_row_idx_buf[:num_asst_rows]
                hidden_batch = out.last_hidden_state[0, row_idx_t]   # [A, D]
                logits_batch = self.lm_head(hidden_batch)              # [A, V]
                for j in range(num_asst_rows):
                    h = int(head_ids_buf[int(row_idx_t[j].item())].item())
                    tok_next = int(self._select_next_token(
                        logits_1d=logits_batch[j], do_sample=do_sample,
                        temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0,
                        presence_penalty=presence_penalty, past_tokens=assistant_histories[h],
                    ))
                    assistant_histories[h].append(tok_next)
                    generated_tail.append(tok_next)
                    head_token_positions[h].append(gen_len)
                    gen_len += 1
                    asst_gen_count[h] += 1

                    if stop_on_im_end and tok_next == im_end:
                        pending_asst[h] = None
                        if h == last_head:
                            stop_triggered = True
                    elif asst_gen_count[h] >= max_new_tokens:
                        pending_asst[h] = None
                    else:
                        pending_asst[h] = tok_next

            if stop_triggered:
                global_stop = True
                for hh in range(H):
                    pending_asst[hh] = None
                break

        all_ids = torch.tensor([generated_prefix + generated_tail], device=device, dtype=torch.long)
        return self._extract_outputs(
            head_token_positions=head_token_positions,
            generated_ids=all_ids,
            stop_at_im_end=True,
            stop_at_eos=False,
        )
