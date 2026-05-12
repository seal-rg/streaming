"""
StreamQwen3_5Moe — Qwen3.5 MoE (35B-A3B) with stream-specific modifications:
  - Channel embedding (additive)
  - Channel-aware SDPA via shared module (only on full_attention layers)
  - Role gating support
  - Custom position_ids handling (mRoPE with [B,S,2] -> [cid, y])

Identical to the dense variant except:
  - Uses Qwen3_5MoeSparseMoeBlock instead of Qwen3_5MLP
  - MoE-specific config parameters

GatedDeltaNet (linear attention) layers use block-causal delta rule functions.
"""

import functools

import torch
from stream_arch.role_gating import RoleGatingMLP
from stream_arch.sdpa_attention import sdpa_attention_forward
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

# Import base components from the MoE transformers module
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDynamicCache,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_qwen3_5_moe import StreamQwen3_5MoeTextConfig


class StreamQwen3_5MoeAttention(nn.Module):
    """Qwen3.5 MoE full attention with stream modifications.
    Identical to the dense StreamQwen3_5Attention but uses MoE config/norm classes.
    """

    def __init__(self, config: StreamQwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        # Q projects to 2x width: first half = query, second half = output gate
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
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

        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ---- role gating config ----
        self.role_gating_enabled = bool(getattr(config, "role_gating_enabled", False))
        self.num_roles = int(getattr(config, "num_channels", 3))
        self.role_gating_granularity = str(
            getattr(config, "role_gating_granularity", "layer")
        )
        self.role_gating_mode = str(getattr(config, "role_gating_mode", "query"))
        self.role_gating_mlp_hidden = int(getattr(config, "role_gating_mlp_hidden", 0))
        self.role_gating_tau = float(getattr(config, "role_gating_tau", 2.0))
        self.role_gating_beta_max = float(getattr(config, "role_gating_beta_max", 0.8))
        self.role_gating_log_eps = float(getattr(config, "role_gating_log_eps", 1e-4))
        self.role_gating_log_clip_min = float(
            getattr(config, "role_gating_log_clip_min", -6.0)
        )
        self.role_gating_uniform_mix = float(
            getattr(config, "role_gating_uniform_mix", 0.05)
        )

        self.gate_in_norm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.role_gating_enabled:
            self.role_gating = RoleGatingMLP(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_roles=self.num_roles,
                granularity=self.role_gating_granularity,
                mode=self.role_gating_mode,
                mlp_hidden=self.role_gating_mlp_hidden,
            )

    def _prefix_ctx_summary(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        csum = x.cumsum(dim=1)
        denom = torch.arange(1, S + 1, device=x.device, dtype=x.dtype)[None, :, None]
        return csum / denom

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        channel_ids: torch.LongTensor | None = None,
        channel_ids_kv: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        gate_in = self.gate_in_norm(hidden_states)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # ---- role gating ----
        g = None
        beta = None
        channel_ids_k = None

        if self.role_gating_enabled and (channel_ids is not None):
            if channel_ids_kv is not None:
                channel_ids_k = channel_ids_kv
            elif past_key_values is None:
                channel_ids_k = channel_ids
            else:
                raise ValueError(
                    "channel_ids_kv required in cache mode with role gating"
                )

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

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            g=g,
            channel_ids_k=channel_ids_k,
            beta=beta,
            log_eps=self.role_gating_log_eps,
            log_clip_min=self.role_gating_log_clip_min,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class StreamQwen3_5MoeDecoderLayer(GradientCheckpointingLayer):
    """Decoder layer using StreamQwen3_5MoeAttention for full_attention,
    GatedDeltaNet for linear_attention, and SparseMoeBlock for MLP."""

    def __init__(self, config: StreamQwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]  # type: ignore
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)  # type: ignore
            num_ch = getattr(config, "num_channels", 10)
            # Patch delta rule: "block_causal" (shared state) | "column" (per-column) | "standard" (no patch)
            dn_mode = str(getattr(config, "deltanet_block_causal", "block_causal"))
            if dn_mode in ("true", "True", "block_causal"):
                from stream_arch.block_causal_deltanet import (
                    block_causal_chunk_gated_delta_rule,
                    block_causal_recurrent_gated_delta_rule,
                )

                self.linear_attn.chunk_gated_delta_rule = functools.partial(
                    block_causal_chunk_gated_delta_rule,
                    chunk_size=num_ch,
                )
                self.linear_attn.recurrent_gated_delta_rule = functools.partial(
                    block_causal_recurrent_gated_delta_rule,
                    num_channels=num_ch,
                )
            elif dn_mode == "column":
                from stream_arch.block_causal_deltanet import (
                    column_chunk_gated_delta_rule,
                    column_recurrent_gated_delta_rule,
                )

                self.linear_attn.chunk_gated_delta_rule = functools.partial(
                    column_chunk_gated_delta_rule,
                    chunk_size=num_ch,
                )
                self.linear_attn.recurrent_gated_delta_rule = functools.partial(
                    column_recurrent_gated_delta_rule,
                    num_channels=num_ch,
                )
            # Optionally wrap conv1d to prevent same-row channel leakage
            conv_mode = getattr(config, "deltanet_conv", "column")
            if conv_mode != "standard":
                from stream_arch.block_causal_deltanet import BlockCausalConv1d

                self.linear_attn.conv1d = BlockCausalConv1d(  # type: ignore
                    self.linear_attn.conv1d,
                    num_ch,
                    mode=conv_mode,
                )
                self.linear_attn.causal_conv1d_fn = None
            # Stream state management params for inference
            K = self.linear_attn.conv_kernel_size
            self._has_block_causal_conv = conv_mode != "standard"
            self._conv_state_len = (K - 1) * num_ch if self._has_block_causal_conv else K - 1
        elif self.layer_type == "full_attention":
            self.self_attn = StreamQwen3_5MoeAttention(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.mlp._z3_leaf = True  # ZeRO-3 leaf: gather all MoE params atomically
        self.input_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        channel_ids: torch.LongTensor | None = None,
        channel_ids_kv: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            if hidden_states.shape[1] == 0:
                return hidden_states  # skip empty sequences (GatedDeltaNet can't reshape seq_len=0) # type: ignore
            from stream_arch.block_causal_deltanet import stream_forward_gated_deltanet

            hidden_states = stream_forward_gated_deltanet(
                self.linear_attn,
                hidden_states,
                past_key_values,
                cache_position,
                attention_mask,
                self._conv_state_len,
                self._has_block_causal_conv,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                channel_ids=channel_ids,
                channel_ids_kv=channel_ids_kv,
                **kwargs,  # type: ignore
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states  # type: ignore


class StreamQwen3_5MoePreTrainedModel(PreTrainedModel):
    config: StreamQwen3_5MoeTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StreamQwen3_5MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]
    _is_stateful = True


class StreamQwen3_5MoeTextModel(StreamQwen3_5MoePreTrainedModel):
    """Qwen3.5 MoE text model with channel embedding and stream position handling."""

    def __init__(self, config: StreamQwen3_5MoeTextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList(
            [
                StreamQwen3_5MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.num_channels = getattr(config, "num_channels", 3)
        self.channel_embedding_method = getattr(
            config, "channel_embedding_method", "additive"
        )
        if self.channel_embedding_method != "none":
            self.channel_embedding = nn.Embedding(self.num_channels, config.hidden_size)

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
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5MoeDynamicCache(config=self.config)  # type: ignore

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],  # type: ignore
                device=inputs_embeds.device,  # type: ignore
            )

        # ---- Stream position handling ----
        channel_ids = kwargs.pop("channel_ids", None)

        if (
            position_ids is not None
            and position_ids.dim() == 3
            and position_ids.size(-1) == 2
        ):
            channel_ids = position_ids[..., 0].long()
            local_y = position_ids[..., 1].long()
            mrope_position_ids = local_y[None, ...].expand(3, local_y.shape[0], -1)
        else:
            if position_ids is None:
                mrope_position_ids = cache_position.view(1, 1, -1).expand(  # type: ignore
                    3,
                    inputs_embeds.shape[0],  # type: ignore
                    -1,
                )
            elif position_ids.ndim == 2:
                mrope_position_ids = position_ids[None, ...].expand(
                    3, position_ids.shape[0], -1
                )
            elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
                mrope_position_ids = position_ids
            else:
                mrope_position_ids = position_ids[None, ...].expand(
                    3, position_ids.shape[0], -1
                )

            if channel_ids is None:
                if position_ids is not None and position_ids.ndim == 2:
                    channel_ids = torch.zeros_like(
                        position_ids[:, : inputs_embeds.shape[1]],  # type: ignore
                        dtype=torch.long,
                    )
                else:
                    channel_ids = torch.zeros(
                        inputs_embeds.shape[0],  # type: ignore
                        inputs_embeds.shape[1],  # type: ignore
                        dtype=torch.long,
                        device=inputs_embeds.device,  # type: ignore
                    )

        channel_ids = channel_ids.contiguous().clamp(0, self.num_channels - 1)  # type: ignore

        if mrope_position_ids.ndim == 3 and mrope_position_ids.shape[0] == 4:
            text_position_ids = mrope_position_ids[0]
            mrope_position_ids = mrope_position_ids[1:]
        else:
            text_position_ids = mrope_position_ids[0]

        # If the collator already built a dict of 4D masks, use them directly.
        if isinstance(attention_mask, dict):
            causal_mask = attention_mask.get("full_attention", None)
            linear_attn_mask = attention_mask.get(
                "linear_attention", attention_mask.get("sliding_attention", None)
            )
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,  # type: ignore
                attention_mask=attention_mask,
                cache_position=cache_position,  # type: ignore
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
            linear_attn_mask = self._update_linear_attn_mask(
                attention_mask, cache_position
            )

        hidden_states = inputs_embeds
        if self.channel_embedding_method != "none":
            channel_emb = self.channel_embedding(channel_ids.to(self.channel_embedding.weight.device))
            hidden_states = hidden_states + channel_emb.to(hidden_states.device)

        position_embeddings = self.rotary_emb(hidden_states, mrope_position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = (
                linear_attn_mask
                if decoder_layer.layer_type == "linear_attention"
                else causal_mask
            )

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=mrope_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                channel_ids=channel_ids,
                **kwargs,  # type: ignore
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask


@auto_docstring
class StreamQwen3_5MoeForCausalLM(StreamQwen3_5MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}  # type: ignore

    def __init__(self, config):
        super().__init__(config)
        self.model = StreamQwen3_5MoeTextModel(config)
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
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,  # type: ignore
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # type: ignore

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
