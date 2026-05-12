"""Channel-aware SDPA attention forward function.

Architecture-agnostic: works with any model that provides query/key/value tensors
and optional role gating parameters (g, channel_ids_k, beta).
"""

import torch
from transformers.utils import is_torch_xpu_available, logging
from transformers.utils.import_utils import is_torch_greater_or_equal

logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA: (batch, num_kv_heads, seqlen, head_dim) -> (batch, num_heads, seqlen, head_dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor) -> bool:
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8 and not isinstance(
            key, torch.fx.Proxy
        )
    return (
        _is_torch_greater_or_equal_than_2_5
        and attention_mask is None
        and not isinstance(key, torch.fx.Proxy)
    )


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    g=None,
    channel_ids_k=None,
    beta=None,
    log_eps: float = 1e-4,
    log_clip_min: float = -6.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(  # type: ignore
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)  # type: ignore
            value = repeat_kv(value, module.num_key_value_groups)  # type: ignore
        else:
            sdpa_kwargs = {"enable_gqa": True}

    # Role gating logic (if enabled)
    if (g is not None) and (channel_ids_k is not None) and (beta is not None):
        B, H, Q, d = query.shape
        K = key.shape[2]

        if g.dim() == 3:
            idx = channel_ids_k[:, None, :].expand(B, Q, K)
            g_qk = torch.gather(g, dim=-1, index=idx)
            log_bias = torch.log(g_qk + log_eps).clamp(min=log_clip_min, max=0.0)
            log_bias = log_bias[:, None, :, :]  # [B,1,Q,K]
        else:
            idx = channel_ids_k[:, None, :].expand(B, Q, K)
            idx = idx[:, None, :, :].expand(B, H, Q, K)
            g_qk = torch.gather(g, dim=-1, index=idx)
            log_bias = torch.log(g_qk + log_eps).clamp(
                min=log_clip_min, max=0.0
            )  # [B,H,Q,K]

        log_bias = log_bias.to(query.dtype)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
            attention_mask = attention_mask + beta.to(attention_mask.dtype) * log_bias
        else:
            attention_mask = beta.to(log_bias.dtype) * log_bias

    else:
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,  # type: ignore
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
