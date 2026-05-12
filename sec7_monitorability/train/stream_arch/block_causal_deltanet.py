"""Block-causal variants of GatedDeltaNet delta rule functions.

Standard GatedDeltaNet is causal: each token sees all prior tokens.
Block-causal: tokens see all prior ROWS + self, but NOT same-row peers.

For each row, the state is "forked": all tokens in the row independently
query the state from the end of the prior row (with their own per-token decay)
and add their own self-correction. Then the state is updated sequentially
with all tokens in the row for future rows.

Also provides BlockCausalConv1d which wraps the DeltaNet's depthwise causal
conv1d to prevent same-row channel leakage through the convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class BlockCausalConv1d(nn.Module):
    """Wraps a depthwise causal Conv1d to prevent same-row channel leakage.

    Modes:
      "column" (recommended): Each channel looks back at the SAME channel
        from K-1 prior rows. Captures within-channel temporal patterns.
          out[r, j] = w[K-1]*x[r,j] + w[K-2]*x[r-1,j] + w[K-3]*x[r-2,j] + ... + bias

      "row_boundary": Each channel sees the last K-1 tokens of the prior
        row (shared across all channels) + its own token.
          out[r, j] = w[K-1]*x[r,j] + w[K-2]*x[r-1,C-1] + w[K-3]*x[r-1,C-2] + ... + bias

    Both modes decompose the depthwise conv analytically for efficiency.
    """

    def __init__(self, conv1d: nn.Conv1d, num_channels: int, mode: str = "column"):
        super().__init__()
        # Store weight/bias directly so state dict keys match pretrained checkpoint
        # (storing conv1d as submodule would add a prefix to keys)
        self.weight = conv1d.weight  # nn.Parameter [D, 1, K]
        self.bias = conv1d.bias  # nn.Parameter [D] or None
        self.num_channels = num_channels
        self.kernel_size = conv1d.kernel_size
        self._conv_groups = conv1d.groups
        self._conv_padding = conv1d.padding
        assert mode in ("column", "row_boundary"), f"Unknown conv mode: {mode}"
        self.mode = mode

    def forward(self, x):
        # x: [B, D, S]
        B, D, S = x.shape
        C = self.num_channels
        K = self.kernel_size[0]

        if S % C != 0:
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                padding=self._conv_padding,
                groups=self._conv_groups,
            )

        num_rows = S // C
        # Depthwise conv weight: [D, 1, K] -> [D, K]
        # PyTorch cross-correlation: w[K-1] = self, w[K-2] = t-1, ..., w[0] = t-K+1
        w = self.weight.squeeze(1)  # [D, K]

        # Self contribution (same for both modes)
        out = x * w[:, K - 1].unsqueeze(0).unsqueeze(-1)  # [B, D, S]

        if K > 1 and num_rows > 1:
            x_rows = x.reshape(B, D, num_rows, C)

            if self.mode == "column":
                # Same channel from K-1 prior rows:
                #   w[K-1-k] * x[r-k, j]  for k = 1..K-1
                for k in range(1, min(K, num_rows)):
                    # Shift rows: row r gets row r-k, rows 0..k-1 get zero
                    shifted = torch.zeros_like(x_rows)
                    shifted[:, :, k:, :] = x_rows[:, :, :-k, :]
                    out = out + shifted.reshape(B, D, S) * w[:, K - 1 - k].unsqueeze(
                        0
                    ).unsqueeze(-1)

            else:  # row_boundary
                # Last K-1 tokens of prior row, shared across all channels
                prior_tokens = x_rows[:, :, :, -(K - 1) :].flip(
                    -1
                )  # [B, D, num_rows, K-1]
                prior_w = w[:, :-1].flip(-1)  # [D, K-1]
                prior_contrib = (prior_tokens * prior_w.unsqueeze(0).unsqueeze(2)).sum(
                    -1
                )  # [B, D, num_rows]
                prior_contrib = F.pad(
                    prior_contrib[:, :, :-1], (1, 0)
                )  # shift: row i uses row i-1
                out = out + prior_contrib.unsqueeze(-1).expand(
                    B, D, num_rows, C
                ).reshape(B, D, S)

        # Bias
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1)

        # Pad right by K-1 to match original conv1d output shape
        # (caller does [:, :, :seq_len] to trim)
        out = F.pad(out, (0, K - 1))
        return out


def _fla_column_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=10,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Fast per-column delta rule using fla's Triton kernel.

    Treats each channel column as an independent causal sequence and
    batches all C columns into a single fla call. ~50x faster than the
    Python-loop fallback.
    """
    # Input: [B, S, H, K/V], g/beta: [B, S, H]
    C = chunk_size
    batch_size = query.shape[0]
    sequence_length = query.shape[1]
    pad_size = (C - sequence_length % C) % C
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, 0, 0, pad_size))
        g = F.pad(g, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, 0, 0, pad_size))
    S_padded = sequence_length + pad_size
    num_rows = S_padded // C

    # Reshape [B, num_rows*C, H, D] -> [B, num_rows, C, H, D] -> [B*C, num_rows, H, D]
    def _col_reshape(x):
        if x.dim() == 4:  # [B, S, H, D]
            return x.reshape(batch_size, num_rows, C, *x.shape[2:]).permute(0, 2, 1, 3, 4).reshape(batch_size * C, num_rows, *x.shape[2:])
        else:  # [B, S, H]
            return x.reshape(batch_size, num_rows, C, x.shape[2]).permute(0, 2, 1, 3).reshape(batch_size * C, num_rows, x.shape[2])

    q_c = _col_reshape(query).contiguous()
    k_c = _col_reshape(key).contiguous()
    v_c = _col_reshape(value).contiguous()
    g_c = _col_reshape(g).contiguous()
    beta_c = _col_reshape(beta).contiguous()

    # Reshape initial state: [B, H, C, Dk, Dv] -> [B*C, H, Dk, Dv]
    h0 = None
    if initial_state is not None:
        h0 = initial_state.permute(0, 2, 1, 3, 4).reshape(batch_size * C, *initial_state.shape[1:2], *initial_state.shape[3:])

    o_fla, s_fla = _fla_chunk_gated_delta_rule(
        q_c, k_c, v_c, g_c, beta_c,
        initial_state=h0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Reshape output: [B*C, num_rows, H, V] -> [B, S, H, V]
    V_dim = o_fla.shape[-1]
    H = o_fla.shape[2]
    out = o_fla.reshape(batch_size, C, num_rows, H, V_dim).permute(0, 2, 1, 3, 4).reshape(batch_size, S_padded, H, V_dim)
    out = out[:, :sequence_length]

    # Reshape state: [B*C, H, Dk, Dv] -> [B, H, C, Dk, Dv]
    states = None
    if output_final_state and s_fla is not None:
        states = s_fla.reshape(batch_size, C, *s_fla.shape[1:]).permute(0, 2, 1, 3, 4)

    return out, states


def column_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=10,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Per-column delta rule: each channel maintains an independent state.

    chunk_size MUST equal num_channels C. Each column j's tokens
    (positions j, C+j, 2C+j, ...) form an independent causal sequence.
    This is more aligned with pretrained weights since each column looks
    like regular text the model was trained on.

    State shape: [B, H, C, Dk, Dv] — one state per column.
    """
    if _HAS_FLA:
        return _fla_column_chunk_gated_delta_rule(
            query, key, value, g, beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    C = chunk_size
    pad_size = (C - sequence_length % C) % C
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    num_rows = total_sequence_length // C
    # Reshape: [B, H, S, D] → [B, H, num_rows, C, D]
    query = query.reshape(batch_size, num_heads, num_rows, C, k_head_dim)
    key = key.reshape(batch_size, num_heads, num_rows, C, k_head_dim)
    value = value.reshape(batch_size, num_heads, num_rows, C, v_head_dim)
    beta = beta.reshape(batch_size, num_heads, num_rows, C)
    g = g.reshape(batch_size, num_heads, num_rows, C)

    # Per-column states: [B, H, C, Dk, Dv]
    states = (
        torch.zeros(batch_size, num_heads, C, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for r in range(num_rows):
        q_r = query[:, :, r]  # [B, H, C, Dk]
        k_r = key[:, :, r]  # [B, H, C, Dk]
        v_r = value[:, :, r]  # [B, H, C, Dv]
        g_r = g[:, :, r].exp()  # [B, H, C]
        beta_r = beta[:, :, r]  # [B, H, C]

        # Decay each column's state independently
        decayed = states * g_r[..., None, None]  # [B, H, C, Dk, Dv]

        # Query: q^T @ decayed_state (per column)
        attn_inter = (q_r.unsqueeze(-1) * decayed).sum(dim=-2)  # [B, H, C, Dv]

        # Self-correction via delta rule
        kv_mem = (k_r.unsqueeze(-1) * decayed).sum(dim=-2)  # [B, H, C, Dv]
        delta = beta_r.unsqueeze(-1) * (v_r - kv_mem)  # [B, H, C, Dv]
        qk_diag = (q_r * k_r).sum(dim=-1, keepdim=True)  # [B, H, C, 1]
        core_attn_out[:, :, r] = attn_inter + qk_diag * delta

        # Update each column's state
        states = decayed + k_r.unsqueeze(-1) * delta.unsqueeze(-2)

    if not output_final_state:
        states = None
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, states


def column_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    num_channels=10,
):
    """Per-column recurrent delta rule for inference.

    Each token updates only its own column's state.
    State shape: [B, H, C, Dk, Dv].
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    C = num_channels
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(
        value
    )
    states = (
        torch.zeros(batch_size, num_heads, C, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        col = i % C
        q_t = query[:, :, i]  # [B, H, Dk]
        k_t = key[:, :, i]  # [B, H, Dk]
        v_t = value[:, :, i]  # [B, H, Dv]
        g_t = g[:, :, i].exp()  # [B, H]
        beta_t = beta[:, :, i]  # [B, H]

        # Decay this column's state
        decayed = states[:, :, col] * g_t[..., None, None]  # [B, H, Dk, Dv]

        # Output: q^T @ (decayed + k * delta^T)
        kv_mem = (decayed * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, Dv]
        delta = beta_t.unsqueeze(-1) * (v_t - kv_mem)  # [B, H, Dv]
        new_state = decayed + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)

        # Update only this column's state
        states[:, :, col] = new_state

    if not output_final_state:
        states = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, states


def block_causal_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=10,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Chunk-mode delta rule with block-causal masking.

    chunk_size MUST equal num_channels so that each chunk = one row.

    Output computation (block-causal):
      Each token independently queries the prior-row state with its own
      per-token decay, plus a self-correction via the delta rule.
      No cross-token dependencies within a row.

    State update (standard sequential):
      Tokens update the state sequentially within each chunk, using the
      standard chunk delta rule's correction matrix. Future rows see all
      prior tokens' contributions.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    C = chunk_size
    pad_size = (C - sequence_length % C) % C
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks: each chunk = one row of channels
    query, key, value, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, -1, C, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    beta = beta.reshape(batch_size, num_heads, -1, C)
    g_raw = g.reshape(batch_size, num_heads, -1, C)  # raw per-token log-decay

    # =====================================================================
    # State update precomputation (standard sequential correction matrix)
    # =====================================================================
    g_cumsum = g_raw.cumsum(dim=-1)
    decay_mask = (
        (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).tril().exp().float()
    ).tril()

    # Standard lower-triangular correction matrix (same as torch_chunk_gated_delta_rule)
    mask_upper = torch.triu(
        torch.ones(C, C, dtype=torch.bool, device=query.device), diagonal=0
    )
    attn_corr = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(
        mask_upper, 0
    )
    for j in range(1, C):
        row = attn_corr[..., j, :j].clone()
        sub = attn_corr[..., :j, :j].clone()
        attn_corr[..., j, :j] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn_corr = attn_corr + torch.eye(C, dtype=attn_corr.dtype, device=attn_corr.device)

    # Precompute corrected values and key-decay for batched state update
    v_corrected = attn_corr @ v_beta  # [B, H, num_chunks, C, v_dim]
    k_cumdecay = attn_corr @ (
        k_beta * g_cumsum.exp().unsqueeze(-1)
    )  # [B, H, num_chunks, C, k_dim]

    # =====================================================================
    # Per-chunk loop
    # =====================================================================
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    num_chunks = total_sequence_length // C

    for i in range(num_chunks):
        q_i = query[:, :, i]  # [B, H, C, k_dim]
        k_i = key[:, :, i]  # [B, H, C, k_dim]
        v_i = value[:, :, i]  # [B, H, C, v_dim]

        # --- Block-causal OUTPUT (batched, independent per token) ---
        # Each token: decay prior-row state by own g, query it, + self-correct
        decay_i = g_raw[:, :, i].exp()  # [B, H, C] per-token decay

        # Query prior-row state with per-token independent decay
        attn_inter = (q_i * decay_i.unsqueeze(-1)) @ last_recurrent_state

        # Self-correction: retrieve from decayed state, compute delta
        kv_mem = (k_i * decay_i.unsqueeze(-1)) @ last_recurrent_state
        delta = beta[:, :, i].unsqueeze(-1) * (v_i - kv_mem)
        qk_diag = (q_i * k_i).sum(dim=-1, keepdim=True)  # [B, H, C, 1]
        core_attn_out[:, :, i] = attn_inter + qk_diag * delta

        # --- State UPDATE (standard sequential, batched) ---
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_corrected[:, :, i] - v_prime
        last_recurrent_state = (
            last_recurrent_state * g_cumsum[:, :, i, -1, None, None].exp()
            + (
                k_i * (g_cumsum[:, :, i, -1, None] - g_cumsum[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def stream_forward_gated_deltanet(
    la,
    hidden_states,
    cache_params,
    cache_position,
    attention_mask,
    conv_state_len,
    has_block_causal_conv,
):
    """GatedDeltaNet forward with stream-aware state management.

    Replaces the upstream Qwen3_5GatedDeltaNet.forward() to fix two bugs
    in multi-token decode (seq_len > 1 with cache):

    1. Recurrent state: the upstream always passes initial_state=None in chunk
       mode (gated by seq_len==1).  We pass the cached state instead.
    2. Conv state: for column/row_boundary BlockCausalConv1d, the upstream
       saves only K-1 positions.  We save (K-1)*C positions (K-1 full rows)
       and prepend them on the next step so the conv sees prior rows.

    Args:
        la: The Qwen3_5GatedDeltaNet (or Moe variant) module.
        hidden_states: [B, S, D] input tensor.
        cache_params: Qwen3_5DynamicCache or None.
        cache_position: LongTensor or None.
        attention_mask: Passed to apply_mask_to_padding_states (usually None
            for linear_attention layers during inference).
        conv_state_len: Number of pre-conv positions to cache.
            (K-1)*C for column/row_boundary modes, K-1 for standard.
        has_block_causal_conv: True when la.conv1d is a BlockCausalConv1d.
    """
    # apply_mask_to_padding_states — only acts on 2D boolean masks (padding)
    if (
        attention_mask is not None
        and attention_mask.ndim == 2
        and attention_mask.shape[0] > 1
        and attention_mask.shape[1] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    batch_size, seq_len, _ = hidden_states.shape

    has_prior_state = (
        cache_params is not None
        and cache_params.has_previous_state
        and cache_position is not None
    )

    # Load cached states
    recurrent_state = None
    conv_state = None
    if has_prior_state:
        conv_state = cache_params.conv_states[la.layer_idx]
        recurrent_state = cache_params.recurrent_states[la.layer_idx]

    # ── Projections ──
    mixed_qkv = la.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, D_conv, S]
    z = la.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, la.head_v_dim)
    b = la.in_proj_b(hidden_states)
    a = la.in_proj_a(hidden_states)

    # ── Conv1d with proper state management ──
    if has_prior_state and has_block_causal_conv and conv_state is not None:
        # BlockCausalConv1d decode (any seq_len): prepend prior rows
        cs = conv_state
        if cs.shape[-1] < conv_state_len:
            cs = F.pad(cs, (conv_state_len - cs.shape[-1], 0))
        elif cs.shape[-1] > conv_state_len:
            cs = cs[:, :, -conv_state_len:]
        mixed_qkv_ext = torch.cat([cs, mixed_qkv], dim=-1)
        cache_params.conv_states[la.layer_idx] = (
            mixed_qkv_ext[:, :, -conv_state_len:].clone()
        )
        conv_out = la.conv1d(mixed_qkv_ext)  # BlockCausalConv1d
        mixed_qkv = F.silu(
            conv_out[:, :, conv_state_len : conv_state_len + seq_len]
        )

    elif has_prior_state and seq_len == 1:
        # Single-token decode (standard HF generate, not stream inference)
        mixed_qkv = la.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            la.conv1d.weight.squeeze(1),
            la.conv1d.bias,
            la.activation,
        )

    else:
        # Training, first inference pass, or standard conv multi-token decode
        if cache_params is not None and conv_state_len > 0:
            if mixed_qkv.shape[-1] >= conv_state_len:
                cache_params.conv_states[la.layer_idx] = (
                    mixed_qkv[:, :, -conv_state_len:].clone()
                )
            else:
                cache_params.conv_states[la.layer_idx] = F.pad(
                    mixed_qkv, (conv_state_len - mixed_qkv.shape[-1], 0)
                )
        if la.causal_conv1d_fn is not None:
            mixed_qkv = la.causal_conv1d_fn(
                x=mixed_qkv,
                weight=la.conv1d.weight.squeeze(1),
                bias=la.conv1d.bias,
                activation=la.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(la.conv1d(mixed_qkv)[:, :, :seq_len])

    # ── QKV split ──
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [la.key_dim, la.key_dim, la.value_dim], dim=-1
    )
    query = query.reshape(batch_size, seq_len, -1, la.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, la.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, la.head_v_dim)

    beta = b.sigmoid()
    g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
    if la.num_v_heads // la.num_k_heads > 1:
        query = query.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
        key = key.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)

    # ── Delta rule with proper state forwarding ──
    if has_prior_state and seq_len == 1:
        # Single-token decode: recurrent mode (upstream path, correct)
        core_attn_out, last_state = la.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        # Training, first pass, or multi-token decode
        # FIX: pass recurrent_state (None for first pass, cached for decode)
        core_attn_out, last_state = la.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.recurrent_states[la.layer_idx] = last_state

    # ── Output norm + projection ──
    core_attn_out = core_attn_out.reshape(-1, la.head_v_dim)
    z = z.reshape(-1, la.head_v_dim)
    core_attn_out = la.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return la.out_proj(core_attn_out)


def block_causal_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    num_channels=10,
):
    """Recurrent-mode delta rule with block-causal masking.

    For each row: save state, compute each token's output from the saved
    state + self contribution only, then update state with the full row.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(
        value
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    C = num_channels
    num_rows = (sequence_length + C - 1) // C

    for row in range(num_rows):
        row_start = row * C
        row_end = min(row_start + C, sequence_length)

        # Save state at the start of this row (contains only prior-row information)
        row_initial_state = last_recurrent_state.clone()

        # Compute output for each token using row_initial_state + self
        for i in range(row_start, row_end):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            # Temporary state: row_initial_state decayed + self token's contribution
            temp_state = row_initial_state * g_t
            kv_mem = (temp_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            temp_state = temp_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (temp_state * q_t.unsqueeze(-1)).sum(dim=-2)

        # After computing all outputs, update the actual state with all tokens
        # in this row (sequentially, for the benefit of future rows)
        for i in range(row_start, row_end):
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
                -1
            ) * delta.unsqueeze(-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
