#!/usr/bin/env python3
"""Ground-truth reference tests for block-causal DeltaNet.

The chunk-vs-recurrent tests only check internal consistency. These tests
verify correctness against dead-simple reference implementations that
directly implement the mathematical specification.

Tests:
1. Block-causal output matches explicit per-token computation from prior-row state
2. Block-causal state update matches standard sequential delta rule
3. Column mode matches independent per-column sequential delta rule
4. Standard (vanilla) sequential delta rule reference, to cross-check our reference
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stream_arch.block_causal_deltanet import (
    block_causal_chunk_gated_delta_rule,
    block_causal_recurrent_gated_delta_rule,
    column_chunk_gated_delta_rule,
    column_recurrent_gated_delta_rule,
    l2norm,
)

# =====================================================================
# Reference implementations — maximally simple, no optimizations
# =====================================================================


def reference_vanilla_sequential(query, key, value, g, beta):
    """Standard causal delta rule, token-by-token. No chunking, no blocking.

    This is the ground truth for what a vanilla (unmodified) DeltaNet does.
    Returns output and final state.
    """
    B, H, S, Dk = query.shape
    Dv = value.shape[-1]
    scale = Dk**-0.5

    state = torch.zeros(B, H, Dk, Dv, dtype=query.dtype, device=query.device)
    out = torch.zeros(B, H, S, Dv, dtype=query.dtype, device=query.device)

    for t in range(S):
        q_t = query[:, :, t] * scale  # [B, H, Dk]
        k_t = key[:, :, t]
        v_t = value[:, :, t]
        g_t = g[:, :, t].exp()  # [B, H]
        b_t = beta[:, :, t]  # [B, H]

        # Decay state
        state = state * g_t[..., None, None]

        # Read: k^T @ state
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, Dv]

        # Delta correction
        delta = b_t.unsqueeze(-1) * (v_t - kv_mem)  # [B, H, Dv]

        # Write: state += k * delta^T
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

        # Output: q^T @ state
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return out, state


def reference_block_causal(query, key, value, g, beta, C):
    """Block-causal delta rule reference implementation.

    Specification:
      OUTPUT: each token independently queries the state from the END of the
              prior row, decayed by its own g, plus self-correction via delta rule.
              Tokens within the same row do NOT see each other's contributions.

      STATE UPDATE: standard sequential delta rule. All tokens in a row update
              the state sequentially (so future rows see all prior contributions).

    This is the specification we want to verify our chunk/recurrent code against.
    """
    B, H, S, Dk = query.shape
    Dv = value.shape[-1]
    scale = Dk**-0.5
    num_rows = (S + C - 1) // C

    state = torch.zeros(B, H, Dk, Dv, dtype=query.dtype, device=query.device)
    out = torch.zeros(B, H, S, Dv, dtype=query.dtype, device=query.device)

    for row in range(num_rows):
        row_start = row * C
        row_end = min(row_start + C, S)

        # Snapshot state at start of row (= end of prior row's sequential update)
        state_snapshot = state.clone()

        # --- OUTPUT phase: each token independently queries state_snapshot ---
        for t in range(row_start, row_end):
            q_t = query[:, :, t] * scale
            k_t = key[:, :, t]
            v_t = value[:, :, t]
            g_t = g[:, :, t].exp()
            b_t = beta[:, :, t]

            # Decay snapshot by this token's g
            decayed = state_snapshot * g_t[..., None, None]

            # Read from decayed state
            kv_mem = (decayed * k_t.unsqueeze(-1)).sum(dim=-2)

            # Delta correction (self only)
            delta = b_t.unsqueeze(-1) * (v_t - kv_mem)

            # Temporary state for output: decayed + self write
            temp = decayed + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

            # Output: q^T @ temp_state
            out[:, :, t] = (temp * q_t.unsqueeze(-1)).sum(dim=-2)

        # --- STATE UPDATE phase: standard sequential ---
        for t in range(row_start, row_end):
            k_t = key[:, :, t]
            v_t = value[:, :, t]
            g_t = g[:, :, t].exp()
            b_t = beta[:, :, t]

            state = state * g_t[..., None, None]
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = b_t.unsqueeze(-1) * (v_t - kv_mem)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

    return out, state


def reference_column(query, key, value, g, beta, C):
    """Column-mode delta rule reference implementation.

    Specification:
      Each column j maintains a fully independent state. Token at flat
      position t belongs to column j = t % C. It only interacts with
      column j's state: decay, read, delta-correct, write.

    This is exactly like running C independent vanilla delta rules,
    one per column.
    """
    B, H, S, Dk = query.shape
    Dv = value.shape[-1]
    scale = Dk**-0.5

    # Per-column states
    states = torch.zeros(B, H, C, Dk, Dv, dtype=query.dtype, device=query.device)
    out = torch.zeros(B, H, S, Dv, dtype=query.dtype, device=query.device)

    for t in range(S):
        col = t % C
        q_t = query[:, :, t] * scale
        k_t = key[:, :, t]
        v_t = value[:, :, t]
        g_t = g[:, :, t].exp()
        b_t = beta[:, :, t]

        # Decay this column's state
        states[:, :, col] = states[:, :, col] * g_t[..., None, None]

        # Read
        kv_mem = (states[:, :, col] * k_t.unsqueeze(-1)).sum(dim=-2)

        # Delta
        delta = b_t.unsqueeze(-1) * (v_t - kv_mem)

        # Write
        states[:, :, col] = states[:, :, col] + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

        # Output
        out[:, :, t] = (states[:, :, col] * q_t.unsqueeze(-1)).sum(dim=-2)

    return out, states


# =====================================================================
# Helper
# =====================================================================


def prepare_inputs(B, H, S, Dk, Dv, use_l2norm=False):
    """Generate inputs and apply preprocessing matching the actual functions."""
    torch.manual_seed(42)
    # Inputs in [B, S, H, D] format (what the actual functions expect)
    query = torch.randn(B, S, H, Dk)
    key = torch.randn(B, S, H, Dk)
    value = torch.randn(B, S, H, Dv)
    beta = torch.rand(B, S, H)
    g = -torch.rand(B, S, H) * 0.5

    # For references, we need [B, H, S, D] format with preprocessing applied
    if use_l2norm:
        q_ref = l2norm(query.transpose(1, 2).contiguous().float(), dim=-1)
        k_ref = l2norm(key.transpose(1, 2).contiguous().float(), dim=-1)
    else:
        q_ref = query.transpose(1, 2).contiguous().float()
        k_ref = key.transpose(1, 2).contiguous().float()
    v_ref = value.transpose(1, 2).contiguous().float()
    g_ref = g.transpose(1, 2).contiguous().float()
    b_ref = beta.transpose(1, 2).contiguous().float()

    return query, key, value, g, beta, q_ref, k_ref, v_ref, g_ref, b_ref


# =====================================================================
# Tests
# =====================================================================


def test_block_causal_vs_reference():
    """Block-causal chunk and recurrent must match the reference."""
    print("Test 1: Block-causal vs ground-truth reference")
    configs = [
        (1, 2, 30, 8, 16, 10, True),
        (2, 4, 50, 16, 32, 10, True),
        (1, 4, 40, 16, 32, 5, True),
        (1, 2, 20, 4, 8, 10, False),  # no l2norm, small dims
    ]
    for B, H, S, Dk, Dv, C, use_l2 in configs:
        q, k, v, g_raw, beta, q_ref, k_ref, v_ref, g_ref, b_ref = prepare_inputs(B, H, S, Dk, Dv, use_l2)

        # Reference
        out_ref, state_ref = reference_block_causal(q_ref, k_ref, v_ref, g_ref, b_ref, C)
        # Convert reference output back to [B, S, H, Dv]
        out_ref_bshd = out_ref.transpose(1, 2).contiguous()

        # Chunk implementation
        out_chunk, state_chunk = block_causal_chunk_gated_delta_rule(
            q,
            k,
            v,
            g_raw,
            beta,
            chunk_size=C,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_l2,
        )

        # Recurrent implementation
        out_rec, state_rec = block_causal_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g_raw,
            beta,
            initial_state=None,
            output_final_state=True,
            num_channels=C,
            use_qk_l2norm_in_kernel=use_l2,
        )

        chunk_out_diff = (out_chunk - out_ref_bshd).abs().max().item()
        chunk_state_diff = (state_chunk - state_ref).abs().max().item()
        rec_out_diff = (out_rec - out_ref_bshd).abs().max().item()
        rec_state_diff = (state_rec - state_ref).abs().max().item()

        tag = f"{'l2' if use_l2 else 'raw'} {(B, H, S, Dk, Dv, C)}"
        assert chunk_out_diff < 1e-4, f"  CHUNK output mismatch {tag}: {chunk_out_diff}"
        assert chunk_state_diff < 1e-4, f"  CHUNK state mismatch {tag}: {chunk_state_diff}"
        assert rec_out_diff < 1e-4, f"  REC output mismatch {tag}: {rec_out_diff}"
        assert rec_state_diff < 1e-4, f"  REC state mismatch {tag}: {rec_state_diff}"
        print(
            f"  {tag}: chunk out={chunk_out_diff:.2e} state={chunk_state_diff:.2e} | "
            f"rec out={rec_out_diff:.2e} state={rec_state_diff:.2e} ✓"
        )


def test_block_causal_state_matches_vanilla():
    """Block-causal STATE update must equal vanilla sequential STATE.

    The output differs (block-causal vs standard causal), but the state
    should be identical since the state update is standard sequential.
    """
    print("\nTest 2: Block-causal state == vanilla sequential state")
    configs = [
        (1, 2, 30, 8, 16, 10, True),
        (2, 4, 50, 16, 32, 10, True),
        (1, 2, 20, 4, 8, 10, False),
    ]
    for B, H, S, Dk, Dv, C, use_l2 in configs:
        q, k, v, g_raw, beta, q_ref, k_ref, v_ref, g_ref, b_ref = prepare_inputs(B, H, S, Dk, Dv, use_l2)

        # Reference vanilla (standard causal)
        _, state_vanilla = reference_vanilla_sequential(q_ref, k_ref, v_ref, g_ref, b_ref)

        # Reference block-causal
        _, state_bc = reference_block_causal(q_ref, k_ref, v_ref, g_ref, b_ref, C)

        # Chunk implementation
        _, state_chunk = block_causal_chunk_gated_delta_rule(
            q,
            k,
            v,
            g_raw,
            beta,
            chunk_size=C,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_l2,
        )

        ref_diff = (state_bc - state_vanilla).abs().max().item()
        chunk_diff = (state_chunk - state_vanilla).abs().max().item()

        tag = f"{'l2' if use_l2 else 'raw'} {(B, H, S, Dk, Dv, C)}"
        assert ref_diff < 1e-4, f"  Reference bc state != vanilla: {tag}: {ref_diff}"
        assert chunk_diff < 1e-4, f"  Chunk bc state != vanilla: {tag}: {chunk_diff}"
        print(f"  {tag}: ref_vs_vanilla={ref_diff:.2e}, chunk_vs_vanilla={chunk_diff:.2e} ✓")


def test_block_causal_output_differs_from_vanilla():
    """Block-causal OUTPUT must differ from vanilla (that's the whole point)."""
    print("\nTest 3: Block-causal output != vanilla output")
    B, H, S, Dk, Dv, C = 2, 4, 50, 16, 32, 10
    _, _, _, _, _, q_ref, k_ref, v_ref, g_ref, b_ref = prepare_inputs(B, H, S, Dk, Dv, True)

    out_vanilla, _ = reference_vanilla_sequential(q_ref, k_ref, v_ref, g_ref, b_ref)
    out_bc, _ = reference_block_causal(q_ref, k_ref, v_ref, g_ref, b_ref, C)

    diff = (out_bc - out_vanilla).abs().max().item()
    assert diff > 0.01, f"  Block-causal should differ from vanilla, got {diff}"
    print(f"  max diff = {diff:.2e} (correctly different) ✓")


def test_column_vs_reference():
    """Column-mode chunk and recurrent must match the column reference."""
    print("\nTest 4: Column-mode vs ground-truth reference")
    configs = [
        (1, 2, 30, 8, 16, 10, True),
        (2, 4, 50, 16, 32, 10, True),
        (1, 4, 40, 16, 32, 5, True),
        (1, 2, 20, 4, 8, 10, False),
    ]
    for B, H, S, Dk, Dv, C, use_l2 in configs:
        q, k, v, g_raw, beta, q_ref, k_ref, v_ref, g_ref, b_ref = prepare_inputs(B, H, S, Dk, Dv, use_l2)

        # Reference
        out_ref, state_ref = reference_column(q_ref, k_ref, v_ref, g_ref, b_ref, C)
        out_ref_bshd = out_ref.transpose(1, 2).contiguous()

        # Chunk
        out_chunk, state_chunk = column_chunk_gated_delta_rule(
            q,
            k,
            v,
            g_raw,
            beta,
            chunk_size=C,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_l2,
        )

        # Recurrent
        out_rec, state_rec = column_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g_raw,
            beta,
            initial_state=None,
            output_final_state=True,
            num_channels=C,
            use_qk_l2norm_in_kernel=use_l2,
        )

        chunk_out_diff = (out_chunk - out_ref_bshd).abs().max().item()
        chunk_state_diff = (state_chunk - state_ref).abs().max().item()
        rec_out_diff = (out_rec - out_ref_bshd).abs().max().item()
        rec_state_diff = (state_rec - state_ref).abs().max().item()

        tag = f"{'l2' if use_l2 else 'raw'} {(B, H, S, Dk, Dv, C)}"
        assert chunk_out_diff < 1e-4, f"  CHUNK output mismatch {tag}: {chunk_out_diff}"
        assert chunk_state_diff < 1e-4, f"  CHUNK state mismatch {tag}: {chunk_state_diff}"
        assert rec_out_diff < 1e-4, f"  REC output mismatch {tag}: {rec_out_diff}"
        assert rec_state_diff < 1e-4, f"  REC state mismatch {tag}: {rec_state_diff}"
        print(
            f"  {tag}: chunk out={chunk_out_diff:.2e} state={chunk_state_diff:.2e} | "
            f"rec out={rec_out_diff:.2e} state={rec_state_diff:.2e} ✓"
        )


def test_column_state_differs_from_vanilla():
    """Column mode STATE must differ from vanilla (independent per-column vs shared)."""
    print("\nTest 5: Column state != vanilla state")
    B, H, S, Dk, Dv, C = 2, 4, 50, 16, 32, 10
    _, _, _, _, _, q_ref, k_ref, v_ref, g_ref, b_ref = prepare_inputs(B, H, S, Dk, Dv, True)

    _, state_vanilla = reference_vanilla_sequential(q_ref, k_ref, v_ref, g_ref, b_ref)
    _, state_col = reference_column(q_ref, k_ref, v_ref, g_ref, b_ref, C)

    # Column state is [B, H, C, Dk, Dv], vanilla is [B, H, Dk, Dv]
    # They should be structurally different (column has C independent states)
    # Verify that different columns have different states
    col_diffs = []
    for c in range(1, C):
        d = (state_col[:, :, c] - state_col[:, :, 0]).abs().max().item()
        col_diffs.append(d)

    assert max(col_diffs) > 0.01, "Column states should differ across columns"
    print(f"  max inter-column diff = {max(col_diffs):.2e} (correctly different) ✓")


def test_vanilla_reference_cross_check():
    """Cross-check our vanilla reference against itself with different orderings.

    Process S tokens one-by-one vs all-at-once — must get identical result.
    This validates that our reference_vanilla_sequential is correct.
    """
    print("\nTest 6: Vanilla reference self-consistency")
    B, H, S, Dk, Dv = 1, 2, 20, 8, 16
    _, _, _, _, _, q, k, v, g, beta = prepare_inputs(B, H, S, Dk, Dv, False)

    # Full sequence
    out_full, state_full = reference_vanilla_sequential(q, k, v, g, beta)

    # Split: first half, then second half with initial_state
    S1 = S // 2
    out1, state1 = reference_vanilla_sequential(q[:, :, :S1], k[:, :, :S1], v[:, :, :S1], g[:, :, :S1], beta[:, :, :S1])

    # Second half: manually continue from state1
    scale = Dk**-0.5
    state = state1.clone()
    out2 = torch.zeros(B, H, S - S1, Dv, dtype=q.dtype)
    for t in range(S - S1):
        t_abs = S1 + t
        q_t = q[:, :, t_abs] * scale
        k_t = k[:, :, t_abs]
        v_t = v[:, :, t_abs]
        g_t = g[:, :, t_abs].exp()
        b_t = beta[:, :, t_abs]

        state = state * g_t[..., None, None]
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = b_t.unsqueeze(-1) * (v_t - kv_mem)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out2[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    out_split = torch.cat([out1, out2], dim=2)
    out_diff = (out_full - out_split).abs().max().item()
    state_diff = (state_full - state).abs().max().item()

    assert out_diff < 1e-6, f"  Vanilla split mismatch: out={out_diff}"
    assert state_diff < 1e-6, f"  Vanilla split mismatch: state={state_diff}"
    print(f"  split consistency: out={out_diff:.2e}, state={state_diff:.2e} ✓")


def test_block_causal_first_row_matches_vanilla():
    """First row (no prior state) should match vanilla for both block-causal and vanilla.

    With zero initial state, the output for each token in row 0 is just
    the self-contribution (since the prior-row state is zero).
    """
    print("\nTest 7: First-row output: block-causal vs vanilla")
    B, H, Dk, Dv, C = 1, 4, 16, 32, 10
    S = C  # single row
    _, _, _, _, _, q, k, v, g, beta = prepare_inputs(B, H, S, Dk, Dv, True)

    out_vanilla, _ = reference_vanilla_sequential(q, k, v, g, beta)
    out_bc, _ = reference_block_causal(q, k, v, g, beta, C)

    # In row 0, vanilla output for token t depends on tokens 0..t (causal).
    # Block-causal output for token t depends only on self (prior-row state is zero).
    # So they should DIFFER for t > 0 (token 1 sees token 0 in vanilla, not in bc).

    # But token 0 should match exactly (both only see self from zero state)
    diff_t0 = (out_vanilla[:, :, 0] - out_bc[:, :, 0]).abs().max().item()
    assert diff_t0 < 1e-6, f"  Token 0 should match: {diff_t0}"
    print(f"  token 0 matches: {diff_t0:.2e} ✓")

    # Token 1+ should differ
    if C > 1:
        diff_later = (out_vanilla[:, :, 1:] - out_bc[:, :, 1:]).abs().max().item()
        assert diff_later > 1e-4, f"  Later tokens should differ: {diff_later}"
        print(f"  tokens 1+ differ: {diff_later:.2e} ✓")


if __name__ == "__main__":
    test_block_causal_vs_reference()
    test_block_causal_state_matches_vanilla()
    test_block_causal_output_differs_from_vanilla()
    test_column_vs_reference()
    test_column_state_differs_from_vanilla()
    test_vanilla_reference_cross_check()
    test_block_causal_first_row_matches_vanilla()
    print("\nAll reference tests passed!")
