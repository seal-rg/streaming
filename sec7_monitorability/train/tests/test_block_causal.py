#!/usr/bin/env python3
"""Tests for block-causal DeltaNet and BlockCausalConv1d.

Tests:
1. Chunk vs recurrent delta rule match (with L2-normed qk, matching Qwen3.5)
2. Chunk vs recurrent match without L2-norm (small dims to avoid fp32 instability)
3. BlockCausalConv1d column mode: no same-row leakage, correct column lookback
4. BlockCausalConv1d row_boundary mode: no same-row leakage
5. Channel independence: identical channel inputs → identical outputs
6. Config switch: standard conv mode falls through to original conv1d
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stream_arch.block_causal_deltanet import (
    BlockCausalConv1d,
    block_causal_chunk_gated_delta_rule,
    block_causal_recurrent_gated_delta_rule,
    column_chunk_gated_delta_rule,
    column_recurrent_gated_delta_rule,
)


def make_inputs(B, H, S, Dk, Dv, device="cpu"):
    """Generate random inputs for delta rule tests."""
    torch.manual_seed(42)
    query = torch.randn(B, S, H, Dk, device=device)
    key = torch.randn(B, S, H, Dk, device=device)
    value = torch.randn(B, S, H, Dv, device=device)
    beta = torch.rand(B, S, H, device=device)
    g = -torch.rand(B, S, H, device=device) * 0.5  # negative log-decay
    return query, key, value, g, beta


def test_chunk_vs_recurrent_l2norm():
    """Chunk and recurrent must match with L2-normed qk (Qwen3.5 default).

    L2-normalization keeps the correction matrix well-conditioned (cond ~1.5),
    so chunk and recurrent should match in fp32 to high precision.
    """
    configs = [
        (2, 4, 100, 16, 32, 10),
        (1, 2, 30, 8, 16, 10),
        (2, 4, 50, 16, 32, 5),
        (1, 8, 80, 16, 32, 8),
    ]
    for B, H, S, Dk, Dv, C in configs:
        q, k, v, g, beta = make_inputs(B, H, S, Dk, Dv)

        out_chunk, state_chunk = block_causal_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=C,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out_rec, state_rec = block_causal_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=True,
            num_channels=C,
            use_qk_l2norm_in_kernel=True,
        )

        out_diff = (out_chunk - out_rec).abs().max().item()
        state_diff = (state_chunk - state_rec).abs().max().item()
        assert out_diff < 1e-4, f"Config {(B, H, S, Dk, Dv, C)}: output diff {out_diff}"
        assert state_diff < 1e-4, f"Config {(B, H, S, Dk, Dv, C)}: state diff {state_diff}"
        print(f"  l2norm {(B, H, S, Dk, Dv, C)}: out={out_diff:.2e}, state={state_diff:.2e} ✓")


def test_chunk_vs_recurrent_small():
    """Chunk and recurrent must match for small dims (no L2-norm).

    Without L2-norm, the correction matrix can be ill-conditioned for large Dk,
    so we test with small dims where fp32 is sufficient.
    """
    configs = [
        (1, 1, 20, 4, 4, 10),
        (1, 2, 30, 4, 8, 10),
        (2, 2, 20, 4, 4, 5),
    ]
    for B, H, S, Dk, Dv, C in configs:
        q, k, v, g, beta = make_inputs(B, H, S, Dk, Dv)

        out_chunk, state_chunk = block_causal_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=C,
            initial_state=None,
            output_final_state=True,
        )
        out_rec, state_rec = block_causal_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=True,
            num_channels=C,
        )

        out_diff = (out_chunk - out_rec).abs().max().item()
        state_diff = (state_chunk - state_rec).abs().max().item()
        assert out_diff < 1e-4, f"Config {(B, H, S, Dk, Dv, C)}: output diff {out_diff}"
        assert state_diff < 1e-4, f"Config {(B, H, S, Dk, Dv, C)}: state diff {state_diff}"
        print(f"  small {(B, H, S, Dk, Dv, C)}: out={out_diff:.2e}, state={state_diff:.2e} ✓")


def test_block_causal_vs_standard_differ():
    """Block-causal and standard causal must produce different outputs."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as standard_chunk

    B, H, S, Dk, Dv, C = 2, 4, 100, 16, 32, 10
    q, k, v, g, beta = make_inputs(B, H, S, Dk, Dv)

    out_bc, _ = block_causal_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=C,
    )
    out_std, _ = standard_chunk(q, k, v, g, beta, chunk_size=C)

    diff = (out_bc - out_std).abs().max().item()
    assert diff > 0.01, f"Block-causal should differ from standard, got diff={diff}"
    print(f"  block-causal vs standard diff: {diff:.2e} (correctly different) ✓")


def test_channel_independence():
    """Tokens in the same row must produce independent outputs.

    If we duplicate channel 0's inputs to all channels in a row,
    every channel should produce identical output (since they all
    independently query the same prior-row state with the same qkv).
    """
    B, H, Dk, Dv, C = 1, 4, 16, 32, 10
    num_rows = 5
    S = num_rows * C

    torch.manual_seed(42)
    q, k, v, g, beta = make_inputs(B, H, S, Dk, Dv)

    # Make all channels in each row identical (copy channel 0 to all)
    q_dup = q.reshape(B, num_rows, C, H, Dk)
    k_dup = k.reshape(B, num_rows, C, H, Dk)
    v_dup = v.reshape(B, num_rows, C, H, Dv)
    g_dup = g.reshape(B, num_rows, C, H)
    beta_dup = beta.reshape(B, num_rows, C, H)

    for r in range(num_rows):
        q_dup[:, r, :] = q_dup[:, r, 0:1]
        k_dup[:, r, :] = k_dup[:, r, 0:1]
        v_dup[:, r, :] = v_dup[:, r, 0:1]
        g_dup[:, r, :] = g_dup[:, r, 0:1]
        beta_dup[:, r, :] = beta_dup[:, r, 0:1]

    q_flat = q_dup.reshape(B, S, H, Dk)
    k_flat = k_dup.reshape(B, S, H, Dk)
    v_flat = v_dup.reshape(B, S, H, Dv)
    g_flat = g_dup.reshape(B, S, H)
    beta_flat = beta_dup.reshape(B, S, H)

    out, _ = block_causal_chunk_gated_delta_rule(
        q_flat,
        k_flat,
        v_flat,
        g_flat,
        beta_flat,
        chunk_size=C,
        use_qk_l2norm_in_kernel=True,
    )

    # Reshape output to [B, num_rows, C, H, Dv] and check all channels match
    out_rows = out.reshape(B, num_rows, C, H, Dv)
    max_diff = 0
    for r in range(num_rows):
        for c in range(1, C):
            diff = (out_rows[:, r, c] - out_rows[:, r, 0]).abs().max().item()
            max_diff = max(max_diff, diff)

    assert max_diff < 1e-5, f"Channel independence violated: max_diff={max_diff}"
    print(f"  channel independence (identical inputs): max_diff={max_diff:.2e} ✓")


def test_conv1d_column_mode():
    """Column mode: each channel looks back at same channel from prior rows."""
    D, C, K = 8, 4, 3
    num_rows = 5
    S = num_rows * C

    conv = nn.Conv1d(D, D, K, padding=K - 1, groups=D)
    nn.init.ones_(conv.weight)  # simple weights for inspection
    nn.init.zeros_(conv.bias)

    wrapper = BlockCausalConv1d(conv, C, mode="column")

    torch.manual_seed(42)
    x = torch.randn(1, D, S)
    out = wrapper(x)[:, :, :S]  # trim padding

    x_rows = x.reshape(1, D, num_rows, C)
    out_rows = out.reshape(1, D, num_rows, C)

    # For K=3, column mode: out[r,j] = w[2]*x[r,j] + w[1]*x[r-1,j] + w[0]*x[r-2,j]
    # With w all 1s: out[r,j] = x[r,j] + x[r-1,j] + x[r-2,j]
    for r in range(num_rows):
        for j in range(C):
            expected = x_rows[0, :, r, j]
            if r >= 1:
                expected = expected + x_rows[0, :, r - 1, j]
            if r >= 2:
                expected = expected + x_rows[0, :, r - 2, j]
            diff = (out_rows[0, :, r, j] - expected).abs().max().item()
            assert diff < 1e-5, f"Column mode wrong at row={r}, ch={j}: diff={diff}"

    print("  conv1d column mode correctness ✓")


def test_conv1d_no_cross_channel_leakage():
    """Column mode must not leak information across channels within a row."""
    D, C, K = 4, 4, 3
    num_rows = 4
    S = num_rows * C

    conv = nn.Conv1d(D, D, K, padding=K - 1, groups=D)
    torch.manual_seed(42)
    nn.init.normal_(conv.weight)
    nn.init.zeros_(conv.bias)

    wrapper = BlockCausalConv1d(conv, C, mode="column")

    # Create input where only channel 0 in each row has non-zero values
    x = torch.zeros(1, D, S)
    x_rows = x.reshape(1, D, num_rows, C)
    x_rows[:, :, :, 0] = torch.randn(1, D, num_rows)

    out = wrapper(x.reshape(1, D, S))[:, :, :S]
    out_rows = out.reshape(1, D, num_rows, C)

    # Channels 1,2,3 should be zero in output (no leakage from channel 0)
    for j in range(1, C):
        leakage = out_rows[:, :, :, j].abs().max().item()
        assert leakage < 1e-6, f"Cross-channel leakage to ch {j}: {leakage}"

    # Channel 0 should be non-zero
    ch0_signal = out_rows[:, :, :, 0].abs().max().item()
    assert ch0_signal > 0.01, f"Channel 0 should have signal, got {ch0_signal}"

    print("  conv1d column mode: no cross-channel leakage ✓")


def test_conv1d_row_boundary_mode():
    """Row boundary mode: each channel sees last K-1 tokens of prior row."""
    D, C, K = 4, 4, 3
    num_rows = 4
    S = num_rows * C

    conv = nn.Conv1d(D, D, K, padding=K - 1, groups=D)
    nn.init.ones_(conv.weight)
    nn.init.zeros_(conv.bias)

    wrapper = BlockCausalConv1d(conv, C, mode="row_boundary")

    torch.manual_seed(42)
    x = torch.randn(1, D, S)
    out = wrapper(x)[:, :, :S]

    x_rows = x.reshape(1, D, num_rows, C)
    out_rows = out.reshape(1, D, num_rows, C)

    # For K=3, row_boundary: out[r,j] = w[2]*x[r,j] + (w[1]*x[r-1,C-1] + w[0]*x[r-1,C-2])
    # With w all 1s: out[r,j] = x[r,j] + x[r-1,C-1] + x[r-1,C-2]
    for r in range(1, num_rows):
        prior_contrib = x_rows[0, :, r - 1, C - 1] + x_rows[0, :, r - 1, C - 2]
        for j in range(C):
            expected = x_rows[0, :, r, j] + prior_contrib
            diff = (out_rows[0, :, r, j] - expected).abs().max().item()
            assert diff < 1e-5, f"Row boundary mode wrong at row={r}, ch={j}: diff={diff}"

    # All channels in the same row should have the same prior contribution
    for r in range(1, num_rows):
        for j in range(1, C):
            contrib_0 = out_rows[0, :, r, 0] - x_rows[0, :, r, 0]
            contrib_j = out_rows[0, :, r, j] - x_rows[0, :, r, j]
            diff = (contrib_0 - contrib_j).abs().max().item()
            assert diff < 1e-6, f"Row boundary contributions differ at row={r}, ch={j}: {diff}"

    print("  conv1d row_boundary mode correctness ✓")


def test_conv1d_standard_fallback():
    """When S % C != 0, BlockCausalConv1d falls through to standard conv."""
    D, C, K = 4, 4, 3
    S = 13  # not divisible by 4

    conv = nn.Conv1d(D, D, K, padding=K - 1, groups=D)
    torch.manual_seed(42)

    wrapper = BlockCausalConv1d(conv, C, mode="column")

    x = torch.randn(1, D, S)
    out_wrapper = wrapper(x)
    out_direct = conv(x)

    diff = (out_wrapper - out_direct).abs().max().item()
    assert diff < 1e-6, f"Standard fallback should match: diff={diff}"
    print(f"  conv1d standard fallback (S%C!=0): diff={diff:.2e} ✓")


def test_column_chunk_vs_recurrent():
    """Column-mode chunk and recurrent must match."""
    configs = [
        (2, 4, 100, 16, 32, 10),
        (1, 2, 30, 8, 16, 10),
        (2, 4, 50, 16, 32, 5),
    ]
    for B, H, S, Dk, Dv, C in configs:
        q, k, v, g, beta = make_inputs(B, H, S, Dk, Dv)

        out_chunk, state_chunk = column_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=C,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out_rec, state_rec = column_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=True,
            num_channels=C,
            use_qk_l2norm_in_kernel=True,
        )

        out_diff = (out_chunk - out_rec).abs().max().item()
        # States have different shapes but same content: chunk=[B,H,C,Dk,Dv], rec=[B,H,C,Dk,Dv]
        state_diff = (state_chunk - state_rec).abs().max().item()
        assert out_diff < 1e-4, f"Column {(B, H, S, Dk, Dv, C)}: output diff {out_diff}"
        assert state_diff < 1e-4, f"Column {(B, H, S, Dk, Dv, C)}: state diff {state_diff}"
        print(f"  column chunk vs rec {(B, H, S, Dk, Dv, C)}: out={out_diff:.2e}, state={state_diff:.2e} ✓")


def test_column_independence():
    """In column mode, changing one column's input must not affect other columns."""
    B, H, S, Dk, Dv, C = 1, 4, 50, 16, 32, 10
    q, k, v, g, beta = make_inputs(B, H, S, Dk, Dv)

    out_base, _ = column_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=C,
        use_qk_l2norm_in_kernel=True,
    )

    # Perturb column 3's key at row 2
    k_pert = k.clone()
    k_pert[:, 2 * C + 3, :, :] += 10.0  # big perturbation to column 3, row 2
    out_pert, _ = column_chunk_gated_delta_rule(
        q,
        k_pert,
        v,
        g,
        beta,
        chunk_size=C,
        use_qk_l2norm_in_kernel=True,
    )

    # Column 3 should change
    col3_diff = (out_pert - out_base).reshape(B, -1, C, H, Dv)[:, :, 3].abs().max().item()
    assert col3_diff > 0.01, f"Column 3 should change, got diff={col3_diff}"

    # All other columns should be EXACTLY unchanged
    for j in range(C):
        if j == 3:
            continue
        col_diff = (out_pert - out_base).reshape(B, -1, C, H, Dv)[:, :, j].abs().max().item()
        assert col_diff == 0.0, f"Column {j} should be unchanged, got diff={col_diff}"

    print("  column mode: perturbing col 3 affects only col 3 ✓")


def test_column_vs_block_causal_differ():
    """Column and block-causal must produce different outputs."""
    B, H, S, Dk, Dv, C = 2, 4, 100, 16, 32, 10
    q, k, v, g, beta = make_inputs(B, H, S, Dk, Dv)

    out_col, _ = column_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=C,
        use_qk_l2norm_in_kernel=True,
    )
    out_bc, _ = block_causal_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=C,
        use_qk_l2norm_in_kernel=True,
    )

    diff = (out_col - out_bc).abs().max().item()
    assert diff > 0.01, f"Column and block-causal should differ, got diff={diff}"
    print(f"  column vs block-causal diff: {diff:.2e} (correctly different) ✓")


if __name__ == "__main__":
    print("Test 1: Chunk vs Recurrent (L2-normed qk, matching Qwen3.5)")
    test_chunk_vs_recurrent_l2norm()

    print("\nTest 2: Chunk vs Recurrent (small dims, no L2-norm)")
    test_chunk_vs_recurrent_small()

    print("\nTest 3: Block-causal vs standard causal differ")
    try:
        test_block_causal_vs_standard_differ()
    except ImportError:
        print("  SKIPPED (fla not installed)")

    print("\nTest 4: Channel independence (identical inputs → identical outputs)")
    test_channel_independence()

    print("\nTest 5: Conv1d column mode correctness")
    test_conv1d_column_mode()

    print("\nTest 6: Conv1d column mode no cross-channel leakage")
    test_conv1d_no_cross_channel_leakage()

    print("\nTest 7: Conv1d row_boundary mode correctness")
    test_conv1d_row_boundary_mode()

    print("\nTest 8: Conv1d standard fallback")
    test_conv1d_standard_fallback()

    print("\nTest 9: Column-mode chunk vs recurrent")
    test_column_chunk_vs_recurrent()

    print("\nTest 10: Column-mode independence (perturb one column)")
    test_column_independence()

    print("\nTest 11: Column vs block-causal differ")
    test_column_vs_block_causal_differ()

    print("\nAll tests passed!")
