#!/usr/bin/env python3
"""Test that GatedDeltaNet state forwarding produces correct results.

Verifies the core invariant: processing a multi-row sequence all at once
(chunk mode) must produce the same output as processing row-by-row with
state carried forward between rows.  This tests the fix for the two bugs:

1. Recurrent state not forwarded (initial_state=None in chunk mode)
2. Conv state not forwarded (BlockCausalConv1d sees only 1 row per step)

Usage:
    uv run python tests/test_deltanet_state.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import torch
import torch.nn.functional as F
from stream_arch.block_causal_deltanet import (
    BlockCausalConv1d,
    block_causal_chunk_gated_delta_rule,
    block_causal_recurrent_gated_delta_rule,
    column_chunk_gated_delta_rule,
    column_recurrent_gated_delta_rule,
)


def test_column_chunk_state_forwarding():
    """Column chunk: processing 4 rows at once == processing 2+2 with state."""
    torch.manual_seed(42)
    B, H, C, Dk, Dv = 1, 2, 3, 8, 8
    num_rows = 4
    S = num_rows * C

    query = torch.randn(B, S, H, Dk)
    key = torch.randn(B, S, H, Dk)
    value = torch.randn(B, S, H, Dv)
    g = torch.randn(B, S, H)
    beta = torch.randn(B, S, H)

    # All at once
    out_full, state_full = column_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=C,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # First half
    S1 = 2 * C
    out1, state1 = column_chunk_gated_delta_rule(
        query[:, :S1],
        key[:, :S1],
        value[:, :S1],
        g=g[:, :S1],
        beta=beta[:, :S1],
        chunk_size=C,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # Second half with state from first
    out2, state2 = column_chunk_gated_delta_rule(
        query[:, S1:],
        key[:, S1:],
        value[:, S1:],
        g=g[:, S1:],
        beta=beta[:, S1:],
        chunk_size=C,
        initial_state=state1,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    out_split = torch.cat([out1, out2], dim=1)
    assert torch.allclose(out_full, out_split, atol=1e-5), f"Column chunk: max diff = {(out_full - out_split).abs().max().item():.2e}"
    assert torch.allclose(state_full, state2, atol=1e-5), f"Column chunk state: max diff = {(state_full - state2).abs().max().item():.2e}"
    print("PASS: column_chunk state forwarding")


def test_block_causal_chunk_state_forwarding():
    """Block-causal chunk: processing 4 rows at once == processing 2+2 with state."""
    torch.manual_seed(42)
    B, H, C, Dk, Dv = 1, 2, 3, 8, 8
    num_rows = 4
    S = num_rows * C

    query = torch.randn(B, S, H, Dk)
    key = torch.randn(B, S, H, Dk)
    value = torch.randn(B, S, H, Dv)
    g = torch.randn(B, S, H)
    beta = torch.randn(B, S, H)

    # All at once
    out_full, state_full = block_causal_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=C,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # First half
    S1 = 2 * C
    out1, state1 = block_causal_chunk_gated_delta_rule(
        query[:, :S1],
        key[:, :S1],
        value[:, :S1],
        g=g[:, :S1],
        beta=beta[:, :S1],
        chunk_size=C,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # Second half with state from first
    out2, state2 = block_causal_chunk_gated_delta_rule(
        query[:, S1:],
        key[:, S1:],
        value[:, S1:],
        g=g[:, S1:],
        beta=beta[:, S1:],
        chunk_size=C,
        initial_state=state1,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    out_split = torch.cat([out1, out2], dim=1)
    assert torch.allclose(out_full, out_split, atol=1e-4), f"Block-causal chunk: max diff = {(out_full - out_split).abs().max().item():.2e}"
    assert torch.allclose(state_full, state2, atol=1e-4), (
        f"Block-causal chunk state: max diff = {(state_full - state2).abs().max().item():.2e}"
    )
    print("PASS: block_causal_chunk state forwarding")


def test_block_causal_conv1d_with_state():
    """BlockCausalConv1d: processing 4 rows at once == prepending prior rows."""
    torch.manual_seed(42)
    C = 3
    K = 4
    D = 8
    B = 1
    num_rows = 4
    S = num_rows * C

    # Create a standard depthwise conv and wrap it
    conv_orig = torch.nn.Conv1d(D, D, K, groups=D, bias=True, padding=K - 1)
    block_conv = BlockCausalConv1d(conv_orig, C, mode="column")

    x = torch.randn(B, D, S)

    # All at once: BlockCausalConv1d processes 4 rows
    out_full = block_conv(x)  # [B, D, S + K - 1]
    out_full = F.silu(out_full[:, :, :S])

    # Simulate row-by-row with state prepending
    state_len = (K - 1) * C
    conv_state = torch.zeros(B, D, state_len)
    outputs = []

    for r in range(num_rows):
        row_data = x[:, :, r * C : (r + 1) * C]  # [B, D, C]
        extended = torch.cat([conv_state, row_data], dim=-1)  # [B, D, state_len + C]
        conv_out = block_conv(extended)  # [B, D, state_len + C + K - 1]
        row_out = F.silu(conv_out[:, :, state_len : state_len + C])
        outputs.append(row_out)
        # Update state: last state_len positions of pre-conv input
        conv_state = extended[:, :, -state_len:].clone()

    out_split = torch.cat(outputs, dim=-1)  # [B, D, S]

    assert torch.allclose(out_full, out_split, atol=1e-5), (
        f"BlockCausalConv1d state: max diff = {(out_full - out_split).abs().max().item():.2e}"
    )
    print("PASS: BlockCausalConv1d state prepending")


def test_column_chunk_vs_recurrent():
    """Column chunk and recurrent modes should produce the same output."""
    torch.manual_seed(42)
    B, H, C, Dk, Dv = 1, 2, 3, 8, 8
    num_rows = 4
    S = num_rows * C

    query = torch.randn(B, S, H, Dk)
    key = torch.randn(B, S, H, Dk)
    value = torch.randn(B, S, H, Dv)
    g = torch.randn(B, S, H)
    beta = torch.randn(B, S, H)

    out_chunk, state_chunk = column_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=C,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    out_recurrent, state_recurrent = column_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        num_channels=C,
    )

    assert torch.allclose(out_chunk, out_recurrent, atol=1e-4), (
        f"Column chunk vs recurrent: max diff = {(out_chunk - out_recurrent).abs().max().item():.2e}"
    )
    assert torch.allclose(state_chunk, state_recurrent, atol=1e-4), (
        f"Column state chunk vs recurrent: max diff = {(state_chunk - state_recurrent).abs().max().item():.2e}"
    )
    print("PASS: column_chunk vs column_recurrent consistency")


def test_block_causal_chunk_vs_recurrent():
    """Block-causal chunk and recurrent modes should produce the same output."""
    torch.manual_seed(42)
    B, H, C, Dk, Dv = 1, 2, 3, 8, 8
    num_rows = 4
    S = num_rows * C

    query = torch.randn(B, S, H, Dk)
    key = torch.randn(B, S, H, Dk)
    value = torch.randn(B, S, H, Dv)
    g = torch.randn(B, S, H)
    beta = torch.randn(B, S, H)

    out_chunk, state_chunk = block_causal_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=C,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    out_recurrent, state_recurrent = block_causal_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        num_channels=C,
    )

    assert torch.allclose(out_chunk, out_recurrent, atol=1e-4), (
        f"Block-causal chunk vs recurrent: max diff = {(out_chunk - out_recurrent).abs().max().item():.2e}"
    )
    assert torch.allclose(state_chunk, state_recurrent, atol=1e-4), (
        f"Block-causal state chunk vs recurrent: max diff = {(state_chunk - state_recurrent).abs().max().item():.2e}"
    )
    print("PASS: block_causal_chunk vs block_causal_recurrent consistency")


if __name__ == "__main__":
    test_column_chunk_state_forwarding()
    test_block_causal_chunk_state_forwarding()
    test_block_causal_conv1d_with_state()
    test_column_chunk_vs_recurrent()
    test_block_causal_chunk_vs_recurrent()
    print("\nAll tests passed!")
