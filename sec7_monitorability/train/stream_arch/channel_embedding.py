"""Channel embedding module for multi-stream models."""

import torch
from torch import nn


class ChannelEmbedding(nn.Module):
    """Learnable per-channel embedding added to token hidden states.

    Args:
        num_channels: Number of channels (streams).
        hidden_size: Model hidden dimension.
        method: "additive" adds embedding to hidden states, "none" is a pass-through.
    """

    def __init__(self, num_channels: int, hidden_size: int, method: str = "additive"):
        super().__init__()
        self.method = method
        self.num_channels = num_channels
        if method == "additive":
            self.embedding = nn.Embedding(num_channels, hidden_size)
        elif method == "none":
            pass
        else:
            raise ValueError(f"Unknown channel embedding method: {method}")

    def forward(
        self, hidden_states: torch.Tensor, channel_ids: torch.LongTensor
    ) -> torch.Tensor:
        if self.method == "none":
            return hidden_states
        channel_emb = self.embedding(channel_ids)
        return hidden_states + channel_emb
