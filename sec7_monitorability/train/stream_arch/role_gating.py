"""Role gating MLP for channel-aware attention bias."""

import torch
from torch import nn


class RoleGatingMLP(nn.Module):
    """Produce role logits for each query position.

    Supports:
      - granularity="layer": output [B,Q,C]
      - granularity="head":  output [B,H,Q,C]
    mode:
      - "query": use query_hidden
      - "query_ctx": concat(query_hidden, ctx_summary)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_roles: int,
        granularity: str = "layer",
        mode: str = "query",
        mlp_hidden: int = 0,
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
            x = torch.cat([q_hidden, ctx], dim=-1)  # [B,Q,2D]
        else:
            x = q_hidden  # [B,Q,D]

        y = self.net(x)  # [B,Q,out_dim]
        B, Q, _ = y.shape

        if self.granularity == "layer":
            return y.view(B, Q, self.num_roles)  # [B,Q,C]
        else:
            return (
                y.view(B, Q, self.num_heads, self.num_roles)
                .permute(0, 2, 1, 3)
                .contiguous()
            )  # [B,H,Q,C]
