"""M4 — bidirectional cross-attention fusion with a learned gate (Research Specification §8, M4).

S~ = S + CA(S, T, T),  T~ = T + CA(T, S, S)  (pre-norm, then an MLP with a residual), stacked
``layers`` times; then g = sigmoid(MLP([mean S~ ; mean T~])) per image and
Z = g * S~_patch + (1 - g) * T~ on the 16 x 16 token grid. The gate is returned (it is evidence
in M5).
"""

from __future__ import annotations

import torch
from torch import nn


class CrossBlock(nn.Module):
    """x + CA(LN(x), LN(y)) followed by x + MLP(LN(x))."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(y)
        x = x + self.attn(self.norm_q(x), kv, kv, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


class BiFusion(nn.Module):
    def __init__(self, d: int = 512, layers: int = 2, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.s_blocks = nn.ModuleList(CrossBlock(d, heads, dropout) for _ in range(layers))
        self.t_blocks = nn.ModuleList(CrossBlock(d, heads, dropout) for _ in range(layers))
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """``s``: (B, 1 + N, d) semantic tokens (class token first); ``t``: (B, N, d) trace
        tokens on the same grid. Returns s~, t~, the gate g (B, 1) and Z (B, N, d)."""
        for sb, tb in zip(self.s_blocks, self.t_blocks, strict=True):
            s, t = sb(s, t), tb(t, s)  # both read the inputs of this layer
        g = torch.sigmoid(self.gate(torch.cat([s[:, 1:].mean(dim=1), t.mean(dim=1)], dim=-1)))
        z = g.unsqueeze(-1) * s[:, 1:] + (1 - g.unsqueeze(-1)) * t
        return {"s": s, "t": t, "gate": g, "z": z}
