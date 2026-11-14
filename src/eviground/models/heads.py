"""M4 — classification and localization heads (Research Specification section 8, M4).

- Classification: attention pooling over the fused tokens Z (Ilse et al., 2018:
  alpha = softmax over tokens of w . tanh(V z), one weight per token, summing to 1), joined with
  the class token of S~ -> LayerNorm -> MLP -> 3 logits {real, fully generated, locally edited}.
  alpha is returned (it carries the attention-alignment loss and is evidence in M5).
- Localization: an FPN-lite decoder. Z and T~ (16 x 16) are fused, then up-sampled and fused with
  the stride-8 and stride-4 trace maps; 1-channel logits at 1/4 resolution are up-sampled
  bilinearly to the input size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

CLASSES = ("real", "generated", "edited")


class AttentionPool(nn.Module):
    def __init__(self, d: int, hidden: int = 128) -> None:
        super().__init__()
        self.v = nn.Linear(d, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.softmax(self.w(torch.tanh(self.v(z))).squeeze(-1), dim=-1)  # (B, N)
        return (alpha.unsqueeze(-1) * z).sum(dim=1), alpha


class ClassHead(nn.Module):
    def __init__(self, d: int, n_classes: int = len(CLASSES), dropout: float = 0.1) -> None:
        super().__init__()
        self.pool = AttentionPool(d)
        self.norm = nn.LayerNorm(2 * d)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes)
        )

    def forward(self, cls: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, alpha = self.pool(z)
        return self.mlp(self.norm(torch.cat([cls, pooled], dim=-1))), alpha


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class MaskDecoder(nn.Module):
    """FPN-lite: (Z, T~) at the token grid -> + s8 -> + s4 -> logits -> input size."""

    def __init__(self, d: int, c8: int, c4: int, width: int = 128) -> None:
        super().__init__()
        self.top = _conv(2 * d, width)
        self.lat8 = nn.Conv2d(c8, width, 1)
        self.lat4 = nn.Conv2d(c4, width, 1)
        self.fuse8 = _conv(width, width)
        self.fuse4 = _conv(width, width)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        s8: torch.Tensor,
        s4: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        """``z``, ``t``: (B, d, g, g) grids; ``s8``/``s4``: trace maps; returns (B, 1, H, W)."""
        x = self.top(torch.cat([z, t], dim=1))
        x = F.interpolate(x, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse8(x + self.lat8(s8))
        x = F.interpolate(x, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse4(x + self.lat4(s4))
        return F.interpolate(self.out(x), size=size, mode="bilinear", align_corners=False)
