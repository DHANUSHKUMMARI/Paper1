"""M3 — forensic-trace branch (Research Specification section 8, M3).

EfficientNet-B0 (``timm``, ImageNet-initialised; timm adapts the first convolution to 4 input
channels) on the concatenated noise and frequency views. Feature maps at strides 4, 8 and 16 are
kept; the stride-16 map is projected (1 x 1 convolution) to width ``d`` and flattened into trace
tokens. For a 256 x 256 crop that is a 16 x 16 grid, one token per CLIP patch.
"""

from __future__ import annotations

import torch
from torch import nn


class TraceBranch(nn.Module):
    def __init__(
        self,
        name: str = "efficientnet_b0",
        *,
        pretrained: bool = True,
        in_chans: int = 4,
        d: int = 512,
        channels_last: bool = False,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            name,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_chans,
            out_indices=(1, 2, 3),
        )
        reductions = list(self.backbone.feature_info.reduction())
        if reductions != [4, 8, 16]:
            raise ValueError(f"{name}: expected strides [4, 8, 16], got {reductions}")
        self.channels = list(self.backbone.feature_info.channels())
        self.proj = nn.Conv2d(self.channels[-1], d, kernel_size=1)
        self.channels_last = channels_last
        if channels_last:
            self.to(memory_format=torch.channels_last)

    def forward(self, trace: torch.Tensor) -> dict[str, torch.Tensor]:
        """``trace``: (B, 4, H, W) -> s4, s8, s16 maps and the projected stride-16 map ``t16``
        (B, d, H/16, W/16)."""
        if self.channels_last:
            trace = trace.contiguous(memory_format=torch.channels_last)
        s4, s8, s16 = self.backbone(trace)
        return {"s4": s4, "s8": s8, "s16": s16, "t16": self.proj(s16)}
