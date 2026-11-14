"""M1 — forensic input views (Research Specification section 8, M1; Implementation Brief M1).

From an image batch in [0, 1] (B, 3, H, W) build, on the device the model runs on:

- ``rgb``: resized to 224 x 224 (antialias) and CLIP-normalised, for the semantic branch only;
- ``noise``: SRM residual R of the grey image, three fixed 5 x 5 high-pass kernels (as in RGB-N,
  Zhou et al., CVPR 2018), clamped to [-3, 3] and scaled to [-1, 1] (pixel scale 0-255);
- ``freq``: block-wise 8 x 8 orthonormal DCT-II of the grey image, log(1 + |c|), standardised per
  image, each coefficient laid out in place of its block (full resolution);
- ``trace``: concat(noise, freq), the 4-channel input of the forensic-trace branch.

R and F are never down-sampled (spec: generator traces disappear otherwise). Rows and columns
beyond the last full 8 x 8 block (only for test images whose side is not a multiple of 8) get a
frequency value of 0 after standardisation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
GREY = (0.299, 0.587, 0.114)

SRM_KERNELS = (
    (1 / 4)
    * torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ),
    (1 / 12)
    * torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    ),
    (1 / 2)
    * torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ),
)


def dct_matrix(n: int = 8) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n x n): coefficients of a block X are D @ X @ D.T."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2 / n)
    d[0] /= math.sqrt(2)
    return d.float()


class ForensicViews(nn.Module):
    """The three M1 views; fixed buffers, no trainable parameters."""

    def __init__(self, clip_size: int = 224, dct_block: int = 8) -> None:
        super().__init__()
        self.clip_size = clip_size
        self.block = dct_block
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("grey_w", torch.tensor(GREY).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("srm", torch.stack(SRM_KERNELS).unsqueeze(1), persistent=False)
        self.register_buffer("dct", dct_matrix(dct_block), persistent=False)

    def rgb_view(self, x: torch.Tensor) -> torch.Tensor:
        rgb = F.interpolate(
            x,
            size=(self.clip_size, self.clip_size),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
        return (rgb - self.mean) / self.std

    def grey(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.grey_w).sum(dim=1, keepdim=True)

    def noise_view(self, grey: torch.Tensor) -> torch.Tensor:
        r = F.conv2d(grey * 255.0, self.srm, padding=2)
        return r.clamp(-3.0, 3.0) / 3.0

    def freq_view(self, grey: torch.Tensor) -> torch.Tensor:
        b, _, h, w = grey.shape
        n = self.block
        hb, wb = h // n, w // n
        out = torch.zeros_like(grey)
        if hb == 0 or wb == 0:
            return out
        g = grey[:, 0, : hb * n, : wb * n] * 255.0
        blocks = g.reshape(b, hb, n, wb, n).permute(0, 1, 3, 2, 4)  # (B, hb, wb, n, n)
        coef = self.dct @ blocks @ self.dct.T
        mag = torch.log1p(coef.abs())
        flat = mag.reshape(b, -1)
        mag = (mag - flat.mean(dim=1).view(b, 1, 1, 1, 1)) / (
            flat.std(dim=1).view(b, 1, 1, 1, 1) + 1e-6
        )
        out[:, 0, : hb * n, : wb * n] = mag.permute(0, 1, 3, 2, 4).reshape(b, hb * n, wb * n)
        return out

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``x``: (B, 3, H, W) float in [0, 1]."""
        grey = self.grey(x)
        noise = self.noise_view(grey)
        freq = self.freq_view(grey)
        return {
            "rgb": self.rgb_view(x),
            "noise": noise,
            "freq": freq,
            "trace": torch.cat([noise, freq], dim=1),
        }
