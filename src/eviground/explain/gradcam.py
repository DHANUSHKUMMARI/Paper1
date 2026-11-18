"""M5 attribution maps (spec section 8, M5; CP5).

- :func:`gradcam_pp`: Grad-CAM++ on the fused token grid Z (16 x 16), for one class logit. It
  returns the **raw** relu map, upsampled to the tile, with no normalization.
- :func:`tiled_map`: raw tile maps stitched like the mask (mean where tiles overlap), then
  normalized **once per image** by the image maximum (Dhanush, 18 Nov 2026), so that a tile with
  little attribution stays near 0 next to a strongly attributed one.
- :func:`decoder_map`: the mask decoder's sigmoid output, the CP5 fallback evidence map.
- :func:`alpha_map`: the attention-pooling weights alpha on the token grid (ablation 4).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from eviground.data import tiles


def _grid(n_tokens: int) -> int:
    g = math.isqrt(n_tokens)
    if g * g != n_tokens:
        raise ValueError(f"{n_tokens} tokens do not form a square grid")
    return g


def gradcam_pp(model: torch.nn.Module, images: torch.Tensor, cls: int) -> torch.Tensor:
    """Raw Grad-CAM++ maps (B, H, W) >= 0 for class ``cls`` on the fused tokens Z."""
    b, _, h, w = images.shape
    with torch.enable_grad():
        out = model(images)
        z = out["z"]
        z.retain_grad()
        model.zero_grad(set_to_none=True)
        out["logits"][:, cls].sum().backward()
        a, g = z.detach(), z.grad.detach()  # (B, N, d)
    g2, g3 = g.pow(2), g.pow(3)
    denom = 2 * g2 + (a * g3).sum(dim=1, keepdim=True)
    weights = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
    w_k = (weights * F.relu(g)).sum(dim=1)  # (B, d)
    cam = F.relu((a * w_k[:, None, :]).sum(dim=-1))  # (B, N)
    grid = _grid(cam.shape[1])
    cam = cam.reshape(b, 1, grid, grid)
    return F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)[:, 0]


def normalize(m: torch.Tensor) -> torch.Tensor:
    """Divide by the maximum (a map that is all 0 stays 0)."""
    top = m.max()
    return m / top if top > 0 else m


def tiled_map(
    model: torch.nn.Module, image: torch.Tensor, cls: int, batch_size: int = tiles.TILE_BATCH
) -> torch.Tensor:
    """Grad-CAM++ over every tile of ``image`` (3, H, W): raw tile maps stitched, then
    normalized once by the image maximum."""
    _, h, w = image.shape
    boxes = tiles.tile_boxes(h, w)
    maps = []
    for i in range(0, len(boxes), batch_size):
        crops = torch.stack([image[:, t:b, le:r] for le, t, r, b in boxes[i : i + batch_size]])
        maps.append(gradcam_pp(model, crops, cls))
    return normalize(tiles.stitch(torch.cat(maps), boxes, h, w))


@torch.no_grad()
def decoder_map(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """The stitched mask probability (the CP5 fallback evidence map)."""
    return model.predict_tiled(image)["mask"]


@torch.no_grad()
def alpha_map(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """The attention-pooling weights alpha (B, H, W), each tile's map summing to 1 on the grid."""
    b, _, h, w = images.shape
    alpha = model(images)["alpha"].float()
    grid = _grid(alpha.shape[1])
    a = alpha.reshape(b, 1, grid, grid)
    return F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)[:, 0]
