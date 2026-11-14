"""Training loss (Research Specification section 8, M4; decision of 15 Nov 2026).

L = CE(y_hat, y) + lambda_seg * (BCE(M_hat, M) + Dice(M_hat, M)) + lambda_att * L_att

- CE with label smoothing (0.05);
- BCE on **all** samples: real and fully generated images have all-zero masks, so the decoder
  learns to say "nothing edited" too;
- Dice on **edited samples only** (Dice of an all-zero target is not informative);
- L_att = 1 - sum_p alpha_p * M_p on edited samples only, with the mask average-pooled to the
  token grid; first-order (no Grad-CAM++ in the loss).

All losses are computed in float32 (safe under bf16/fp16 autocast).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from eviground.models.heads import CLASSES

EDITED = CLASSES.index("edited")


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss per sample (B,), 0 for a perfect prediction."""
    prob = torch.sigmoid(logits.float()).flatten(1)
    t = target.float().flatten(1)
    inter = (prob * t).sum(1)
    return 1 - (2 * inter + eps) / (prob.sum(1) + t.sum(1) + eps)


def attention_loss(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """1 - sum(alpha * M_grid) per sample; ``alpha`` (B, g*g) sums to 1, ``mask`` (B, 1, H, W)."""
    grid = int(round(alpha.shape[1] ** 0.5))
    m = F.adaptive_avg_pool2d(mask.float(), (grid, grid)).flatten(1)
    return 1 - (alpha.float() * m).sum(1)


def eviground_loss(
    out: dict[str, torch.Tensor],
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    lambda_seg: float = 1.0,
    lambda_att: float = 0.1,
    label_smoothing: float = 0.05,
) -> dict[str, torch.Tensor]:
    """``labels`` (B,) class indices; ``masks`` (B, 1, H, W) in {0, 1} (zeros unless edited)."""
    logits = out["logits"].float()
    mask_logits = out["mask_logits"].float()
    ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    bce = F.binary_cross_entropy_with_logits(mask_logits, masks.float())
    edited = labels == EDITED
    zero = logits.new_zeros(())
    if edited.any():
        dice = dice_loss(mask_logits[edited], masks[edited]).mean()
        att = attention_loss(out["alpha"][edited], masks[edited]).mean()
    else:
        dice, att = zero, zero
    total = ce + lambda_seg * (bce + dice) + lambda_att * att
    return {"total": total, "ce": ce, "bce": bce, "dice": dice, "att": att}
