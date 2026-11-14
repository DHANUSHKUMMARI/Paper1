"""EviGround: M1 views + M2 semantic + M3 trace + M4 fusion and heads (spec section 8).

``EviGround.forward`` takes images in [0, 1] (B, 3, H, W) and returns class logits (B, 3), mask
logits (B, 1, H, W), the pooling weights alpha (B, 256), the gate g (B, 1) and the fused tokens Z.
Views are computed inside the model, on its device. ``predict_tiled`` applies the test-time rule:
256 tiles, P(real) and P(fully generated) the mean over tiles, P(edited) the maximum, edited if
P(edited) >= the edit threshold, otherwise the larger of the other two; tile masks are stitched.

The detection-protocol model is the same network with the edited class never sampled; its P1
score is 1 - P(real) (decision of 15 Nov 2026).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from eviground.data import tiles as tiling
from eviground.models.fusion import BiFusion
from eviground.models.heads import CLASSES, ClassHead, MaskDecoder
from eviground.models.semantic import DEFAULT_BACKBONE, SemanticBranch
from eviground.models.trace import TraceBranch
from eviground.models.views import ForensicViews

EDITED = CLASSES.index("edited")
REAL = CLASSES.index("real")


class EviGround(nn.Module):
    def __init__(
        self,
        *,
        d: int = 512,
        clip_size: int = 224,
        semantic: Mapping[str, Any] | None = None,
        trace: Mapping[str, Any] | None = None,
        fusion: Mapping[str, Any] | None = None,
        head_dropout: float = 0.1,
        edit_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        semantic = dict(semantic or {})
        trace = dict(trace or {})
        fusion = dict(fusion or {})
        self.views = ForensicViews(clip_size=clip_size)
        self.semantic = SemanticBranch(d=d, **semantic)
        self.trace = TraceBranch(d=d, **trace)
        self.fusion = BiFusion(d=d, **fusion)
        self.cls_head = ClassHead(d, dropout=head_dropout)
        c4, c8, _ = self.trace.channels
        self.decoder = MaskDecoder(d, c8=c8, c4=c4)
        self.d = d
        self.edit_threshold = edit_threshold

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = images.shape
        v = self.views(images)
        s = self.semantic(v["rgb"])  # (B, 1 + g*g, d)
        grid = int(round((s.shape[1] - 1) ** 0.5))
        tr = self.trace(v["trace"])
        t16 = tr["t16"]
        if t16.shape[-2:] != (grid, grid):  # inputs other than 256: align with CLIP's grid
            t16 = F.adaptive_avg_pool2d(t16, (grid, grid))
        t = t16.flatten(2).transpose(1, 2)
        f = self.fusion(s, t)
        logits, alpha = self.cls_head(f["s"][:, 0], f["z"])
        z_map = f["z"].transpose(1, 2).reshape(b, self.d, grid, grid)
        t_map = f["t"].transpose(1, 2).reshape(b, self.d, grid, grid)
        mask_logits = self.decoder(z_map, t_map, tr["s8"], tr["s4"], (h, w))
        return {
            "logits": logits,
            "mask_logits": mask_logits,
            "alpha": alpha,
            "gate": f["gate"],
            "z": f["z"],
        }

    @torch.no_grad()
    def predict_tiled(self, image: torch.Tensor, batch_size: int = 9) -> dict[str, Any]:
        """``image``: (3, H, W) in [0, 1]. Tiles, aggregated probabilities, stitched mask."""
        _, h, w = image.shape
        boxes = tiling.tile_boxes(h, w)
        crops = torch.stack([image[:, t:b, le:r] for le, t, r, b in boxes])
        probs, masks = [], []
        for i in range(0, len(crops), batch_size):
            out = self(crops[i : i + batch_size])
            probs.append(torch.softmax(out["logits"].float(), dim=-1))
            masks.append(torch.sigmoid(out["mask_logits"].float())[:, 0])
        p = torch.cat(probs)
        agg = {
            "real": float(p[:, REAL].mean()),
            "generated": float(p[:, CLASSES.index("generated")].mean()),
            "edited": float(p[:, EDITED].max()),
        }
        if agg["edited"] >= self.edit_threshold:
            label = "edited"
        else:
            label = "real" if agg["real"] >= agg["generated"] else "generated"
        mask = tiling.stitch(torch.cat(masks), boxes, h, w)
        return {
            "probs": agg,
            "label": label,
            "mask": mask,
            "tiles": len(boxes),
            "coverage": tiling.coverage(h) * tiling.coverage(w),
        }

    def param_groups(self, lr: float, lr_lora: float, weight_decay: float) -> list[dict[str, Any]]:
        """AdamW groups: LoRA at ``lr_lora``; projection, trace, fusion and heads at ``lr``."""
        lora = {id(p) for p in self.semantic.lora_parameters()}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in lora]
        return [
            {
                "params": self.semantic.lora_parameters(),
                "lr": lr_lora,
                "weight_decay": weight_decay,
            },
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]


def disable_dropout(model: nn.Module) -> None:
    """Dropout off everywhere (64-image overfit test of CP2)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        elif isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.0
        elif hasattr(m, "lora_dropout"):
            for drop in m.lora_dropout.values():
                if isinstance(drop, nn.Dropout):
                    drop.p = 0.0


def build_model(cfg: Mapping[str, Any]) -> EviGround:
    """From the ``model`` section of a config (see ``configs/base.yaml``)."""
    sem = dict(cfg.get("semantic", {}))
    lora = dict(sem.pop("lora", {}) or {})
    semantic = {
        "backbone": sem.get("backbone", DEFAULT_BACKBONE),
        "tiny": dict(sem["tiny"]) if sem.get("tiny") else None,
        "cache_dir": sem.get("cache_dir"),
        "local_files_only": bool(sem.get("local_files_only", False)),
        "lora_rank": int(lora.get("rank", 8)),
        "lora_alpha": int(lora.get("alpha", 16)),
        "lora_layers": list(lora.get("layers", [20, 21, 22, 23])),
        "lora_targets": list(lora.get("targets", ["q_proj", "v_proj"])),
        "lora_dropout": float(lora.get("dropout", 0.0)),
    }
    tr = dict(cfg.get("trace", {}))
    fu = dict(cfg.get("fusion", {}))
    return EviGround(
        d=int(cfg.get("d", 512)),
        clip_size=int(cfg.get("clip_size", 224)),
        semantic=semantic,
        trace={
            "name": tr.get("name", "efficientnet_b0"),
            "pretrained": bool(tr.get("pretrained", True)),
            "channels_last": bool(tr.get("channels_last", False)),
        },
        fusion={
            "layers": int(fu.get("layers", 2)),
            "heads": int(fu.get("heads", 8)),
            "dropout": float(fu.get("dropout", 0.1)),
        },
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        edit_threshold=float(cfg.get("edit_threshold", 0.5)),
    )
