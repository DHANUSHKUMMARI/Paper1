"""CLIP linear probe, the "same training data" baseline of CP4 (decisions of 16 Nov 2026).

- Features: the pinned CLIP ViT-L/14, frozen, without LoRA, on M2's input (the 256 crop resized
  to 224, CLIP-normalised); the last layer's class token and patch tokens after
  ``post_layernorm`` (:meth:`SemanticBranch.joint_tokens`, which shares the frozen layers with the
  model at test time).
- Training images: the models' own shards and epoch plans (crop rule and augmentation), several
  epochs, so several crops per image.
- Class-token probe: L2 logistic regression (binary real vs generated for the detection protocol,
  three classes for the unified one), fitted by L-BFGS on all training features; the L2 strength
  is chosen by validation AUC on the fixed validation crops (in-distribution only).
- Patch probe (unified): one linear layer on each 16 x 16 patch token, trained online in a single
  pass; a patch is edited when at least half of its pixels are; real and generated crops give
  all-zero targets. Several L2 strengths are trained side by side and chosen by validation
  per-image mask F1 (threshold 0.5).
- At test time the probe takes the model's tiles: class probabilities from the class token, and
  patch probabilities upsampled (bilinear, on the logits) to the tile, then the same stitching
  and thresholds as the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eviground.eval import metrics

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
PATCH_L2_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
NEG = -1.0e4  # logit of a class the probe never predicts


def patch_targets(masks: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """(B, 1, H, W) masks in {0, 1} -> (B, grid*grid): 1 where at least half the cell is edited."""
    share = F.adaptive_avg_pool2d(masks.float(), (grid, grid))
    return (share >= 0.5).float().flatten(1)


def fit_logreg(
    x: torch.Tensor, y: torch.Tensor, n_classes: int, l2: float, iters: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multinomial logistic regression with an L2 penalty on the weights, full-batch L-BFGS.
    Returns (W (D, C), b (C,))."""
    w = torch.zeros(x.shape[1], n_classes, device=x.device, requires_grad=True)
    b = torch.zeros(n_classes, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + 0.5 * l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def _auc(scores: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        auc = metrics.binary_auc(y == 1, scores[:, 1])
    else:
        auc = metrics.macro_ovr_auc(y, scores)["macro"]
    return float("nan") if auc is None else float(auc)


def choose_logreg(
    x: torch.Tensor,
    y: torch.Tensor,
    xv: torch.Tensor,
    yv: torch.Tensor,
    n_classes: int,
    grid: Sequence[float] = L2_GRID,
) -> dict[str, Any]:
    """Fits one probe per L2 strength and keeps the best by validation AUC."""
    tried = []
    best = None
    for l2 in grid:
        w, b = fit_logreg(x, y, n_classes, l2)
        p = torch.softmax(xv @ w + b, dim=1).cpu().numpy()
        auc = _auc(p, yv.cpu().numpy(), n_classes)
        tried.append({"l2": l2, "val_auc": auc})
        if best is None or not np.isfinite(best["val_auc"]) or auc > best["val_auc"]:
            best = {"l2": l2, "val_auc": auc, "w": w, "b": b}
    return {**best, "tried": tried}


class PatchProbes(nn.Module):
    """Several linear patch probes side by side (one per L2 strength), trained online."""

    def __init__(self, width: int, l2s: Sequence[float] = PATCH_L2_GRID) -> None:
        super().__init__()
        self.l2s = list(l2s)
        self.w = nn.Parameter(torch.zeros(len(self.l2s), width))
        self.b = nn.Parameter(torch.zeros(len(self.l2s)))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, N, width) -> logits (H, B, N), one row per probe."""
        return torch.einsum("bnd,hd->hbn", patches, self.w) + self.b[:, None, None]

    def loss(self, patches: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(patches)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.expand_as(logits), reduction="none"
        ).mean(dim=(1, 2))
        pen = torch.as_tensor(self.l2s, device=patches.device) * (self.w * self.w).sum(1)
        return (bce + 0.5 * pen).sum()


class ProbeHead(nn.Module):
    """The fitted probe as a model head: features -> 3-class logits and mask logits."""

    def __init__(
        self,
        w: torch.Tensor,
        b: torch.Tensor,
        classes: Sequence[int],
        patch_w: torch.Tensor | None = None,
        patch_b: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("w", w.float())
        self.register_buffer("b", b.float())
        self.classes = list(classes)  # model class index of each probe output
        self.register_buffer("patch_w", None if patch_w is None else patch_w.float())
        self.register_buffer("patch_b", None if patch_b is None else patch_b.float())

    def forward(
        self, cls: torch.Tensor, patches: torch.Tensor, size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        n = cls.shape[0]
        z = cls.float() @ self.w + self.b
        logits = torch.full((n, 3), NEG, device=cls.device)
        logits[:, self.classes] = z
        if self.patch_w is None:
            mask = torch.full((n, 1, *size), NEG, device=cls.device)
        else:
            g = int(round(patches.shape[1] ** 0.5))
            pl = (patches.float() @ self.patch_w + self.patch_b).view(n, 1, g, g)
            mask = F.interpolate(pl, size=size, mode="bilinear", align_corners=False)
        return {"logits": logits, "mask_logits": mask}

    def state(self) -> dict[str, Any]:
        return {
            "w": self.w.cpu(),
            "b": self.b.cpu(),
            "classes": self.classes,
            "patch_w": None if self.patch_w is None else self.patch_w.cpu(),
            "patch_b": None if self.patch_b is None else self.patch_b.cpu(),
        }

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> ProbeHead:
        return cls(s["w"], s["b"], s["classes"], s.get("patch_w"), s.get("patch_b"))


def _rgb(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return model.views.rgb_view(images)


def _features(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The probe's features in fp32 (bf16 autocast for the CLIP pass on a GPU only; the fits
    themselves always run in fp32). The class token is copied out: it is a view of the whole
    (B, 1 + patches, D) token tensor, and a view kept across batches would keep every token of
    every batch (notebook 08 ran out of GPU memory, 17 Nov 2026)."""
    amp = images.device.type == "cuda"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        cls, patches = model.semantic.probe_features(_rgb(model, images))
    return cls.float().clone(), patches.float()


@torch.no_grad()
def collect_cls(
    model: nn.Module, batches: Iterable[dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-token features and labels of every batch (fixed validation crops)."""
    xs, ys = [], []
    for batch in batches:
        cls, _p = _features(model, batch["image"].to(device))
        xs.append(cls)
        ys.append(torch.as_tensor(batch["label"], device=device))
    return torch.cat(xs), torch.cat(ys)


def train_probes(
    model: nn.Module,
    train_batches: Iterable[dict[str, Any]],
    val_batches: Callable[[], Iterable[dict[str, Any]]],
    *,
    profile: str,
    device: torch.device,
    lr: float = 1e-3,
    log: Any = print,
) -> tuple[ProbeHead, dict[str, Any]]:
    """Fits the probes of ``profile`` ('detection': real vs generated, class token only;
    'unified': three classes plus the patch probe). ``model`` supplies the frozen CLIP (its
    LoRA is switched off for the features). ``val_batches()`` gives a fresh pass over the fixed
    validation crops (it is read twice). Returns the head and a report."""
    classes = [0, 1] if profile == "detection" else [0, 1, 2]
    remap = {c: i for i, c in enumerate(classes)}
    patch = PatchProbes(model.semantic.width).to(device) if profile == "unified" else None
    opt = torch.optim.AdamW(patch.parameters(), lr=lr, weight_decay=0.0) if patch else None
    xs, ys = [], []
    crops = 0
    for batch in train_batches:
        images = batch["image"].to(device)
        cls, patches = _features(model, images)
        xs.append(cls)
        ys.append(torch.as_tensor([remap[int(v)] for v in batch["label"]], device=device))
        crops += images.shape[0]
        if patch is not None:
            opt.zero_grad(set_to_none=True)
            patch.loss(patches, patch_targets(batch["mask"].to(device))).backward()
            opt.step()
    x, y = torch.cat(xs), torch.cat(ys)
    xv, yv_model = collect_cls(model, val_batches(), device)
    yv = torch.as_tensor([remap[int(v)] for v in yv_model.tolist()], device=device)
    fit = choose_logreg(x, y, xv, yv, len(classes))
    report: dict[str, Any] = {
        "profile": profile,
        "train_crops": crops,
        "val_crops": int(xv.shape[0]),
        "class_token": {"l2": fit["l2"], "val_auc": fit["val_auc"], "tried": fit["tried"]},
    }
    pw = pb = None
    if patch is not None:
        f1 = patch_val_f1(model, patch, val_batches(), device)
        k = int(np.argmax(f1))
        pw, pb = patch.w[k].detach()[:, None], patch.b[k].detach()[None]
        report["patch"] = {
            "l2": patch.l2s[k],
            "val_mask_f1": float(f1[k]),
            "tried": [
                {"l2": l2, "val_mask_f1": float(v)} for l2, v in zip(patch.l2s, f1, strict=True)
            ],
        }
    log(f"probe {profile}: {report}")
    return ProbeHead(fit["w"], fit["b"], classes, pw, pb), report


@torch.no_grad()
def patch_val_f1(
    model: nn.Module,
    patch: PatchProbes,
    val_batches: Iterable[dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    """Per-image mean mask F1 on the edited validation crops, one value per patch probe."""
    scores: list[np.ndarray] = []
    for batch in val_batches:
        edited = torch.as_tensor(batch["label"]) == 2
        if not edited.any():
            continue
        images = batch["image"][edited].to(device)
        masks = batch["mask"][edited].to(device) > 0.5
        _c, patches = _features(model, images)
        logits = patch(patches)  # (H, B, N)
        h, b, n = logits.shape
        g = int(round(n**0.5))
        up = F.interpolate(
            logits.reshape(h * b, 1, g, g), size=masks.shape[-2:], mode="bilinear"
        ).view(h, b, *masks.shape[-2:])
        pred = up > 0
        tp = (pred & masks[None, :, 0]).sum(dim=(2, 3)).float()
        fp = (pred & ~masks[None, :, 0]).sum(dim=(2, 3)).float()
        fn = (~pred & masks[None, :, 0]).sum(dim=(2, 3)).float()
        f1 = torch.where(tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1), 1.0)
        scores.append(f1.cpu().numpy())
    if not scores:
        return np.zeros(len(patch.l2s))
    return np.concatenate(scores, axis=1).mean(axis=1)
