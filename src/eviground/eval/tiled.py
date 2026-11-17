"""Tiled evaluation of whole images (CP4; test-time rule with full coverage, 16 Nov 2026).

Every image is cut into 256 tiles (:mod:`eviground.data.tiles`). Tiles of many images are put
into fixed-size batches (one queue per tile shape, so an image smaller than 256 batches with
others of its size) and run through a *predictor* that returns one output per head, e.g.
``{"model": {...}, "probe": {...}}``, each with ``logits`` (B, 3) and ``mask_logits``
(B, 1, h, w). When all tiles of an image are done, each head gets the image-level
probabilities (mean of P(real) and P(generated) over tiles, maximum of P(edited)), the label
and, if asked, the stitched probability map (mean where tiles overlap).

:func:`mask_record` turns a probability map and a ground-truth mask into the counts every pixel
metric is built from, so metrics can be pooled over any group of images afterwards.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from eviground.data import tiles as tiling
from eviground.models.eviground import aggregate_tiles, decide

AUC_BINS = 1000
BOUNDARY_TOLERANCE = 2  # pixels; ours: the DiffSeg30k paper reports bF1 without a tolerance
HELDOUT_VALUES = (2, 6)  # DiffSeg30k mask values of Kolors and Glide
# DiffSeg30k (arXiv 2511.19111v2, section 4.1): "The background class is excluded from mIoU and
# bF1 calculations", so its binary mIoU is the edited-class IoU (pooled over pixels here; the
# paper does not say how it averages, mmsegmentation's default accumulates over the dataset).
# Its image-level rule for a segmentation model (section 4.2): edited when the predicted edited
# area exceeds 1% of the image.
AREA_RULE = 0.01


@dataclass
class Item:
    """One whole image: ``image`` (H, W, 3) uint8; ``mask`` (H, W) bool or None; ``editors``
    (H, W) int per-pixel editor values (DiffSeg30k) or None; ``meta`` copied into records."""

    uid: str
    image: np.ndarray
    mask: np.ndarray | None = None
    editors: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


Predictor = Callable[[torch.Tensor], Mapping[str, Mapping[str, torch.Tensor]]]


class TiledEvaluator:
    def __init__(
        self,
        predict: Predictor,
        on_image: Callable[[Item, dict[str, dict[str, Any]]], None],
        *,
        batch: int = tiling.TILE_BATCH,
        device: torch.device | str = "cpu",
        want_maps: bool = True,
        edit_threshold: float = 0.5,
    ) -> None:
        self.predict, self.on_image = predict, on_image
        self.batch, self.device = batch, torch.device(device)
        self.want_maps, self.edit_threshold = want_maps, edit_threshold
        self.queues: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
        self.pending: dict[str, dict[str, Any]] = {}
        self.max_tiles = 0
        self.images = 0
        self.tiles = 0

    def add(self, item: Item) -> None:
        h, w = item.image.shape[:2]
        boxes = tiling.tile_boxes(h, w)
        self.max_tiles = max(self.max_tiles, len(boxes))
        self.pending[item.uid] = {"item": item, "boxes": boxes, "left": len(boxes), "out": {}}
        for i, (le, t, r, b) in enumerate(boxes):
            q = self.queues[(b - t, r - le)]
            q.append((item.uid, i))
            if len(q) >= self.batch:
                self._flush((b - t, r - le))

    def close(self) -> None:
        for shape in self.queues:  # flushing changes values only, never the keys
            while self.queues[shape]:
                self._flush(shape)

    def _flush(self, shape: tuple[int, int]) -> None:
        q = self.queues[shape]
        take, self.queues[shape] = q[: self.batch], q[self.batch :]
        if not take:
            return
        crops = []
        for uid, i in take:
            p = self.pending[uid]
            le, t, r, b = p["boxes"][i]
            # a writable copy: torch warns on read-only arrays (decoded images can be read-only)
            crops.append(torch.from_numpy(np.array(p["item"].image[t:b, le:r])))
        x = torch.stack(crops).to(self.device).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            outs = self.predict(x)
        self.tiles += len(take)
        for head, out in outs.items():
            probs = torch.softmax(out["logits"].float(), dim=-1).cpu()
            maps = torch.sigmoid(out["mask_logits"].float())[:, 0].cpu() if self.want_maps else None
            for j, (uid, i) in enumerate(take):
                acc = self.pending[uid]["out"].setdefault(head, {"probs": {}, "maps": {}})
                acc["probs"][i] = probs[j]
                if maps is not None:
                    acc["maps"][i] = maps[j]
        for uid, _i in take:
            p = self.pending[uid]
            p["left"] -= 1
            if p["left"] == 0:
                self._finish(uid)

    def _finish(self, uid: str) -> None:
        p = self.pending.pop(uid)
        item, boxes = p["item"], p["boxes"]
        h, w = item.image.shape[:2]
        results = {}
        for head, acc in p["out"].items():
            probs = torch.stack([acc["probs"][i] for i in range(len(boxes))])
            agg = aggregate_tiles(probs)
            res: dict[str, Any] = {
                "probs": agg,
                "label": decide(agg, self.edit_threshold),
                "tiles": len(boxes),
            }
            if self.want_maps:
                maps = torch.stack([acc["maps"][i] for i in range(len(boxes))])
                res["prob_map"] = tiling.stitch(maps, boxes, h, w).numpy()
                res["tile_probs"] = probs.numpy()  # per tile (real, generated, edited)
                res["boxes"] = list(boxes)
            results[head] = res
        self.images += 1
        self.on_image(item, results)


def _boundary(m: torch.Tensor) -> torch.Tensor:
    """Pixels of a binary mask (1, 1, H, W) with a 4-neighbour (up, down, left, right) outside
    the mask. The image border is not a boundary: the frame is padded with the pixel's own value,
    since an edit reaching the edge of the image ends there because the image does."""
    x = F.pad(m.float(), (1, 1, 1, 1), mode="replicate")
    c = x[..., 1:-1, 1:-1]
    neighbours = torch.stack(
        [x[..., :-2, 1:-1], x[..., 2:, 1:-1], x[..., 1:-1, :-2], x[..., 1:-1, 2:]]
    )
    return (c > 0) & (neighbours.amin(dim=0) < 1)


def boundary_counts(
    pred: np.ndarray, gt: np.ndarray, tolerance: int = BOUNDARY_TOLERANCE
) -> dict[str, int]:
    """Boundary pixels of the binary prediction and of the truth, and how many of each lie within
    ``tolerance`` pixels of the other's boundary (boundary F1)."""
    bp = _boundary(torch.from_numpy(np.asarray(pred, bool))[None, None])
    bg = _boundary(torch.from_numpy(np.asarray(gt, bool))[None, None])
    k = 2 * tolerance + 1
    near_g = F.max_pool2d(bg.float(), k, stride=1, padding=tolerance) > 0
    near_p = F.max_pool2d(bp.float(), k, stride=1, padding=tolerance) > 0
    return {
        "b_pred": int(bp.sum()),
        "b_gt": int(bg.sum()),
        "b_pred_hit": int((bp & near_g).sum()),
        "b_gt_hit": int((bg & near_p).sum()),
    }


def boundary_f1(c: Mapping[str, int]) -> float | None:
    """Boundary F1 of one image's :func:`boundary_counts` (None when either boundary is empty)."""
    bp = c["b_pred_hit"] / c["b_pred"] if c["b_pred"] else None
    br = c["b_gt_hit"] / c["b_gt"] if c["b_gt"] else None
    if bp is None or br is None:
        return None
    return 2 * bp * br / (bp + br) if bp + br else 0.0


def mask_record(
    prob: np.ndarray,
    gt: np.ndarray,
    *,
    threshold: float = 0.5,
    editors: np.ndarray | None = None,
    tolerance: int = BOUNDARY_TOLERANCE,
    bins: int = AUC_BINS,
) -> dict[str, Any]:
    """Pixel counts of one image: confusion (tp, fp, fn, tn), the score histograms of edited and
    unedited pixels (pooled AUC), boundary matches (boundary F1 at ``tolerance`` pixels) and,
    with ``editors``, the confusion on held-out-editor pixels (other edited pixels ignored)."""
    pred = prob >= threshold
    g = gt.astype(bool)
    rec: dict[str, Any] = {
        "pixels": int(g.size),
        "gt_pixels": int(g.sum()),
        "tp": int((pred & g).sum()),
        "fp": int((pred & ~g).sum()),
        "fn": int((~pred & g).sum()),
        "tn": int((~pred & ~g).sum()),
    }
    idx = np.minimum((prob * bins).astype(np.int64), bins - 1)
    rec["hist_pos"] = np.bincount(idx[g], minlength=bins)
    rec["hist_neg"] = np.bincount(idx[~g], minlength=bins)
    rec.update(boundary_counts(pred, g, tolerance))
    if editors is not None:
        held = np.isin(editors, HELDOUT_VALUES) & g
        keep = held | ~g
        rec.update(
            ho_tp=int((pred & held).sum()),
            ho_fn=int((~pred & held).sum()),
            ho_fp=int((pred & ~g & keep).sum()),
            ho_tn=int((~pred & ~g & keep).sum()),
        )
    return rec


def pooled(records: list[Mapping[str, Any]], prefix: str = "") -> dict[str, float | None]:
    """Pixel metrics pooled over ``records`` (``prefix`` 'ho_' for held-out-editor pixels)."""
    tp, fp, fn, tn = (sum(r[prefix + k] for r in records) for k in ("tp", "fp", "fn", "tn"))
    iou_e = tp / (tp + fp + fn) if tp + fp + fn else None
    iou_b = tn / (tn + fp + fn) if tn + fp + fn else None
    out: dict[str, float | None] = {
        "iou_edited": iou_e,
        "iou_background": iou_b,
        "miou_diffseg30k": iou_e,  # the paper's binary mIoU (background excluded); G2
        "miou_two_class": None if iou_e is None or iou_b is None else (iou_e + iou_b) / 2,
        "f1_pooled": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "pixel_acc": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else None,
        "fp_share_of_pixels": fp / (tp + fp + fn + tn) if tp + fp + fn + tn else None,
        "fn_share_of_edited": fn / (tp + fn) if tp + fn else None,
    }
    if not prefix:
        out["auc_pixel"] = hist_auc(
            sum(r["hist_pos"] for r in records), sum(r["hist_neg"] for r in records)
        )
    return out


def per_image(r: Mapping[str, Any]) -> dict[str, float | None]:
    tp, fp, fn = r["tp"], r["fp"], r["fn"]
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "boundary_f1": boundary_f1(r),
        "share": r["gt_pixels"] / r["pixels"],
        "pred_share": (r["tp"] + r["fp"]) / r["pixels"],
    }


def area_rule(pred_share: float, min_share: float = AREA_RULE) -> bool:
    """The DiffSeg30k image rule: edited when the predicted edited area exceeds 1% of the image."""
    return pred_share > min_share


def hist_auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """ROC AUC from score histograms (ties within a bin count one half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))
