"""Localization diagnosis on validation data (Dhanush, 16 Nov 2026): before P2 is scored.

From per-image records (:func:`eviground.eval.tiled.mask_record` plus the image's metadata) of
the training pool's validation images (training editors only; no test image):

- edited-class and background IoU, pooled and per image, and the split of the pixel error into
  over-marking (false positives) and missing (false negatives);
- the same by edited share of the image, by editor and by base type;
- resolution ceilings from the ground truth alone: each tile's true mask reduced to a coarse grid
  (majority per cell; 16 x 16 = the CLIP patch grid, 64 x 64 = the decoder's 1/4 grid), stitched
  back and compared with the true mask: the best a predictor at that grid can reach;
- the false "edited" rate and the marks on real images by tile count (P(edited) is a maximum
  over up to 16 tiles now).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from eviground.data import tiles as tiling
from eviground.eval.tiled import area_rule, mask_record, per_image, pooled

SIZE_BINS = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.40), (0.40, 1.0001))
TILE_BINS = ((1, 1), (2, 4), (5, 9), (10, 16), (17, 10**6))
GRIDS = (16, 64)


def size_bin(share: float) -> str:
    for lo, hi in SIZE_BINS:
        if lo <= share < hi:
            return f"{lo:.0%}-{min(hi, 1):.0%}"
    return "none"


def tile_bin(n: int) -> str:
    for lo, hi in TILE_BINS:
        if not lo <= n <= hi:
            continue
        if lo == hi:
            return f"{lo}"
        return f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
    return "?"


def grid_ceiling_map(gt: np.ndarray, grid: int) -> np.ndarray:
    """The true mask as a predictor limited to a ``grid`` x ``grid`` cell grid per tile would
    best give it: majority per cell, back to pixels, stitched with the test-time rule."""
    h, w = gt.shape
    boxes = tiling.tile_boxes(h, w)
    g = torch.from_numpy(gt.astype(np.float32))
    maps = []
    for le, t, r, b in boxes:
        tile = g[t:b, le:r][None, None]
        cells = (F.adaptive_avg_pool2d(tile, (grid, grid)) >= 0.5).float()
        maps.append(F.interpolate(cells, size=tile.shape[-2:], mode="nearest")[0, 0])
    return tiling.stitch(torch.stack(maps), boxes, h, w).numpy()


INTERIOR = 0.9  # a tile at least this share edited lies inside an edit (Dhanush, 17 Nov 2026)
GENERATED = 1  # class index of "fully generated" in (real, generated, edited)


def interior_record(
    prob_map: np.ndarray,
    gt: np.ndarray,
    boxes: Sequence[Sequence[int]],
    tile_probs: np.ndarray,
    threshold: float = 0.5,
    interior: float = INTERIOR,
) -> dict[str, Any]:
    """Tiles inside an edit (true edited share >= ``interior``): how many, how many the tile's
    class head calls fully generated, and their summed P(generated); and the missed edited pixels
    split into interior ones (covered by such a tile) and the rest."""
    gt = np.asarray(gt, bool)
    inside = np.zeros(gt.shape, bool)
    n = called = 0
    p_gen = 0.0
    for (left, top, right, bottom), probs in zip(boxes, np.asarray(tile_probs), strict=True):
        if gt[top:bottom, left:right].mean() >= interior:
            inside[top:bottom, left:right] = True
            n += 1
            called += int(int(np.argmax(probs)) == GENERATED)
            p_gen += float(probs[GENERATED])
    missed = gt & (np.asarray(prob_map) < threshold)
    return {
        "interior_tiles": n,
        "interior_called_generated": called,
        "interior_p_generated_sum": p_gen,
        "edited_interior": int((gt & inside).sum()),
        "edited_rest": int((gt & ~inside).sum()),
        "missed_interior": int((missed & inside).sum()),
        "missed_rest": int((missed & ~inside).sum()),
    }


def interior_summary(records: Sequence[Mapping[str, Any]], head: str) -> dict[str, Any]:
    """Pooled :func:`interior_record` counts of ``head`` over edited-image records."""
    rs = [r["interior"][head] for r in records if (r.get("interior") or {}).get(head)]
    tot = {k: sum(r[k] for r in rs) for k in rs[0]} if rs else {}
    share = lambda a, b: tot[a] / tot[b] if tot.get(b) else None  # noqa: E731
    return {
        "images": len(rs),
        "interior_tiles": tot.get("interior_tiles", 0),
        "interior_called_generated": share("interior_called_generated", "interior_tiles"),
        "interior_mean_p_generated": share("interior_p_generated_sum", "interior_tiles"),
        "miss_share_interior": share("missed_interior", "edited_interior"),
        "miss_share_rest": share("missed_rest", "edited_rest"),
        "edited_pixels_interior_share": (
            tot["edited_interior"] / (tot["edited_interior"] + tot["edited_rest"])
            if tot and tot["edited_interior"] + tot["edited_rest"]
            else None
        ),
    }


def ceiling_records(gt: np.ndarray, grids: Sequence[int] = GRIDS) -> dict[int, dict[str, Any]]:
    return {g: mask_record(grid_ceiling_map(gt, g), gt) for g in grids}


def _group(records: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Any]) -> dict:
    out: dict[str, list] = defaultdict(list)
    for r in records:
        k = key(r)
        for kk in k if isinstance(k, (list, tuple, set)) else [k]:
            out[str(kk)].append(r)
    return dict(sorted(out.items()))


def _summary(recs: Sequence[Mapping[str, Any]], head: str) -> dict[str, Any]:
    counts = [r["heads"][head] for r in recs]
    imgs = [per_image(c) for c in counts]
    vals = lambda k: [i[k] for i in imgs if i[k] is not None]  # noqa: E731
    return {
        "images": len(recs),
        **pooled(counts),
        "mean_iou_per_image": float(np.mean(vals("iou"))) if vals("iou") else None,
        "mean_f1_per_image": float(np.mean(vals("f1"))) if vals("f1") else None,
        "mean_boundary_f1": float(np.mean(vals("boundary_f1"))) if vals("boundary_f1") else None,
        "called_edited_area_rule": float(np.mean([area_rule(i["pred_share"]) for i in imgs]))
        if imgs
        else None,
    }


def summarize_edited(records: Sequence[Mapping[str, Any]], heads: Sequence[str]) -> dict:
    """Records of edited images: ``share``, ``editors`` (list), ``base_type``, ``heads``
    ({head: mask_record}) and ``ceilings`` ({grid: mask_record})."""
    groups = {
        "by_size": _group(records, lambda r: size_bin(r["share"])),
        "by_editor": _group(records, lambda r: r.get("editors") or ["unknown"]),
        "by_base_type": _group(records, lambda r: r.get("base_type") or "unknown"),
    }
    out: dict[str, Any] = {"images": len(records)}
    for head in heads:
        out[head] = {
            "all": _summary(records, head),
            **{g: {k: _summary(v, head) for k, v in grp.items()} for g, grp in groups.items()},
        }
        if any(r.get("interior") for r in records):
            out[head]["interior"] = {
                "all": interior_summary(records, head),
                "by_size": {k: interior_summary(v, head) for k, v in groups["by_size"].items()},
            }
    if records and records[0].get("ceilings"):
        out["ceilings"] = {}
        for grid in records[0]["ceilings"]:
            out["ceilings"][f"grid_{grid}"] = {
                "all": pooled([r["ceilings"][grid] for r in records]),
                "by_size": {
                    k: pooled([r["ceilings"][grid] for r in v])
                    for k, v in groups["by_size"].items()
                },
            }
    return out


def summarize_real(records: Sequence[Mapping[str, Any]], heads: Sequence[str]) -> dict:
    """Records of real images: ``tiles``, ``source``, ``label_pred`` and ``marked`` share per
    head (``heads``: {head: {"label": ..., "marked_share": ...}})."""
    out: dict[str, Any] = {"images": len(records)}
    by_tiles = _group(records, lambda r: tile_bin(r["tiles"]))
    by_source = _group(records, lambda r: r.get("source") or "unknown")
    for head in heads:
        out[head] = {
            "all": _real_rates(records, head) if records else None,
            "by_tiles": {k: _real_rates(v, head) for k, v in by_tiles.items()},
            "by_source": {k: _real_rates(v, head) for k, v in by_source.items()},
        }
    return out


def _real_rates(rs: Sequence[Mapping[str, Any]], head: str) -> dict[str, Any]:
    n = len(rs)
    shares = [r["heads"][head]["marked_share"] for r in rs]
    return {
        "images": n,
        "called_edited": sum(r["heads"][head]["label"] == "edited" for r in rs) / n,
        "any_marked": sum(s > 0 for s in shares) / n,
        "called_edited_area_rule": sum(area_rule(s) for s in shares) / n,
        "marked_area": float(np.mean(shares)),
    }
