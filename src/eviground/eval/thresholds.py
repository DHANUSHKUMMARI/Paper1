"""Validation-only threshold sweep for the localization diagnosis (Dhanush, 16 Nov 2026).

Diagnostic only: every headline number and the CP4 gate stay at the fixed 0.5 thresholds. The
sweep shows how far each head is from its best threshold on validation data (training editors,
no test image); whether a validation-chosen threshold is used anywhere later is decided after
the diagnosis.

- Pixel threshold t = k / bins: the confusion counts come exactly from the per-image score
  histograms :func:`eviground.eval.tiled.mask_record` already keeps (a pixel is "edited at t"
  when its histogram bin is >= k), so no second model pass is needed.
- Boundary F1 needs the binary map, so it is measured at a coarse grid when the image is
  evaluated (:data:`BOUNDARY_GRID`, :func:`boundary_grid`).
- Image threshold tau on P(edited) (a maximum over tiles): the share of edited validation images
  called edited, and the false "edited" rate on real validation images, overall and by tile count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from eviground.eval.tiled import AUC_BINS, boundary_counts, boundary_f1

PIXEL_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 ... 0.95
BOUNDARY_GRID = tuple(round(0.1 * i, 1) for i in range(1, 10))  # 0.1 ... 0.9
IMAGE_GRID = PIXEL_GRID


def _bin(t: float, bins: int) -> int:
    return int(round(t * bins))


def confusion_at(hist_pos: np.ndarray, hist_neg: np.ndarray, k: int) -> dict[str, int]:
    """(tp, fp, fn, tn) when every pixel in histogram bin >= ``k`` is called edited."""
    pos, neg = np.asarray(hist_pos), np.asarray(hist_neg)
    return {
        "tp": int(pos[k:].sum()),
        "fn": int(pos[:k].sum()),
        "fp": int(neg[k:].sum()),
        "tn": int(neg[:k].sum()),
    }


def _metrics(c: Mapping[str, int]) -> dict[str, float | None]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    iou_e = tp / (tp + fp + fn) if tp + fp + fn else None
    iou_b = tn / (tn + fp + fn) if tn + fp + fn else None
    n = tp + fp + fn + tn
    return {
        "iou_edited": iou_e,
        "iou_background": iou_b,
        "miou_diffseg30k": iou_e,
        "miou_two_class": None if iou_e is None or iou_b is None else (iou_e + iou_b) / 2,
        "f1_pooled": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "fp_share_of_pixels": fp / n if n else None,
        "fn_share_of_edited": fn / (tp + fn) if tp + fn else None,
    }


def _mean(values: Sequence[float | None]) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def pixel_sweep(
    records: Sequence[Mapping[str, Any]],
    head: str,
    grid: Sequence[float] = PIXEL_GRID,
    bins: int = AUC_BINS,
) -> list[dict[str, Any]]:
    """Pooled pixel metrics and the per-image mean F1 of ``head`` at each threshold of ``grid``
    (edited-image records with ``heads[head]`` a :func:`mask_record`). Boundary F1 is added at
    the thresholds the records carry in ``boundary_grid[head]``."""
    counts = [r["heads"][head] for r in records]
    pos = [np.asarray(c["hist_pos"]) for c in counts]
    neg = [np.asarray(c["hist_neg"]) for c in counts]
    out = []
    for t in grid:
        k = _bin(t, bins)
        per = [confusion_at(p, n, k) for p, n in zip(pos, neg, strict=True)]
        total = {key: sum(c[key] for c in per) for key in ("tp", "fp", "fn", "tn")}
        f1s = [
            2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] + c["fp"] + c["fn"] else None
            for c in per
        ]
        row: dict[str, Any] = {"threshold": t, **_metrics(total), "mean_f1_per_image": _mean(f1s)}
        bf = [_grid_bf1(r, head, t) for r in records]
        if any(b is not _MISSING for b in bf):
            row["mean_boundary_f1"] = _mean([b for b in bf if b is not _MISSING])
        out.append(row)
    return out


_MISSING = object()


def _grid_bf1(r: Mapping[str, Any], head: str, t: float) -> Any:
    grid = (r.get("boundary_grid") or {}).get(head) or {}
    c = grid.get(t, grid.get(str(t)))
    return _MISSING if c is None else boundary_f1(c)


def boundary_grid(
    prob: np.ndarray, gt: np.ndarray, grid: Sequence[float] = BOUNDARY_GRID
) -> dict[float, dict[str, int]]:
    """Boundary counts of one image at each threshold of ``grid``."""
    return {t: boundary_counts(prob >= t, gt) for t in grid}


def image_sweep(
    edited: Sequence[Mapping[str, Any]],
    real: Sequence[Mapping[str, Any]],
    head: str,
    grid: Sequence[float] = IMAGE_GRID,
    tile_bin: Any = None,
) -> list[dict[str, Any]]:
    """At each tau: the share of edited images with P(edited) >= tau, and the share of real
    images with P(edited) >= tau, overall and by tile-count bin (``tile_bin``: tiles -> label).
    Records carry ``p_edited[head]`` and ``tiles``."""
    pe = np.array([r["p_edited"][head] for r in edited], float)
    pr = np.array([r["p_edited"][head] for r in real], float)
    bins: dict[str, np.ndarray] = {}
    if tile_bin is not None and len(real):
        labels = np.array([tile_bin(int(r["tiles"])) for r in real])
        bins = {b: labels == b for b in sorted(set(labels.tolist()))}
    out = []
    for t in grid:
        row: dict[str, Any] = {
            "threshold": t,
            "edited_called_edited": float((pe >= t).mean()) if len(pe) else None,
            "real_called_edited": float((pr >= t).mean()) if len(pr) else None,
        }
        if bins:
            row["real_called_edited_by_tiles"] = {
                b: float((pr[m] >= t).mean()) for b, m in bins.items()
            }
        out.append(row)
    return out


def best(
    sweep: Sequence[Mapping[str, Any]], key: str = "miou_diffseg30k", at: float = 0.5
) -> dict[str, Any]:
    """The threshold of ``sweep`` with the largest ``key``, its value, and the value at ``at``."""
    rows = [r for r in sweep if r.get(key) is not None]
    if not rows:
        return {"key": key, "threshold": None, "value": None, "at_default": None}
    top = max(rows, key=lambda r: r[key])
    ref = next((r[key] for r in sweep if abs(r["threshold"] - at) < 1e-9), None)
    return {
        "key": key,
        "threshold": top["threshold"],
        "value": top[key],
        "default_threshold": at,
        "at_default": ref,
    }


def diagnosis_sweeps(
    edited: Sequence[Mapping[str, Any]],
    real: Sequence[Mapping[str, Any]],
    heads: Sequence[str],
    tile_bin: Any = None,
) -> dict[str, Any]:
    """The ``threshold_sweep`` block of the localization diagnosis, one entry per head."""
    out: dict[str, Any] = {
        "note": "validation only, diagnostic; headline numbers and the gate stay at 0.5",
        "pixel_grid": list(PIXEL_GRID),
        "boundary_grid": list(BOUNDARY_GRID),
        "image_grid": list(IMAGE_GRID),
    }
    for head in heads:
        px = pixel_sweep(edited, head) if edited else []
        img = image_sweep(edited, real, head, tile_bin=tile_bin)
        out[head] = {
            "pixel": px,
            "image": img,
            "best": {
                "miou_diffseg30k": best(px, "miou_diffseg30k"),
                "miou_two_class": best(px, "miou_two_class"),
                "f1_pooled": best(px, "f1_pooled"),
                "mean_f1_per_image": best(px, "mean_f1_per_image"),
                "mean_boundary_f1": best(px, "mean_boundary_f1"),
            },
        }
    return out
