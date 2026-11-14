"""Why edited crops fail: CP1 run 4's failed draws rebuilt from its manifest (14 Nov 2026).

CP1 run 4 drew 3,000 edited crops (DiffSeg30k) with the CP1 rule (:func:`crops.edited_box`, a
uniform window, 1-90% edited share, 10 tries); 1,585 failed. Before the 1% minimum is changed,
Dhanush asked for the edited-share histogram of those failures. The draws are deterministic, so
they are rebuilt exactly from ``manifest_cp1.csv.gz``: the same eligible rows, the same sampler
draw (seed 42) and the same per-draw seed (42 * 1,000,003 + k). The number of failures must
equal the recorded one, or the result says it was not reproduced.

Also reported, on the same draws and over every edited training image: how the training rule
(:func:`crops.edited_crop`) does with the 1% minimum and with a pixel minimum, and how many images
can never give a valid crop (checked over every window).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.data import crops, sampler

DRAW_SEED_STEP = 1_000_003  # cp1.run_cp1: draw seed = seed * 1_000_003 + k
PIXEL_MINIMUM = 256  # the alternative minimum to compare (edited pixels in the crop)
SHARE_BINS = (  # (label, low exclusive, high inclusive) on the edited share of one try
    ("0", -1.0, 0.0),
    ("(0, 0.1%]", 0.0, 0.001),
    ("(0.1%, 0.5%]", 0.001, 0.005),
    ("(0.5%, 1%)", 0.005, 0.01 - 1e-12),
    ("[1%, 90%]", 0.01 - 1e-12, 0.90),
    ("(90%, 99%]", 0.90, 0.99),
    ("(99%, 100%)", 0.99, 1.0 - 1e-12),
    ("100%", 1.0 - 1e-12, 1.0),
)


def trace_edited_box(
    mask: np.ndarray,
    rng: np.random.Generator,
    size: int = crops.CROP,
    share: tuple[float, float] = crops.EDITED_SHARE,
    tries: int = crops.TRIES,
) -> dict[str, Any]:
    """:func:`crops.edited_box` with every try recorded (same random numbers, same outcome)."""
    edited = np.asarray(mask) > 0
    h, w = edited.shape[:2]
    if w < size or h < size:
        return {"ok": False, "shares": [], "pixels": [], "too_small": True}
    ii = crops._integral(edited)
    lo, hi = share
    shares: list[float] = []
    pixels: list[int] = []
    for _ in range(tries):
        left, top, _, _ = crops.random_box(w, h, rng, size)  # type: ignore[misc]
        n = crops._box_sum(ii, left, top, size)
        shares.append(n / (size * size))
        pixels.append(n)
        if lo <= shares[-1] <= hi:
            return {"ok": True, "shares": shares, "pixels": pixels, "too_small": False}
    return {"ok": False, "shares": shares, "pixels": pixels, "too_small": False}


def eligible_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows ``run_cp1`` draws from, in the same order."""
    return [
        r for r in rows if r.get("split") == "train" and r.get("h_action") in ("keep", "compress")
    ]


def edited_draws(
    rows: Sequence[dict[str, Any]], crops_per_class: int, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    """``(k, row)`` of every edited draw of ``run_cp1`` (k = position among all draws)."""
    eligible = eligible_rows(rows)
    picks = sampler.draw(
        [r["source"] for r in eligible], [r["label"] for r in eligible], crops_per_class, seed
    )
    return [
        (k, eligible[int(i)]) for k, i in enumerate(picks) if eligible[int(i)]["label"] == "edited"
    ]


def load_mask(row: dict[str, Any]) -> np.ndarray:
    with Image.open(row["mask_path"]) as m:
        return np.asarray(m.convert("L"))


def _draw_task(task: tuple[int, str, int, int]) -> dict[str, Any]:
    k, mask_path, seed, min_pixels = task
    mask = load_mask({"mask_path": mask_path})
    draw_seed = seed * DRAW_SEED_STEP + k
    old = trace_edited_box(mask, np.random.default_rng(draw_seed))
    new = crops.edited_crop(mask, np.random.default_rng(draw_seed))[0] is not None
    new_px = (
        crops.edited_crop(mask, np.random.default_rng(draw_seed), min_pixels=min_pixels)[0]
        is not None
    )
    edited = int((mask > 0).sum())
    return {**old, "new_ok": new, "new_px_ok": new_px, "image_pixels": edited, "area": mask.size}


def _image_task(task: tuple[str, int]) -> dict[str, Any]:
    mask_path, min_pixels = task
    mask = load_mask({"mask_path": mask_path})
    counts = crops.window_counts(mask)
    area = crops.CROP * crops.CROP
    lo, hi = crops.EDITED_SHARE[0] * area, crops.EDITED_SHARE[1] * area
    ok = bool(counts.size) and bool(np.any((counts >= lo) & (counts <= hi)))
    ok_px = bool(counts.size) and bool(np.any((counts >= min_pixels) & (counts <= hi)))
    edited = int((mask > 0).sum())
    return {
        "valid": ok,
        "valid_px": ok_px,
        "pixels": edited,
        "share": edited / mask.size,
        "max_window": int(counts.max()) if counts.size else 0,
    }


def _bin(s: float) -> str:
    for name, lo, hi in SHARE_BINS:
        if lo < s <= hi:
            return name
    return "other"


def _quantiles(v: Sequence[float]) -> dict[str, float] | None:
    if not len(v):
        return None
    q = np.percentile(np.asarray(v, dtype=float), [0, 10, 25, 50, 75, 90, 100])
    return {
        k: round(float(x), 6)
        for k, x in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), q, strict=True)
    }


def _map(fn, tasks, workers):
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, tasks, chunksize=16))
    return [fn(t) for t in tasks]


def run_diagnostic(
    manifest: Path,
    *,
    crops_per_class: int = 3000,
    seed: int = 42,
    expected_failures: int | None = None,
    min_pixels: int = PIXEL_MINIMUM,
    workers: int = 1,
) -> dict[str, Any]:
    """The crop diagnostic on a CP1 manifest (masks must be on the local disk)."""
    from eviground.data.cp1 import _read_csv_gz

    rows = _read_csv_gz(manifest)
    draws = edited_draws(rows, crops_per_class, seed)
    traced = _map(_draw_task, [(k, r["mask_path"], seed, min_pixels) for k, r in draws], workers)
    failed = [t for t in traced if not t["ok"]]
    lo, hi = crops.EDITED_SHARE

    def kind(t: dict[str, Any]) -> str:
        if t["too_small"]:
            return "image_too_small"
        if all(s < lo for s in t["shares"]):
            return "all_tries_below_1pct"
        if all(s > hi for s in t["shares"]):
            return "all_tries_above_90pct"
        return "mixed_below_and_above"

    tries_hist = Counter(_bin(s) for t in failed for s in t["shares"])
    best_px = [max(t["pixels"]) for t in failed if t["pixels"] and all(s < lo for s in t["shares"])]
    pool = [r for r in eligible_rows(rows) if r["label"] == "edited"]
    images = _map(_image_task, [(r["mask_path"], min_pixels) for r in pool], workers)
    n_fail = len(failed)
    return {
        "manifest": str(manifest),
        "rule_cp1": "uniform 256 window, edited share 1-90%, 10 tries (crops.edited_box)",
        "draws": {
            "edited_draws": len(traced),
            "failed": n_fail,
            "expected_failed": expected_failures,
            "reproduced": None if expected_failures is None else n_fail == expected_failures,
            "failed_by_kind": dict(sorted(Counter(kind(t) for t in failed).items())),
            "failed_tries_share_histogram": {
                name: tries_hist.get(name, 0) for name, *_ in SHARE_BINS
            },
            "failed_below_1pct_best_try_edited_pixels": _quantiles(best_px),
            "failed_below_1pct_best_try_ge_min_pixels": sum(1 for p in best_px if p >= min_pixels),
            "failed_image_edited_share": _quantiles(
                [t["image_pixels"] / t["area"] for t in failed]
            ),
            "training_rule_failed_1pct": sum(1 for t in traced if not t["new_ok"]),
            f"training_rule_failed_min_{min_pixels}px": sum(
                1 for t in traced if not t["new_px_ok"]
            ),
        },
        "images": {
            "edited_training_images": len(images),
            "never_valid_1_90pct": sum(1 for i in images if not i["valid"]),
            f"never_valid_min_{min_pixels}px_le_90pct": sum(1 for i in images if not i["valid_px"]),
            "edited_pixels": _quantiles([i["pixels"] for i in images]),
            "edited_share": _quantiles([i["share"] for i in images]),
            "max_window_edited_pixels": _quantiles([i["max_window"] for i in images]),
            "below_1pct_of_a_crop_in_total": sum(
                1 for i in images if i["pixels"] < lo * crops.CROP * crops.CROP
            ),
        },
        "min_pixels_compared": min_pixels,
    }
