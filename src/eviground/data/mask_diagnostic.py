"""What DiffSeg30k's mask values mean: a read-only diagnostic (14 Nov 2026).

The crop diagnostic found that about half of DiffSeg30k's edited training images have a mask
with no zero pixel (median edited share 100%, mostly 1024 x 1024 images). Spec section 8 assumes
the unedited background of AI-generated base images is labelled 0. This module measures, for
every DiffSeg30k mask in a CP1 manifest, which values cover the image and how much, so Dhanush can
decide how the masks are read:

- per mask: size class, the share of every value, whether no pixel is 0 ("full"), the dominant
  value and the second one;
- by role and size class: full vs partial masks, dominant values, shares;
- per value 1-8: how often it is the dominant (>= 50%) value vs a minority region;
- the Glide / Kolors hold-out: whether value 2 or 6 occurs only as the dominant (background-like)
  value or as an edit region;
- a what-if, counted only: a full mask's dominant value read as background (0). How many images
  then have a valid training crop, how many have no edit left, and how the hold-out roles change.

Nothing here changes training data, roles or the spec.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from eviground.data import crops, splits

DOMINANT = 0.5  # a value covering at least half of the image is "dominant"
PALETTE = {  # mask value -> colour for the example sheet
    0: (0, 0, 0),
    1: (230, 25, 75),
    2: (60, 180, 75),
    3: (255, 225, 25),
    4: (0, 130, 200),
    5: (245, 130, 48),
    6: (145, 30, 180),
    7: (70, 240, 240),
    8: (240, 50, 230),
}
EXAMPLE_SIZE = 192


def size_class(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "unknown"
    if (width, height) == (1024, 1024):
        return "1024x1024"
    return "both_sides_le_640" if max(width, height) <= 640 else "other"


def mask_facts(mask: np.ndarray) -> dict[str, Any]:
    """Value shares of one mask, its dominant and second value, and whether it has no zero."""
    m = np.asarray(mask)
    h, w = m.shape[:2]
    counts = np.bincount(m.ravel(), minlength=9)
    total = m.size
    shares = {int(v): counts[v] / total for v in np.flatnonzero(counts)}
    nonzero = sorted(((s, v) for v, s in shares.items() if v != 0), reverse=True)
    dominant = nonzero[0] if nonzero else (0.0, None)
    second = nonzero[1] if len(nonzero) > 1 else (0.0, None)
    return {
        "width": int(w),
        "height": int(h),
        "size_class": size_class(w, h),
        "shares": {v: round(float(s), 6) for v, s in sorted(shares.items())},
        "zero_share": round(float(shares.get(0, 0.0)), 6),
        "full": bool(shares.get(0, 0.0) == 0.0),
        "n_values": len(nonzero),
        "dominant": dominant[1],
        "dominant_share": round(float(dominant[0]), 6),
        "second": second[1],
        "second_share": round(float(second[0]), 6),
    }


def whatif_edited(mask: np.ndarray, facts: dict[str, Any]) -> np.ndarray:
    """Edited pixels if a full mask's dominant value is background (other masks unchanged)."""
    m = np.asarray(mask)
    if facts["full"] and facts["dominant"] is not None:
        return (m > 0) & (m != facts["dominant"])
    return m > 0


def _task(task: tuple[str, str, str, str, bool]) -> dict[str, Any]:
    uid, mask_path, dataset_split, role, check_crops = task
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L"))
    f = mask_facts(mask)
    values = {v for v in f["shares"] if v != 0}
    whatif_values = values - ({f["dominant"]} if f["full"] else set())
    out = {
        "uid": uid,
        "dataset_split": dataset_split,
        "role": role,
        **f,
        # the role as the mask values give it (before the leak filter), now and in the what-if
        "values_role": splits.diffseg_role(dataset_split, values),
        "whatif_role": splits.diffseg_role(dataset_split, whatif_values),
        "whatif_no_edit_left": f["full"] and not whatif_values,
    }
    if check_crops:
        out["valid_now"] = crops.has_valid_window(mask)
        out["valid_whatif"] = crops.has_valid_window(whatif_edited(mask, f).astype(np.uint8))
    return out


def _quantiles(v: Sequence[float]) -> dict[str, float] | None:
    if not len(v):
        return None
    q = np.percentile(np.asarray(v, dtype=float), [0, 10, 25, 50, 75, 90, 100])
    names = ("min", "p10", "p25", "median", "p75", "p90", "max")
    return {k: round(float(x), 4) for k, x in zip(names, q, strict=True)}


def _holdout_kind(r: dict[str, Any], value: int) -> str | None:
    if value not in r["shares"]:
        return None
    if r["full"] and r["dominant"] == value:
        return "dominant_background_of_full_mask"
    return "edit_region"


def summarize(facts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates of :func:`mask_facts` rows (with uid, role, dataset_split, what-if fields)."""
    by_role_size: Counter[str] = Counter()
    for r in facts:
        by_role_size[f"{r['role']} | {r['size_class']} | {'full' if r['full'] else 'partial'}"] += 1
    full = [r for r in facts if r["full"]]
    partial = [r for r in facts if not r["full"] and r["n_values"] > 0]
    per_value: dict[str, dict[str, int]] = {}
    for v, name in splits.DIFFSEG_EDITORS.items():
        dom = Counter()
        minority = Counter()
        for r in facts:
            s = r["shares"].get(v, 0.0)
            if s == 0:
                continue
            (dom if s >= DOMINANT else minority)[r["size_class"]] += 1
        per_value[f"{v} {name}"] = {
            **{f"dominant_{k}": n for k, n in sorted(dom.items())},
            **{f"minority_{k}": n for k, n in sorted(minority.items())},
        }
    holdout: dict[str, dict[str, int]] = {}
    for v in sorted(splits.HOLDOUT_VALUES):
        c: Counter[str] = Counter()
        for r in facts:
            kind = _holdout_kind(r, v)
            if kind:
                c[f"{r['role']} | {kind}"] += 1
        holdout[f"{v} {splits.DIFFSEG_EDITORS[v]}"] = dict(sorted(c.items()))
    role_change = Counter(
        f"{r['values_role']} -> {r['whatif_role']}"
        for r in facts
        if r["values_role"] != r["whatif_role"]
    )
    checked = [r for r in facts if "valid_now" in r]
    return {
        "masks": len(facts),
        "by_role_size_full": dict(sorted(by_role_size.items())),
        "full_masks": {
            "count": len(full),
            "size_class": dict(Counter(r["size_class"] for r in full)),
            "n_values": dict(sorted(Counter(min(r["n_values"], 3) for r in full).items())),
            "dominant_value": dict(sorted(Counter(r["dominant"] for r in full).items())),
            "dominant_share": _quantiles([r["dominant_share"] for r in full]),
            "second_share": _quantiles([r["second_share"] for r in full if r["second"]]),
        },
        "partial_masks": {
            "count": len(partial),
            "size_class": dict(Counter(r["size_class"] for r in partial)),
            "zero_share": _quantiles([r["zero_share"] for r in partial]),
            "n_values": dict(sorted(Counter(min(r["n_values"], 3) for r in partial).items())),
        },
        "empty_masks": sum(1 for r in facts if r["n_values"] == 0),
        "per_value_dominant_vs_minority": per_value,
        "holdout_values": holdout,
        "whatif_dominant_as_background": {
            "note": "counted only: a full mask's dominant value read as 0 (background)",
            "full_masks_with_no_edit_left": sum(1 for r in facts if r["whatif_no_edit_left"]),
            "role_changes": dict(sorted(role_change.items())),
            "train_images_checked": len(checked),
            "valid_crop_now": sum(1 for r in checked if r["valid_now"]),
            "valid_crop_whatif": sum(1 for r in checked if r["valid_whatif"]),
        },
    }


def _colour(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for v, c in PALETTE.items():
        rgb[mask == v] = c
    return Image.fromarray(rgb)


def examples_sheet(
    rows: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    out: Path,
    n_full: int = 8,
    n_partial: int = 4,
) -> dict[str, Any]:
    """Image | colour-coded mask pairs for a few full and partial masks (sorted by uid)."""
    by_uid = {r["uid"]: r for r in rows}
    ordered = sorted(facts, key=lambda r: r["uid"])
    picks = [r for r in ordered if r["full"]][:n_full]
    picks += [r for r in ordered if not r["full"] and r["n_values"]][:n_partial]
    s = EXAMPLE_SIZE
    sheet = Image.new("RGB", (2 * s, max(1, len(picks)) * (s + 14)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    listed = []
    for k, f in enumerate(picks):
        r = by_uid[f["uid"]]
        y = k * (s + 14)
        with Image.open(r["path"]) as im:
            sheet.paste(im.convert("RGB").resize((s, s)), (0, y + 14))
        with Image.open(r["mask_path"]) as m:
            sheet.paste(
                _colour(np.asarray(m.convert("L"))).resize((s, s), Image.NEAREST), (s, y + 14)
            )
        text = f"{f['uid']} {f['size_class']} " + " ".join(
            f"{v}:{sh:.0%}" for v, sh in f["shares"].items()
        )
        draw.text((2, y + 1), text[:60], fill=(0, 0, 0))
        listed.append({"uid": f["uid"], "full": f["full"], "shares": f["shares"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return {"file": out.name, "examples": listed, "palette": PALETTE}


def run_diagnostic(manifest: Path, *, out_dir: Path, workers: int = 1) -> dict[str, Any]:
    """The mask-label diagnostic on a CP1 manifest (DiffSeg30k masks on the local disk)."""
    from eviground.data.cp1 import _read_csv_gz

    rows = [r for r in _read_csv_gz(manifest) if r["source"] == "diffseg30k" and r.get("mask_path")]
    tasks = [
        (r["uid"], r["mask_path"], r["dataset_split"], r["role"], r["dataset_split"] == "train")
        for r in rows
    ]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            facts = list(pool.map(_task, tasks, chunksize=32))
    else:
        facts = [_task(t) for t in tasks]
    summary = summarize(facts)
    summary["examples"] = examples_sheet(rows, facts, out_dir / "mask_examples.png")
    summary["manifest"] = str(manifest)
    return summary
