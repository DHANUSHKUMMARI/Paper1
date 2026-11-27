"""CP8 Claims-Edit case study: the generation plan, frozen by hash before any image is made
(spec P7 / CP8 rows; Dhanush, 27 Nov 2026).

The 150 COCO val2017 photos reserved by CP1 (``cp1.reserve_claims_edit``: car, couch, chair,
dining table, laptop, with an instance covering >= 5% of the image) each get one planned edit:

- **mask:** the largest non-crowd instance of those categories whose area is 2-40% of the image
  (ties broken by a seeded hash); a photo with none is dropped and counted;
- **damage:** drawn by a seeded hash from :data:`DAMAGES` (``broken glass`` only for a car or a
  laptop); prompt ``"realistic {damage} on the {category}, photo"``, a fixed negative prompt;
- **seed, steps, guidance, dilation:** fixed or seeded per photo.

The edited image is the original outside the dilated mask and the generated pixels inside it
(:func:`composite`: a hard seam at the mask border, a known property stated in the spec card).
A pair is dropped if generation fails or the edit is not visible (:func:`visible_change`).
Ground truth for scoring: the photo is ``real``; its edited copy is ``edited`` with the dilated
mask (:func:`claims`).
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CATEGORIES = ("car", "couch", "chair", "dining table", "laptop")
DAMAGES = ("cracks", "dents", "stains", "broken glass")
GLASS_OK = ("car", "laptop")
MIN_SHARE, MAX_SHARE = 0.02, 0.40
DILATE_PX = 8
STEPS, GUIDANCE = 25, 6.0
NEGATIVE = "blurry, cartoon, drawing, text, watermark, distorted, low quality"
MIN_VISIBLE = 10.0  # mean |change| inside the mask, on the 0-255 scale
SALT = "claims_edit_v1"
SETTINGS = {"min_share": MIN_SHARE, "max_share": MAX_SHARE, "dilate_px": DILATE_PX,
            "steps": STEPS, "guidance": GUIDANCE, "negative": NEGATIVE,
            "min_visible": MIN_VISIBLE}  # fmt: skip


def _h(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


def image_id(uid: str) -> int:
    """COCO image id from a CP1 uid such as ``coco/val2017/000000000139.jpg``."""
    return int(Path(uid).stem)


def instance_mask(ann: Mapping[str, Any], width: int, height: int) -> np.ndarray:
    """(H, W) bool mask of a COCO annotation: its polygons, or its box if it has none."""
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    seg = ann.get("segmentation")
    if isinstance(seg, list) and seg:
        for poly in seg:
            if len(poly) >= 6:
                draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)], fill=1)
    else:
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=1)
    return np.asarray(img, dtype=bool)


def plan_edits(
    rows: Iterable[Mapping[str, Any]],
    instances: Mapping[str, Any],
    *,
    seed: int = 42,
    salt: str = SALT,
) -> dict[str, Any]:
    """One planned edit per reserved photo (``rows``: CP1's ``test_P7`` rows), and the photos
    dropped with their reason. Deterministic in ``seed`` and ``salt``."""
    names = {c["id"]: c["name"] for c in instances["categories"]}
    size = {im["id"]: (im["width"], im["height"]) for im in instances["images"]}
    by_image: dict[int, list[Mapping[str, Any]]] = {}
    for a in instances["annotations"]:
        if names.get(a["category_id"]) in CATEGORIES and not a.get("iscrowd"):
            by_image.setdefault(a["image_id"], []).append(a)
    plans, dropped = [], []
    for r in sorted(rows, key=lambda r: str(r["uid"])):
        uid = str(r["uid"])
        iid = image_id(uid)
        w, h = size[iid]
        ok = [a for a in by_image.get(iid, []) if MIN_SHARE <= a["area"] / (w * h) <= MAX_SHARE]
        if not ok:
            dropped.append({"uid": uid, "reason": "no instance of the categories with 2-40% area"})
            continue
        top = max(a["area"] for a in ok)
        tied = sorted((a for a in ok if a["area"] == top), key=lambda a: a["id"])
        ann = tied[_h(seed, salt, uid, "tie") % len(tied)]
        cat = names[ann["category_id"]]
        damages = [d for d in DAMAGES if d != "broken glass" or cat in GLASS_OK]
        damage = damages[_h(seed, salt, uid, "damage") % len(damages)]
        plans.append(
            {
                "uid": uid,
                "image_id": iid,
                "ann_id": int(ann["id"]),
                "category": cat,
                "area_share": round(ann["area"] / (w * h), 6),
                "width": w,
                "height": h,
                "damage": damage,
                "prompt": f"realistic {damage} on the {cat}, photo",
                "gen_seed": _h(seed, salt, uid, "gen") % (2**31),
            }
        )
    return {"seed": seed, "salt": salt, "settings": dict(SETTINGS), "plans": plans,
            "dropped": dropped}  # fmt: skip


def plan_sha256(plan: Mapping[str, Any]) -> str:
    keep = {k: plan[k] for k in ("seed", "salt", "settings", "plans", "dropped")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def freeze(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The plan with its SHA-256 and counts, ready to store (``eval_subsets.json``, key
    ``CP8_plan``) before any image is generated."""
    return {**plan, "images": len(plan["plans"]), "dropped_count": len(plan["dropped"]),
            "sha256": plan_sha256(plan)}  # fmt: skip


def intact(frozen: Mapping[str, Any]) -> bool:
    return plan_sha256(frozen) == frozen.get("sha256")


# The files that decide a generated pair's pixels (Dhanush, 27 Nov): only these enter the
# fingerprint, so an unrelated merge (e.g. CP9 code) never blocks a resume. The Kolors pins live in
# inpaint.py and in the inpainter's name; the plan settings in the plan hash.
GEN_CODE_PATHS = (
    "src/eviground/data/claims_edit.py",
    "src/eviground/data/inpaint.py",
    "notebooks/19_cp8_generation.ipynb",
)


def generation_fingerprint(inpainter: str, plan_sha: str, code: Mapping[str, str]) -> str:
    """What decides a generated pair beyond the plan: the inpainter with its pins, the frozen plan
    and the git hashes of ``GEN_CODE_PATHS`` (``phasec.code_tree(repo, GEN_CODE_PATHS)``). A
    resumed pair is reused only under the same fingerprint (Codex review, PR #65)."""
    key = {"inpainter": inpainter, "plan_sha256": plan_sha, "code": dict(sorted(code.items()))}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def check_resume(records: Iterable[Mapping[str, Any]], fingerprint: str) -> None:
    """Refuse to reuse records made under another generation fingerprint (a record without
    one counts as different), so the tar never mixes generations."""
    other = sorted(r["uid"] for r in records if r.get("fingerprint") != fingerprint)
    if other:
        raise ValueError(
            f"{len(other)} saved pairs were made under another generation configuration "
            f"(e.g. {other[0]}): use a new output folder or delete them; nothing is mixed"
        )


def dilate(mask: np.ndarray, px: int = DILATE_PX) -> np.ndarray:
    return ndimage.binary_dilation(np.asarray(mask, bool), iterations=int(px)) if px else mask


def composite(original: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The original outside ``mask``, ``generated`` inside it (same shape; a hard seam)."""
    if original.shape != generated.shape or original.shape[:2] != mask.shape:
        raise ValueError(f"shapes differ: {original.shape}, {generated.shape}, {mask.shape}")
    out = np.array(original, copy=True)
    out[mask] = generated[mask]
    return out


def visible_change(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> float:
    """Mean |edited - original| inside the mask (0-255 scale, over the 3 channels)."""
    if not mask.any():
        return 0.0
    d = np.abs(edited.astype(np.int16) - original.astype(np.int16))
    return float(d[mask].mean())


def keep_pair(
    original: np.ndarray, edited: np.ndarray | None, mask: np.ndarray
) -> tuple[bool, str]:
    """The drop rule: a pair is kept only if it was generated and the edit is visible."""
    if edited is None:
        return False, "generation failed"
    change = visible_change(original, edited, mask)
    if change < MIN_VISIBLE:
        return False, f"edit not visible (mean change {change:.2f} < {MIN_VISIBLE})"
    return True, "kept"


def boundary_band(mask: np.ndarray, px: int = 8) -> np.ndarray:
    """The thin band of ``px`` iterations (cross structuring element) on each side of the mask
    border (the composite seam). ``border_value=1``: the image edge is not a mask edge, so a mask
    touching it gets no fake seam strip there (review of PR #65)."""
    m = np.asarray(mask, bool)
    grow = ndimage.binary_dilation(m, iterations=px)
    shrink = ndimage.binary_erosion(m, iterations=px, border_value=1)
    return grow & ~shrink


def consecutive_failures(records: Sequence[Mapping[str, Any]]) -> int:
    """How many of the latest pair records in a row failed to generate (notebook 19 stops at
    ``MAX_FAILS``; review of PR #65)."""
    n = 0
    for r in reversed(records):
        if r.get("reason") != "generation failed":
            break
        n += 1
    return n


def band_iou(pred: np.ndarray, gt: np.ndarray, px: int = 8) -> float | None:
    """IoU of the predicted mask with the boundary band of the ground-truth mask: a diagnostic
    of how much the detector keys on the seam (Dhanush, 27 Nov), reported beside pixel F1."""
    band = boundary_band(gt, px)
    p = np.asarray(pred, bool)
    union = (p | band).sum()
    return None if union == 0 else float((p & band).sum() / union)


def eye_check_sample(plan: Mapping[str, Any], n: int = 30, seed: int = 42) -> list[str]:
    """The ``n`` pairs Dhanush checks by eye, drawn by a seeded hash (sorted uids)."""
    uids = sorted(p["uid"] for p in plan["plans"])
    return sorted(heapq.nsmallest(n, uids, key=lambda u: _h(seed, "eye_check", u)))


def claims(plan_row: Mapping[str, Any], edited_mask: np.ndarray) -> dict[str, Any]:
    """The ground truth of one pair for scoring (HR-GT, pixel F1): the real photo and its
    edited copy with the saved (dilated) mask."""
    return {
        "real": {"uid": plan_row["uid"], "label": "real", "mask": None},
        "edited": {
            "uid": edited_uid(plan_row["uid"]),
            "label": "edited",
            "mask": edited_mask,
            "category": plan_row["category"],
            "damage": plan_row["damage"],
        },  # fmt: skip
    }


def edited_uid(uid: str) -> str:
    return f"claims_edit/{Path(uid).stem}_edited.png"


def selected(rows: Sequence[Mapping[str, Any]], uids: Iterable[str]) -> list[Mapping[str, Any]]:
    keep = set(uids)
    return [r for r in rows if str(r["uid"]) in keep]
