"""A5 explanation variants (spec section 9, ablation 5; Dhanush, 24 Nov 2026).

1. **Grounded LLM with the verifier**: :func:`verifier.run_grounded`, final reports.
2. **No verifier**: the same run's first pass (stored next to the final report).
3. **Fixed template, no LLM**: :func:`template_report`, deterministic from the evidence record.
4. **LLM shown the image**: Qwen2-VL-7B-Instruct (AWQ) on P6_mllm, same report schema with a pixel
   box per region claim (:func:`image_reports`). It gets no evidence record, so GR / HR-E are
   not defined for it; HR-GT and verdict accuracy are the cross-system comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eviground.explain import prompts, verifier


def template_report(
    record: Mapping[str, Any], *, min_mask_conf: float | None = verifier.TAU_E
) -> dict[str, Any]:
    """A report written by fixed rules from the evidence record. Every number is copied from
    the cited entry, and an E region is described only if its ``mask_conf`` >= ``min_mask_conf``
    (rule (c), the same rule the LLM must obey), so every claim passes the verifier (tested).

    ``min_mask_conf=None`` reproduces the template as run in Phase D (notebook 18, 25 Nov), which
    described every E region and so broke rule (c) on low-confidence regions; it is kept only to
    report the as-run numbers beside the fixed ones."""
    pred, top = record["prediction"], record["verdict"]
    g1 = next(g for g in record["global"] if g["id"] == "G1")
    claims: list[dict[str, Any]] = [
        {
            "claim_type": "verdict",
            "text": f"The detector gives this verdict a probability of {pred[top]}.",
            "evidence_ids": [prompts.PRED_ID],
        },
        {
            "claim_type": "global",
            "text": f"The decision relies on the semantic branch with weight {g1['semantic']} "
            f"and on the trace branch with weight {g1['trace']}.",
            "evidence_ids": ["G1"],
        },
    ]
    for reg in record.get("regions", []):
        if reg["id"].startswith("E"):
            if min_mask_conf is not None and float(reg.get("mask_conf", 0.0)) < min_mask_conf:
                continue
            claims.append(
                {
                    "claim_type": "edited_region",
                    "text": f"Region {reg['id']} is marked as edited; it covers {reg['area_pct']}% "
                    f"of the image with mean mask confidence {reg['mask_conf']}.",
                    "evidence_ids": [reg["id"]],
                }
            )
        else:
            claims.append(
                {
                    "claim_type": "attribution",
                    "text": f"The model's evidence concentrates in {reg['id']}, "
                    f"which holds {reg['attr_mass']} of the attribution mass.",
                    "evidence_ids": [reg["id"]],
                }
            )
    return {
        "verdict": top,
        "confidence": pred[top],
        "claims": claims,
        "limitations": "Written from the detector's evidence record by fixed rules.",
    }


def grid_box_to_pixels(box: Sequence[float], width: int, height: int) -> list[float]:
    """A Qwen2-VL box (x1, y1, x2, y2) on the 0-1000 grid as a pixel box [x, y, w, h] on an
    image of ``width`` x ``height`` (the corners may be given in either order)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    sx, sy = width / prompts.BOX_GRID, height / prompts.BOX_GRID
    return [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]


def _box_status(box: Any) -> str:
    """'ok', 'partly_off' or 'fully_off' (relative to the 0-1000 grid), 'missing' (no box) or
    'unparseable'."""
    if box is None:
        return "missing"
    if not isinstance(box, list | tuple) or len(box) != 4:
        return "unparseable"
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in box):
        return "unparseable"
    lo, hi = 0, prompts.BOX_GRID
    x1, x2 = sorted((box[0], box[2]))
    y1, y2 = sorted((box[1], box[3]))
    if x2 <= lo or x1 >= hi or y2 <= lo or y1 >= hi:
        return "fully_off"
    if x1 < lo or y1 < lo or x2 > hi or y2 > hi:
        return "partly_off"
    return "ok"


def to_pixel_boxes(report: Mapping[str, Any] | None, width: int, height: int):
    """The image-baseline report with each region box converted to pixel [x, y, w, h], and the
    box counts by status. An unparseable box becomes an empty box [0, 0, 0, 0], so its claim
    still counts as a wrong region in HR-GT (it is not dropped). An edited-region claim with no
    box keeps ``box`` None; HR-GT counts it as wrong and as a claim without a region."""
    counts = {"ok": 0, "partly_off": 0, "fully_off": 0, "missing": 0, "unparseable": 0}
    if report is None:
        return None, counts
    claims = []
    for c in report.get("claims", []):
        c = dict(c)
        if "box" in c or c.get("claim_type") == "edited_region":
            status = _box_status(c.get("box"))
            counts[status] += 1
            if status == "missing":
                c["box"] = None
            elif status == "unparseable":
                c["box"] = [0, 0, 0, 0]
            else:
                c["box"] = grid_box_to_pixels(c["box"], width, height)
        claims.append(c)
    return {**report, "claims": claims}, counts


def raw_grid_boxes(results: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    """Every numeric 4-value box in the raw (grid) reports of :func:`image_reports` results."""
    out = []
    for r in results:
        for c in (r.get("report") or {}).get("claims", []):
            if _box_status(c.get("box")) not in ("missing", "unparseable"):
                out.append([float(v) for v in c["box"]])
    return out


def boxes_on_grid(boxes: Sequence[Sequence[float]]) -> bool:
    """True if every coordinate lies in 0-1000 (Qwen2-VL's native grid)."""
    return all(0 <= v <= prompts.BOX_GRID for b in boxes for v in b)


def image_reports(llm: Any, images: Sequence[Any]) -> list[dict[str, Any]]:
    """The image baseline: one prompt per image (the same schema, a 0-1000 grid box per region
    claim), run through ``llm.generate_with_images``. Each result keeps the raw report and the
    report with pixel boxes for HR-GT, with its box counts. Unparseable answers keep
    ``report=None`` (a wrong verdict in HR-GT)."""
    texts = llm.generate_with_images([prompts.IMAGE_PROMPT] * len(images), images)
    out = []
    for text, img in zip(texts, images, strict=True):
        h, w = _size(img)
        try:
            rep, err = prompts.parse_report(text), None
        except prompts.ReportError as e:
            rep, err = None, str(e)
        pixel, counts = to_pixel_boxes(rep, w, h)
        out.append(
            {
                "text": text,
                "report": rep,
                "pixel_report": pixel,
                "box_counts": counts,
                "image_size": [h, w],
                "parse_error": err,
            }
        )
    return out


def _size(img: Any) -> tuple[int, int]:
    if hasattr(img, "height") and hasattr(img, "width"):
        return int(img.height), int(img.width)
    return int(img.shape[0]), int(img.shape[1])
