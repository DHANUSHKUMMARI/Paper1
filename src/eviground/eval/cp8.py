"""CP8 Claims-Edit scoring (the ``cp8`` block of notebook 17, unified seed 42; Dhanush, 27 Nov).

One pass per image (the 150 real photos and their edited copies) gives everything the later T4
reports need, so no further L4 session is required:

- the verdict and probabilities (:func:`cp5.explain_image`);
- pixel counts of the decoder map against the saved edit mask, and against its 8 px boundary
  band (the composite seam diagnostic, :func:`claims_edit.boundary_band`);
- the M5 evidence bundle (as notebook 14 wrote for P6), for the grounded reports;
- CC (the Phase C rule) for every E region, run by the notebook on the saved bundle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eviground.data import claims_edit
from eviground.eval.phasec import pixel_summary
from eviground.eval.tiled import Item, mask_record, per_image
from eviground.explain import cc, cp5, evidence

BAND_PX = 8


def score_item(
    model: Any, item: Item, device: Any, amp: Any, bundles: Path | str, *, band_px: int = BAND_PX
) -> dict[str, Any]:
    """Explain one image, save its evidence bundle, and return its record."""
    out = cp5.explain_image(model, item.image, device, amp)
    record, rles = evidence.evidence_record(
        out["probs"], out["gate"], out["decoder"], out["gradcam_pred"], item.image
    )
    maps = {
        "decoder": out["decoder"],
        "gradcam": out["gradcam"],
        "gradcam_pred": out["gradcam_pred"],
    }
    evidence.save_bundle(bundles, item.uid, record, rles, maps, out["probs"], out["gate"])
    rec: dict[str, Any] = {
        "uid": item.uid,
        "label": item.meta.get("label"),
        "pair": item.meta.get("pair"),
        "category": item.meta.get("category"),
        "damage": item.meta.get("damage"),
        "label_pred": out["verdict"],
        "probs": dict(out["probs"]),
        "tiles": out["tiles"],
        "regions": len(record["regions"]),
    }
    if item.mask is not None and item.mask.any():
        rec["mask"] = mask_record(out["decoder"], item.mask)
        band = mask_record(out["decoder"], claims_edit.boundary_band(item.mask, band_px))
        # counts only: the histograms are not needed for the band IoU
        rec["band"] = {k: v for k, v in band.items() if not isinstance(v, np.ndarray)}
    return rec


def _acc(rs: Sequence[Mapping[str, Any]]) -> float | None:
    return float(np.mean([r["label_pred"] == r["label"] for r in rs])) if rs else None


def cp8_summary(
    records: Iterable[Mapping[str, Any]], cc_regions: Iterable[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """The P7 table numbers (spec): image accuracy (all, real, edited) and the confusion, pooled
    pixel metrics on the edited copies, the boundary-band IoU beside them, and CC."""
    rs = list(records)
    real = [r for r in rs if r["label"] == "real"]
    edited = [r for r in rs if r["label"] == "edited"]
    preds = ("real", "generated", "edited")
    confusion = {lab: {p: sum(r["label_pred"] == p for r in rs if r["label"] == lab) for p in preds}
                 for lab in ("real", "edited")}  # fmt: skip
    bands = [per_image(r["band"])["iou"] for r in edited if "band" in r]
    bands = [b for b in bands if b is not None]
    return {
        "images": len(rs),
        "acc": _acc(rs),
        "acc_real": _acc(real),
        "acc_edited": _acc(edited),
        "confusion": confusion,
        "pixel": pixel_summary(edited),
        "band_iou_mean": float(np.mean(bands)) if bands else None,
        "band_px": BAND_PX,
        "cc": cc.cc_summary(cc_regions),
    }
