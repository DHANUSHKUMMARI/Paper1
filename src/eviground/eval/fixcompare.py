"""The CP4 fix against the CP3 model, on validation data only (Dhanush, 17 Nov 2026).

Dhanush asked for these numbers of the retrained unified model next to the CP3 ``best.pt``:

- **on validation crops** (both runs' ``metrics.csv`` at their best epoch): macro AUC, pooled
  edited IoU and mask F1, and the fully generated crops of SID-Set and GenImage called edited
  or with any pixel marked;
- **on whole validation images** (both runs' localization diagnosis): over-marking on
  AI-generated bases (target: clearly below 35.9%), over-marking on real bases, the miss share
  (should not rise much above 19.1%), and the pooled edited-class IoU, overall and by base.

Nothing here reads a test image.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eviground.eval import cp3

CROP_KEYS = {
    "val_macro_auc": "macro AUC (3 classes)",
    "val_mask_iou_pooled": "pooled edited IoU (crops)",
    "val_mask_f1": "mask F1 (crops, per image)",
    "val_genimage_generated_called_edited": "GenImage generated crops called edited",
    "val_sidset_generated_called_edited": "SID-Set generated crops called edited",
    "val_genimage_generated_any_pred": "GenImage generated crops with any mark",
    "val_sidset_generated_any_pred": "SID-Set generated crops with any mark",
    "val_real_any_pred": "real crops with any mark",
}
WHOLE_KEYS = {
    "miou_diffseg30k": "edited IoU (DiffSeg30k mIoU)",
    "fp_share_of_pixels": "over-marking (share of all pixels)",
    "fn_share_of_edited": "misses (share of edited pixels)",
}


def _num(v: Any) -> float | None:
    return cp3._num(v)


def _pair(a: float | None, b: float | None) -> dict[str, float | None]:
    return {
        "cp3": a,
        "fix": b,
        "change": None if a is None or b is None else round(b - a, 6),
    }


def compare_runs(
    cp3_rows: Sequence[Mapping[str, str]], fix_rows: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    """Validation-crop metrics of both runs at their best epochs (``cp3.best_row``)."""
    a, b = cp3.best_row(list(cp3_rows)), cp3.best_row(list(fix_rows))

    def get(row: Mapping[str, str] | None, k: str) -> float | None:
        return None if row is None else _num(row.get(k))

    return {
        "best_epoch": {"cp3": get(a, "epoch"), "fix": get(b, "epoch")},
        "metrics": {label: _pair(get(a, key), get(b, key)) for key, label in CROP_KEYS.items()},
    }


def compare_diagnoses(cp3_diag: Mapping[str, Any], fix_diag: Mapping[str, Any]) -> dict[str, Any]:
    """Whole-image localization of both runs (the ``edited`` block of notebook 08's result),
    overall and per base type, for the model head."""

    def block(diag: Mapping[str, Any], base: str | None) -> Mapping[str, Any]:
        model = diag["edited"]["model"]
        return model["all"] if base is None else model["by_base_type"].get(base, {})

    out: dict[str, Any] = {}
    for name, base in (("all", None), ("ai_base", "ai"), ("real_base", "real")):
        a, b = block(cp3_diag, base), block(fix_diag, base)
        out[name] = {
            label: _pair(_num(a.get(key)), _num(b.get(key))) for key, label in WHOLE_KEYS.items()
        }
    return out
