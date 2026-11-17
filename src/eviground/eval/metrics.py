"""Validation accuracy per training source (decision of 14 Nov 2026).

COCO negatives are half of the real class in the unified sampler. If the model learns COCO
photos rather than real-image evidence, COCO's validation accuracy stays high while the other
sources' real images fall behind; per-source numbers make that visible during training.
The trainer (Milestone 3) calls :func:`per_source_accuracy` on every validation pass and writes
:func:`metrics_row` to ``metrics.csv``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

CLASSES = ("real", "generated", "edited")
SOURCES = ("genimage", "sidset", "coco", "diffseg30k")


def _acc(t: np.ndarray, p: np.ndarray) -> float | None:
    return round(float((t == p).mean()), 4) if t.size else None


def _balanced(t: np.ndarray, p: np.ndarray) -> float | None:
    per = [float((p[t == c] == c).mean()) for c in np.unique(t)]
    return round(float(np.mean(per)), 4) if per else None


def per_source_accuracy(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    sources: Sequence[str],
    *,
    known_sources: Sequence[str] = SOURCES,
) -> dict[str, Any]:
    """Accuracy overall and per source, and per class within each source.

    Returns ``{"overall": {n, accuracy, balanced_accuracy}, "by_source": {source: {n, accuracy,
    by_class: {class: {n, accuracy}}}}}``. Every source in ``known_sources`` appears, with
    ``n = 0`` and ``None`` accuracies when the validation set has none of it.
    """
    t, p, s = np.asarray(y_true), np.asarray(y_pred), np.asarray(sources)
    if not (len(t) == len(p) == len(s)):
        raise ValueError("y_true, y_pred and sources must have the same length")
    by_source: dict[str, Any] = {}
    for src in list(known_sources) + sorted(set(s.tolist()) - set(known_sources)):
        m = s == src
        by_class = {}
        for c in CLASSES:
            mc = m & (t == c)
            if mc.any():
                by_class[c] = {"n": int(mc.sum()), "accuracy": _acc(t[mc], p[mc])}
        by_source[src] = {"n": int(m.sum()), "accuracy": _acc(t[m], p[m]), "by_class": by_class}
    return {
        "overall": {
            "n": int(t.size),
            "accuracy": _acc(t, p),
            "balanced_accuracy": _balanced(t, p),
        },
        "by_source": by_source,
    }


def metrics_row(epoch: int, result: dict[str, Any], prefix: str = "val") -> dict[str, Any]:
    """One flat row for ``metrics.csv``: overall and per-source accuracy, fixed column order.

    Columns: ``epoch``, ``{prefix}_acc``, ``{prefix}_bal_acc``, then ``{prefix}_acc_<source>`` for
    every source and ``{prefix}_acc_<source>_<class>`` for every class present in that source.
    """
    row: dict[str, Any] = {
        "epoch": epoch,
        f"{prefix}_acc": result["overall"]["accuracy"],
        f"{prefix}_bal_acc": result["overall"]["balanced_accuracy"],
    }
    for src, v in result["by_source"].items():
        row[f"{prefix}_acc_{src}"] = v["accuracy"]
    for src, v in result["by_source"].items():
        for c, cv in v["by_class"].items():
            row[f"{prefix}_acc_{src}_{c}"] = cv["accuracy"]
    return row


# --------------------------------------------------------------------------------------
# Model selection and mask metrics (Milestone 3; Dhanush's fix 4 of 14 Nov 2026).
# Everything here works on float64 NumPy arrays, outside any autocast.
# --------------------------------------------------------------------------------------


def binary_auc(positive: Sequence[bool], score: Sequence[float]) -> float | None:
    """ROC AUC by the rank statistic (ties get their average rank); None if one class is
    missing."""
    y = np.asarray(positive, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_ovr_auc(labels: Sequence[int], probs: np.ndarray) -> dict[str, float | None]:
    """One-vs-rest AUC per class and their mean over the classes present."""
    y = np.asarray(labels)
    p = np.asarray(probs, dtype=np.float64)
    per = {c: binary_auc(y == i, p[:, i]) for i, c in enumerate(CLASSES)}
    have = [v for v in per.values() if v is not None]
    return {"macro": float(np.mean(have)) if have else None, **per}


MASK_KEYS = (
    "mask_f1",
    "mask_f1_pooled",
    "mask_iou_pooled",
    "mask_miou_binary",
    "real_any_pred",
    "real_pred_area",
    "gen_any_pred",
    "gen_pred_area",
)


def _unmarked(pr: np.ndarray, sel: np.ndarray) -> tuple[float, float]:
    """Share of the selected crops with any predicted pixel, and their mean predicted area."""
    p = pr[sel].reshape(int(sel.sum()), -1)
    return float(p.any(1).mean()), float(p.mean(1).mean())


def mask_scores(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[int],
    edited: int = 2,
    real: int = 0,
    generated: int = 1,
) -> dict[str, float | None]:
    """Mask metrics of binary predictions ``pred`` and targets ``target`` (N, H, W).

    - ``mask_f1``: mean of per-image F1 on edited crops (the selection score's mask term);
    - ``mask_f1_pooled`` and ``mask_iou_pooled``: over all edited pixels together;
    - ``mask_miou_binary``: the mean of the edited-class IoU and the background IoU, from one
      confusion matrix pooled over the edited crops; a guide in CP3, not CP4's P2 number (tiled
      whole images). The DiffSeg30k paper excludes the background class from mIoU (arXiv
      2511.19111v2, section 4.1, checked 17 Nov 2026), so its binary mIoU is ``mask_iou_pooled``;
    - ``real_any_pred`` / ``gen_any_pred``: share of real / fully generated crops with any pixel
      predicted edited (their target is all zeros);
    - ``real_pred_area`` / ``gen_pred_area``: mean predicted-edited share on those crops.
    """
    y = np.asarray(labels)
    pr = np.asarray(pred, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    out: dict[str, float | None] = dict.fromkeys(MASK_KEYS)
    e = y == edited
    if e.any():
        p, t = pr[e].reshape(int(e.sum()), -1), gt[e].reshape(int(e.sum()), -1)
        tp = (p & t).sum(1).astype(np.float64)
        fp = (p & ~t).sum(1).astype(np.float64)
        fn = (~p & t).sum(1).astype(np.float64)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0)
        out["mask_f1"] = float(f1.mean())
        out["mask_f1_pooled"] = float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1))
        out["mask_iou_pooled"] = float(tp.sum() / max(tp.sum() + fp.sum() + fn.sum(), 1))
        tn = (~p & ~t).sum(1).astype(np.float64)
        iou_bg = tn.sum() / max(tn.sum() + fp.sum() + fn.sum(), 1)
        out["mask_miou_binary"] = float((out["mask_iou_pooled"] + iou_bg) / 2)
    for cls, name in ((real, "real"), (generated, "gen")):
        sel = y == cls
        if sel.any():
            out[f"{name}_any_pred"], out[f"{name}_pred_area"] = _unmarked(pr, sel)
    return out


def per_source_mask_report(
    pred: np.ndarray, labels: Sequence[int], predicted_class: Sequence[int], sources: Sequence[str]
) -> dict[str, float]:
    """Per source and class: the share called edited (class argmax), the share with any pixel
    predicted edited, and the mean predicted area (single-source check R1, 15 Nov 2026). Keys are
    ``<source>_<class>_called_edited`` / ``_any_pred`` / ``_pred_area``."""
    y = np.asarray(labels)
    yp = np.asarray(predicted_class)
    s = np.asarray(sources)
    pr = np.asarray(pred, dtype=bool)
    out: dict[str, float] = {}
    for src in sorted(set(s.tolist())):
        for i, cls in enumerate(CLASSES):
            sel = (s == src) & (y == i)
            if not sel.any():
                continue
            key = f"{src}_{cls}"
            out[f"{key}_called_edited"] = float((yp[sel] == 2).mean())
            out[f"{key}_any_pred"], out[f"{key}_pred_area"] = _unmarked(pr, sel)
    return out


def selection_score(
    profile: str, labels: Sequence[int], probs: np.ndarray, mask_f1: float | None
) -> dict[str, float | None]:
    """Model-selection score (decision of 15 Nov 2026, spec section 9 "Batch / stopping").

    unified: mean(macro one-vs-rest AUC, per-image mask F1 on edited validation crops);
    detection: binary AUC real vs generated with the score 1 - P(real).
    A missing term (e.g. no edited validation crop) is left out of the mean and reported None.
    """
    y = np.asarray(labels)
    p = np.asarray(probs, dtype=np.float64)
    if profile == "detection":
        keep = y != 2
        score = 1.0 - p[keep, 0]
        auc = binary_auc(y[keep] != 0, score)
        acc = float(((score >= 0.5) == (y[keep] != 0)).mean()) if keep.any() else None
        return {"selection": auc, "det_auc": auc, "det_acc": acc}
    auc = macro_ovr_auc(y, p)
    terms = [v for v in (auc["macro"], mask_f1) if v is not None]
    return {
        "selection": float(np.mean(terms)) if terms else None,
        "macro_auc": auc["macro"],
        **{f"auc_{c}": auc[c] for c in CLASSES},
    }


PLATEAU_GAIN = 0.005  # CP3 S2b: selection gain over the last 4 epochs below this = plateau
PLATEAU_WINDOW = 4


def plateau_gain(history: Sequence[float | None], window: int = PLATEAU_WINDOW) -> float | None:
    """Gain of the best selection score of the last ``window`` epochs over the best before them
    (None while there are not more than ``window`` epochs)."""
    vals = [v for v in history if v is not None]
    if len(vals) <= window:
        return None
    return max(vals[-window:]) - max(vals[:-window])
