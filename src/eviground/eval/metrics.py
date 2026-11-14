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
