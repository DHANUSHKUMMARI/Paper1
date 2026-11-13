"""The CP1 metadata shortcut check.

A classifier that sees only file-level facts (format and JPEG quality of the harmonized file,
crop width and height) must not tell real from generated: CP1 requires a score of at most 60%.

The classifier is a lookup table over the exact feature values: for every combination it
predicts the class most typical of that combination in the training folds. On discrete
features this is the best any classifier can do (the Bayes rule of the empirical
distribution), so it is an upper bound for gradient boosting or a decision tree on the same
features, and needs no extra library. Scores are 5-fold cross-validated **balanced**
accuracy: the sampler draws real : generated : edited = 1 : 1 : 1, so plain accuracy of
"real vs not real" would start at 66.7% by predicting "not real" for everything; balanced
accuracy starts at 50%.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence

import numpy as np

CRITERION = 0.60
FOLDS = 5


def balanced_accuracy(
    y: Sequence[Hashable], pred: Sequence[Hashable], weights: Sequence[float] | None = None
) -> float:
    """Mean per-class recall; with ``weights``, each row counts with its weight."""
    y, pred = np.asarray(y, dtype=object), np.asarray(pred, dtype=object)
    w = np.ones(y.size) if weights is None else np.asarray(weights, dtype=np.float64)
    recalls = []
    for c in sorted(set(y.tolist()), key=str):
        m = y == c
        recalls.append(float(w[m & (pred == c)].sum() / w[m].sum()))
    return float(np.mean(recalls)) if recalls else float("nan")


def _folds(y: Sequence[Hashable], k: int, seed: int) -> np.ndarray:
    """Stratified fold number per row."""
    y = np.asarray(y, dtype=object)
    fold = np.zeros(y.size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for c in sorted(set(y.tolist()), key=str):
        idx = np.flatnonzero(y == c)
        fold[idx[rng.permutation(idx.size)]] = np.arange(idx.size) % k
    return fold


def _fit_predict(x_train, y_train, x_test, classes) -> list:
    """Lookup table with class-balanced scores; unseen combinations get the first class."""
    n_class = Counter(y_train)
    table: dict[Hashable, Counter] = defaultdict(Counter)
    for xi, yi in zip(x_train, y_train, strict=True):
        table[xi][yi] += 1
    pred = []
    for xi in x_test:
        cnt = table.get(xi)
        if not cnt:
            pred.append(classes[0])
            continue
        pred.append(max(classes, key=lambda c: (cnt[c] / max(n_class[c], 1), str(c))))
    return pred


def lookup_cv(
    rows: Sequence[tuple], y: Sequence[Hashable], folds: int = FOLDS, seed: int = 42
) -> float:
    """Cross-validated balanced accuracy of the lookup-table classifier."""
    rows = list(rows)
    y = list(y)
    classes = sorted(set(y), key=str)
    if len(classes) < 2:
        return float("nan")
    fold = _folds(y, folds, seed)
    pred: list = [None] * len(y)
    for f in range(folds):
        tr = np.flatnonzero(fold != f)
        te = np.flatnonzero(fold == f)
        p = _fit_predict([rows[i] for i in tr], [y[i] for i in tr], [rows[i] for i in te], classes)
        for i, pi in zip(te, p, strict=True):
            pred[i] = pi
    return balanced_accuracy(y, pred)


def feature_report(
    features: dict[str, Sequence[Hashable]],
    y: Sequence[Hashable],
    folds: int = FOLDS,
    seed: int = 42,
) -> dict[str, object]:
    """Score with all features, each feature alone, and each feature left out."""
    names = list(features)
    n = len(y)
    cols = [list(features[k]) for k in names]

    def score(use: list[int]) -> float:
        if not use:
            return float("nan")
        rows = [tuple(cols[j][i] for j in use) for i in range(n)]
        return lookup_cv(rows, y, folds, seed)

    all_idx = list(range(len(names)))
    return {
        "features": names,
        "n": n,
        "balanced_accuracy": round(score(all_idx), 4),
        "single_feature": {names[j]: round(score([j]), 4) for j in all_idx},
        "drop_one": {names[j]: round(score([i for i in all_idx if i != j]), 4) for j in all_idx}
        if len(names) > 1
        else {},
        "distinct_values": {k: len(set(features[k])) for k in names},
    }


def threshold_cv(
    values: Sequence[float],
    y: Sequence[Hashable],
    folds: int = FOLDS,
    seed: int = 42,
    weights: Sequence[float] | None = None,
) -> float:
    """Balanced accuracy of the best single threshold on one numeric value (binary ``y``),
    cross-validated. Used to say whether blockiness separates two classes.

    ``weights`` give each row its share (e.g. a source's share in the sampler) without
    repeating rows: repeated rows would sit in both training and test folds and make chance
    differences look real."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=object)
    w = np.ones(v.size) if weights is None else np.asarray(weights, dtype=np.float64)
    classes = sorted(set(y.tolist()), key=str)
    if len(classes) != 2:
        raise ValueError("threshold_cv needs exactly two classes")
    fold = _folds(y, folds, seed)
    pred = np.empty(y.size, dtype=object)
    for f in range(folds):
        tr, te = fold != f, fold == f
        cuts = np.unique(np.quantile(v[tr], np.linspace(0.02, 0.98, 49)))
        best = (-1.0, cuts[0], classes[0])
        for c in cuts:
            for hi_class in classes:
                lo_class = classes[1] if hi_class == classes[0] else classes[0]
                p = np.where(v[tr] > c, hi_class, lo_class)
                s = balanced_accuracy(y[tr], p, w[tr])
                if s > best[0]:
                    best = (s, c, hi_class)
        _, c, hi_class = best
        lo_class = classes[1] if hi_class == classes[0] else classes[0]
        pred[te] = np.where(v[te] > c, hi_class, lo_class)
    return balanced_accuracy(y, pred, w)


def describe(values: Sequence[float]) -> dict[str, float | int]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 4),
        "p10": round(float(q[0]), 4),
        "p25": round(float(q[1]), 4),
        "median": round(float(q[2]), 4),
        "p75": round(float(q[3]), 4),
        "p90": round(float(q[4]), 4),
    }
