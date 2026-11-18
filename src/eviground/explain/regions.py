"""M5 region extraction (spec section 8, M5).

Candidate regions are the connected components (``scipy.ndimage.label``, 4-connectivity) of a
binary map; components under 0.5% of the image are ignored. For each region: bounding box, area %,
mean mask confidence, attribution mass inside, frequency anomaly (high-frequency 8x8 block-DCT
energy inside relative to the rest) and noise anomaly (variance of the residual image minus its
3x3 mean, inside relative to the rest).

Each region's pixel mask is kept as a run-length encoding (:func:`rle_encode`), saved beside the
evidence JSON keyed by region ID (for HR-GT, counterfactual consistency and deletion); the LLM
never sees it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

MIN_FRAC = 0.005
BLOCK = 8
EPS = 1e-8


def components(binary: np.ndarray, min_frac: float = MIN_FRAC) -> list[np.ndarray]:
    """Boolean masks of the 4-connected components of ``binary`` covering at least ``min_frac``
    of the image, in label order (top-left first)."""
    labels, n = ndimage.label(np.asarray(binary, dtype=bool))
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel())
    keep = [k for k in range(1, n + 1) if sizes[k] >= min_frac * labels.size]
    return [labels == k for k in keep]


def bbox(mask: np.ndarray) -> list[int]:
    """[x, y, width, height] of a boolean mask."""
    ys, xs = np.nonzero(mask)
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def _grey(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img @ np.array([0.299, 0.587, 0.114])
    return img / 255.0 if img.max() > 1.0 else img


def _dct(n: int = BLOCK) -> np.ndarray:
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


def image_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel maps used by the anomalies: ``freq`` (share of high-frequency energy of the 8x8
    block the pixel lies in; edge pixels outside full blocks take 0) and ``resid`` (image minus
    its 3x3 mean)."""
    g = _grey(image)
    h, w = g.shape
    hb, wb = h // BLOCK * BLOCK, w // BLOCK * BLOCK
    freq = np.zeros_like(g)
    if hb and wb:
        d = _dct()
        blocks = g[:hb, :wb].reshape(hb // BLOCK, BLOCK, wb // BLOCK, BLOCK).transpose(0, 2, 1, 3)
        coef = d @ blocks @ d.T
        u, v = np.meshgrid(np.arange(BLOCK), np.arange(BLOCK), indexing="ij")
        high = (u + v) >= BLOCK
        energy = coef**2
        ratio = energy[..., high].sum(-1) / (energy.sum((-1, -2)) + EPS)
        freq[:hb, :wb] = np.kron(ratio, np.ones((BLOCK, BLOCK)))
    resid = g - ndimage.uniform_filter(g, size=3, mode="reflect")
    return {"freq": freq, "resid": resid}


def _ratio_inside(values: np.ndarray, mask: np.ndarray, stat: str) -> float:
    inside, rest = values[mask], values[~mask]
    if inside.size == 0 or rest.size == 0:
        return 1.0
    f = np.var if stat == "var" else np.mean
    return float(f(inside) / (f(rest) + EPS))


def region_stats(
    mask: np.ndarray, mask_prob: np.ndarray, attr: np.ndarray, maps: dict[str, np.ndarray]
) -> dict[str, Any]:
    """The M5 numbers of one region (``maps`` from :func:`image_maps`)."""
    total = float(np.asarray(attr).sum())
    return {
        "bbox": bbox(mask),
        "area_pct": float(100.0 * mask.sum() / mask.size),
        "mask_conf": float(np.asarray(mask_prob)[mask].mean()),
        "attr_mass": float(np.asarray(attr)[mask].sum() / total) if total > 0 else 0.0,
        "freq_anomaly": _ratio_inside(maps["freq"], mask, "mean"),
        "noise_anomaly": _ratio_inside(maps["resid"], mask, "var"),
    }


def top_attribution_regions(
    attr: np.ndarray, k: int = 3, top: float = 0.10, min_frac: float = MIN_FRAC
) -> list[np.ndarray]:
    """Up to ``k`` components of the top ``top`` share of the attribution map, largest
    attribution mass first (the A1-A3 regions of a fully generated verdict)."""
    attr = np.asarray(attr, dtype=np.float64)
    if attr.max() <= 0:
        return []
    cut = np.quantile(attr, 1.0 - top)
    comps = components(attr >= cut, min_frac)
    comps.sort(key=lambda m: -attr[m].sum())
    return comps[:k]


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    """Run lengths of a boolean mask in row-major order, starting with a run of False."""
    flat = np.asarray(mask, dtype=bool).ravel()
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts = [0, *counts]
    return {"shape": list(np.asarray(mask).shape), "counts": [int(c) for c in counts]}


def rle_decode(rle: dict[str, Any]) -> np.ndarray:
    values = np.zeros(len(rle["counts"]), dtype=bool)
    values[1::2] = True
    return np.repeat(values, rle["counts"]).reshape(rle["shape"])
