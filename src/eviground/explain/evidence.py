"""M5 evidence record and the per-image bundle (spec section 8, M5; Dhanush, 18 Nov 2026).

:func:`evidence_record` builds the JSON the LLM reads (IDs E1... for mask regions, A1-A3 for the
attribution regions of a fully generated verdict, G1... for global evidence). The LLM never sees
pixels, paths or region masks.

:func:`save_bundle` writes, per image, everything CP6 needs so that it never loads the detector:
the evidence JSON (with the probabilities and the gate), the region RLEs keyed by region ID, and
float16 maps at 1/4 resolution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from eviground.explain import regions

MASK_THRESHOLD = 0.5
PRED_KEYS = {"real": "real", "generated": "fully_generated", "edited": "locally_edited"}
DOWNSCALE = 4


def verdict(probs: Mapping[str, float], edit_threshold: float = MASK_THRESHOLD) -> str:
    """Edited if P(edited) >= the threshold, else the larger of real and generated (the
    test-time rule of ``models.eviground.decide``)."""
    if probs["edited"] >= edit_threshold:
        return "edited"
    return "real" if probs["real"] >= probs["generated"] else "generated"


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def evidence_record(
    probs: Mapping[str, float],
    gate: float,
    mask_prob: np.ndarray,
    attr: np.ndarray,
    image: np.ndarray,
    *,
    threshold: float = MASK_THRESHOLD,
    min_frac: float = regions.MIN_FRAC,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The evidence JSON and the region masks (RLE, keyed by region ID)."""
    maps = regions.image_maps(image)
    decided = verdict(probs, threshold)
    record: dict[str, Any] = {
        "prediction": {PRED_KEYS[k]: _r(probs[k]) for k in ("real", "generated", "edited")},
        "verdict": PRED_KEYS[decided],
        "global": [
            {"id": "G1", "type": "branch_reliance", "semantic": _r(gate), "trace": _r(1 - gate)},
            {"id": "G2", "type": "image_freq_anomaly", "value": _r(float(maps["freq"].mean()))},
            {"id": "G3", "type": "image_noise_anomaly", "value": _r(float(maps["resid"].std()))},
        ],
        "regions": [],
    }
    if decided == "edited":
        comps, prefix = regions.components(np.asarray(mask_prob) >= threshold, min_frac), "E"
    elif decided == "generated":
        comps, prefix = regions.top_attribution_regions(attr, min_frac=min_frac), "A"
    else:
        comps, prefix = [], ""
    rles = {}
    for i, comp in enumerate(comps, start=1):
        rid = f"{prefix}{i}"
        stats = regions.region_stats(comp, mask_prob, attr, maps)
        record["regions"].append(
            {"id": rid, **{k: (_r(v, 2) if isinstance(v, float) else v) for k, v in stats.items()}}
        )
        rles[rid] = regions.rle_encode(comp)
    ids = [g["id"] for g in record["global"]] + [r["id"] for r in record["regions"]]
    assert len(ids) == len(set(ids)), ids
    return record, rles


def _stem(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()[:20]


def _down(m: np.ndarray, k: int = DOWNSCALE) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    h, w = m.shape[0] // k * k, m.shape[1] // k * k
    if h == 0 or w == 0:
        return m.astype(np.float16)
    return m[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3)).astype(np.float16)


def save_bundle(
    folder: Path,
    uid: str,
    record: Mapping[str, Any],
    rles: Mapping[str, Any],
    maps: Mapping[str, np.ndarray],
    probs: Mapping[str, float],
    gate: float,
) -> Path:
    """Write ``<stem>.json`` (uid, evidence, probabilities, gate), ``<stem>.regions.json`` and
    ``<stem>.npz`` (maps as float16 at 1/4 resolution) into ``folder``; returns the JSON path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = _stem(uid)
    meta = {
        "uid": uid,
        "evidence": dict(record),
        "probs": {k: float(v) for k, v in probs.items()},
        "gate": float(gate),
        "maps_downscale": DOWNSCALE,
    }
    (folder / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    (folder / f"{stem}.regions.json").write_text(json.dumps(dict(rles)))
    np.savez_compressed(folder / f"{stem}.npz", **{k: _down(v) for k, v in maps.items()})
    return folder / f"{stem}.json"


def load_bundle(folder: Path, uid: str) -> dict[str, Any]:
    """The saved bundle of ``uid``: the JSON fields, ``regions`` (id -> boolean mask) and
    ``maps`` (name -> float16 array at 1/4 resolution)."""
    folder, stem = Path(folder), _stem(uid)
    meta = json.loads((folder / f"{stem}.json").read_text())
    rles = json.loads((folder / f"{stem}.regions.json").read_text())
    with np.load(folder / f"{stem}.npz") as z:
        maps = {k: z[k] for k in z.files}
    return {
        **meta,
        "regions": {k: regions.rle_decode(v) for k, v in rles.items()},
        "maps": maps,
    }
