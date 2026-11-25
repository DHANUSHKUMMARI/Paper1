"""T4 (fp16) vs L4 (bf16) parity check before Phase C may run on a T4 (Dhanush, 25 Nov 2026).

The L4 references are what was saved on the L4 for the final model (unified, context crops, seed
42): the pooled edited IoU of the 486 P2 dev images (CP4 G2, 0.5101339042612494) and notebook
08's per-image validation results (``diagnosis_edited_seed42_ctx.csv``: per-image IoU of edited
images; ``diagnosis_real_seed42_ctx.csv``: the verdict on real images). P(edited) was never saved
on the L4, and the edited CSV has no predicted label, so verdicts are compared on real images and
the edited images are compared by per-image IoU.

Pass = every criterion (pre-registered in a spec card before the run):

1. pooled edited IoU on P2 dev within ``IOU_TOL`` of the L4 value;
2. verdict agreement on the sampled real validation images >= ``MIN_AGREEMENT``;
3. median |T4 - L4| per-image IoU on the sampled edited validation images <= ``MAX_MEDIAN_DIOU``;
4. (probe, added 25 Nov) the cached unified probe's verdicts on the same real images agree with
   notebook 08's L4 ``probe_label`` >= ``MIN_AGREEMENT``.

Outcome (Dhanush, 25 Nov): 1-3 and 4 pass -> Phase C may run on a T4 with the unified probe rows;
1-3 pass, 4 fails -> on a T4 without probe rows (they run later on an L4); 1-3 fail -> wait for an
L4. The detection probe has no L4 per-image reference in notebook 08's files, so its rows never
run on a T4.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

L4_P2_IOU = 0.5101339042612494  # cp4_generalization_gate_seed42_ctx.json, p2.model.all.iou_edited
IOU_TOL = 0.005
MIN_AGREEMENT = 0.995
MAX_MEDIAN_DIOU = 0.01
N_EACH = 300
SALT = "t4_parity_v1"


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _hash(uids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(uids)).encode()).hexdigest()


def sample_parity(
    edited_uids: Iterable[str],
    real_uids: Iterable[str],
    n: int = N_EACH,
    seed: int = 42,
    salt: str = SALT,
) -> dict[str, Any]:
    """``n`` edited and ``n`` real uids drawn with a fixed seed from the sorted reference uids
    (all of them if fewer), with the SHA-256 of each sorted list and of both together."""
    out: dict[str, Any] = {"seed": seed, "salt": salt, "n_each": n}
    for name, uids in (("edited", edited_uids), ("real", real_uids)):
        pool = sorted(set(uids))
        # a reproducible draw (not security): numpy's generator seeded from the seed, salt and set
        digest = hashlib.sha256(f"{seed}:{salt}:{name}".encode()).digest()
        key = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(key)
        picks = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        chosen = sorted(pool[i] for i in picks)
        out[name] = {"uids": chosen, "images": len(chosen), "pool": len(pool),
                     "sha256": _hash(chosen)}  # fmt: skip
    out["sha256"] = _hash([f"edited:{u}" for u in out["edited"]["uids"]]
                          + [f"real:{u}" for u in out["real"]["uids"]])  # fmt: skip
    return out


def intact(sample: Mapping[str, Any]) -> bool:
    """True if the frozen sample's lists still match their hashes."""
    both = [f"edited:{u}" for u in sample["edited"]["uids"]] + [
        f"real:{u}" for u in sample["real"]["uids"]
    ]
    return (
        _hash(sample["edited"]["uids"]) == sample["edited"]["sha256"]
        and _hash(sample["real"]["uids"]) == sample["real"]["sha256"]
        and _hash(both) == sample["sha256"]
    )


def parity_verdict(
    t4_p2_iou: float,
    t4_edited_iou: Mapping[str, float],
    t4_real_label: Mapping[str, str],
    ref_edited_iou: Mapping[str, float],
    ref_real_label: Mapping[str, str],
    *,
    t4_probe_label: Mapping[str, str] | None = None,
    ref_probe_label: Mapping[str, str] | None = None,
    ref_p2_iou: float = L4_P2_IOU,
    iou_tol: float = IOU_TOL,
    min_agreement: float = MIN_AGREEMENT,
    max_median_diou: float = MAX_MEDIAN_DIOU,
) -> dict[str, Any]:
    """The criteria, ``model_pass`` (1-3), ``probe_pass`` (4, None without probe labels), ``pass``
    (= ``model_pass``: whether Phase C may run on a T4 at all) and the ``outcome``. Every sampled
    uid must have a T4 result and a reference; a missing one is an error, never dropped."""
    missing_e = sorted(set(ref_edited_iou) - set(t4_edited_iou))
    missing_r = sorted(set(ref_real_label) - set(t4_real_label))
    if missing_e or missing_r:
        raise ValueError(f"no T4 result for {len(missing_e)} edited / {len(missing_r)} real images")
    d_iou = np.abs(
        np.array([t4_edited_iou[u] - ref_edited_iou[u] for u in sorted(ref_edited_iou)], float)
    )
    agree = [t4_real_label[u] == ref_real_label[u] for u in sorted(ref_real_label)]
    disagree = [u for u in sorted(ref_real_label) if t4_real_label[u] != ref_real_label[u]]
    c1 = {"t4": t4_p2_iou, "l4": ref_p2_iou, "diff": t4_p2_iou - ref_p2_iou, "tol": iou_tol}
    c1["pass"] = abs(c1["diff"]) <= iou_tol
    c2 = {"images": len(agree), "agreement": float(np.mean(agree)) if agree else None,
          "min": min_agreement, "disagreeing_uids": disagree}  # fmt: skip
    c2["pass"] = c2["agreement"] is not None and c2["agreement"] >= min_agreement
    some = bool(d_iou.size)
    c3 = {
        "images": int(d_iou.size),
        "median_abs_diff": float(np.median(d_iou)) if some else None,
        "max_abs_diff": float(d_iou.max()) if some else None,
        "max": max_median_diou,
    }
    c3["pass"] = c3["median_abs_diff"] is not None and c3["median_abs_diff"] <= max_median_diou
    out = {"p2_pooled_iou": c1, "real_verdict_agreement": c2, "edited_per_image_iou": c3}
    model_pass = bool(c1["pass"] and c2["pass"] and c3["pass"])
    probe_pass = None
    if ref_probe_label is not None:
        c4 = _agreement(t4_probe_label or {}, ref_probe_label, min_agreement)
        out["real_probe_verdict_agreement"] = c4
        probe_pass = c4["pass"]
    if not model_pass:
        outcome = "wait for an L4"
    elif probe_pass:
        outcome = "T4 with the unified probe rows"
    else:
        outcome = "T4 without probe rows (probe rows later on an L4)"
    return {**out, "model_pass": model_pass, "probe_pass": probe_pass, "pass": model_pass,
            "outcome": outcome}  # fmt: skip


def _agreement(t4: Mapping[str, str], ref: Mapping[str, str], minimum: float) -> dict[str, Any]:
    missing = sorted(set(ref) - set(t4))
    if missing:
        raise ValueError(f"no T4 probe result for {len(missing)} real images")
    agree = [t4[u] == ref[u] for u in sorted(ref)]
    share = float(np.mean(agree)) if agree else None
    return {"images": len(agree), "agreement": share, "min": minimum,
            "disagreeing_uids": [u for u in sorted(ref) if t4[u] != ref[u]],
            "pass": share is not None and share >= minimum}  # fmt: skip
