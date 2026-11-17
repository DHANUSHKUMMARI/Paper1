"""CP4, generalization gate (spec section 9.1; plan approved 16 Nov 2026), on the development split.

- G1: P1 mean accuracy over the 8 generators of the detection-protocol model above 0.721 (the
  GenImage ResNet-50 average, the go/no-go bar) and above the detection CLIP probe, on raw and
  on harmonized files. Score 1 - P(real) (mean over tiles), generated at >= 0.5.
- RF: accuracy >= 0.98 on every unseen generator (all but SD v1.4) on raw files: stop and ask.
- G2: P2 mIoU >= 0.80 (unified model), with the DiffSeg30k paper's definition (Dhanush, 17 Nov):
  the edited-class IoU, background excluded, pooled over all P2 dev pixels; the mean of the
  edited and background IoU is reported beside it as ``miou_two_class``. P2 is scored only once
  Dhanush has seen the localization diagnosis; until then G2 is ``NOT_RUN`` and the verdict
  ``PENDING_P2``.
- Reported: the unified model's P1 row, AP, the probes on both versions, the false "edited" rate
  on real images by tile count, and published GenImage results for context (full-set numbers,
  checked against arXiv 2511.01293 Table 7 on 17 Nov 2026).
- One fix only; no second fix: a second failure is recorded and Dhanush decides how to report it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eviground.eval.locdiag import tile_bin

G1_BAR = 0.721
RF_BAR = 0.98
G2_BAR = 0.80
SD14 = {"stable_diffusion_v_1_4", "sd14", "sd_v1_4"}  # the in-distribution generator
PASS, FAIL, STOP, NOT_RUN, REPORT = "PASS", "FAIL", "STOP_AND_ASK", "NOT_RUN", "REPORT"
ARXIV_T7 = "arXiv 2511.01293, Table 7"
CHECKED = "2026-11-17: Table 7 'Average' column, SD v1.4 training (the paper's appendix)"
PUBLISHED = [  # context only; mean accuracy over the 8 GenImage subsets, SD v1.4 training
    {"method": "ResNet-50", "mean_acc": 0.721, "source": ARXIV_T7, "checked": CHECKED},
    {"method": "NPR", "mean_acc": 0.886, "source": ARXIV_T7, "checked": CHECKED},
    {"method": "FatFormer", "mean_acc": 0.889, "source": ARXIV_T7, "checked": CHECKED},
    {"method": "Ojha et al. (UnivFD)", "mean_acc": 0.733, "source": ARXIV_T7, "checked": CHECKED},
]


def average_precision(positive: Sequence[bool], score: Sequence[float]) -> float | None:
    y = np.asarray(positive, dtype=bool)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def p1_table(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Records: ``head``, ``version``, ``generator``, ``label`` (real / generated), ``p_real``.
    Returns accuracy and AP per generator and their means, per head and version."""
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[(r["head"], r["version"])][r["generator"]].append(r)
    out: dict[str, Any] = {}
    for (head, version), gens in sorted(groups.items()):
        per = {}
        for gen, rs in sorted(gens.items()):
            score = [1 - r["p_real"] for r in rs]
            fake = [r["label"] == "generated" for r in rs]
            acc = float(np.mean([(s >= 0.5) == f for s, f in zip(score, fake, strict=True)]))
            per[gen] = {
                "images": len(rs),
                "acc": acc,
                "acc_real": _mean([(s < 0.5) for s, f in zip(score, fake, strict=True) if not f]),
                "acc_generated": _mean([(s >= 0.5) for s, f in zip(score, fake, strict=True) if f]),
                "ap": average_precision(fake, score),
            }
        out.setdefault(head, {})[version] = {
            "per_generator": per,
            "mean_acc": float(np.mean([v["acc"] for v in per.values()])),
            "mean_ap": _mean([v["ap"] for v in per.values() if v["ap"] is not None]),
        }
    return out


def _mean(v: Sequence[float]) -> float | None:
    return float(np.mean(v)) if len(v) else None


def false_edited(records: Sequence[Mapping[str, Any]], head: str) -> dict[str, Any]:
    """Real images called edited by ``head``, by tile count."""
    real = [r for r in records if r["head"] == head and r["label"] == "real"]
    bins: dict[str, list] = defaultdict(list)
    for r in real:
        bins[tile_bin(r["tiles"])].append(r["label_pred"] == "edited")
    return {
        "images": len(real),
        "rate": _mean([r["label_pred"] == "edited" for r in real]),
        "by_tiles": {k: {"images": len(v), "rate": float(np.mean(v))} for k, v in bins.items()},
    }


def _g1(m: Mapping[str, Any], p: Mapping[str, Any]) -> dict[str, Any]:
    rows = {
        v: {"model": m[v]["mean_acc"], "probe": p[v]["mean_acc"]}
        for v in ("raw", "harmonized")
        if v in m and v in p
    }
    ok = (
        set(rows) == {"raw", "harmonized"}
        and rows["raw"]["model"] > G1_BAR
        and all(r["model"] > r["probe"] for r in rows.values())
    )
    return {"status": PASS if ok else FAIL, "bar": G1_BAR, "mean_acc": rows}


def _rf(m: Mapping[str, Any]) -> dict[str, Any]:
    unseen = {g: v["acc"] for g, v in m["raw"]["per_generator"].items() if g not in SD14}
    raised = bool(unseen) and all(a >= RF_BAR for a in unseen.values())
    return {
        "status": STOP if raised else PASS,
        "bar": RF_BAR,
        "unseen_acc": unseen,
        "raised": raised,
    }


def _g2(p2: Mapping[str, Any] | None) -> dict[str, Any]:
    if p2 is None:
        return {"status": NOT_RUN, "bar": G2_BAR, "note": "held until Dhanush decides"}
    miou = p2.get("miou_diffseg30k")
    ok = miou is not None and miou >= G2_BAR
    return {
        "status": PASS if ok else FAIL,
        "bar": G2_BAR,
        "metric": "miou_diffseg30k (edited-class IoU, background excluded, pooled; 17 Nov)",
        "miou_diffseg30k": miou,
        "miou_two_class": p2.get("miou_two_class"),
    }


def _verdict(checks: Mapping[str, Mapping[str, Any]]) -> str:
    if checks["RF"]["raised"]:
        return STOP
    if FAIL in (checks["G1"]["status"], checks["G2"]["status"]):
        return FAIL
    return "PENDING_P2" if checks["G2"]["status"] == NOT_RUN else PASS


def evaluate(
    table: Mapping[str, Any],
    *,
    model: str = "detection",
    probe: str = "detection_probe",
    p2: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The gate from the P1 table (and the P2 summary, once P2 is scored)."""
    checks = {"G1": _g1(table[model], table[probe]), "RF": _rf(table[model]), "G2": _g2(p2)}
    return {
        "verdict": _verdict(checks),
        "checks": checks,
        "p1": table,
        "published_context": PUBLISHED,
        **(dict(extra) if extra else {}),
    }


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str))
    return path
