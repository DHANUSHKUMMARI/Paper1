"""CP3 — training health (spec section 9.1, with Dhanush's additions of 15 Nov 2026).

Reads one run folder (``metrics.csv``, ``steps.csv``, ``nonfinite.json`` if any) and, for the
unified model, the 64-image overfit record, and scores every criterion at the best epoch (the
epoch with the highest validation selection score):

- S1  no NaN or divergence: every logged loss term and gradient norm finite, no ``nonfinite.json``,
      best-epoch mean training loss below epoch 0's;
- S2a validation AUC with a floor: unified macro one-vs-rest AUC >= 0.90 (any per-class AUC below
      0.80 is a stop-and-ask); detection binary AUC >= 0.90;
- S2b plateau: early stopping triggered, or the selection score gained < 0.005 over the last 4
      epochs; still improving at the last epoch is a stop-and-ask;
- S3  in-distribution SD v1.4 accuracy >= 0.95 (detection model; the unified model's GenImage
      accuracy is reported);
- A1  (unified) 64-image overfit: accuracy >= 0.99 within 300 steps and per-image mean mask F1 on
      the edited images >= 0.80 within 1,000 steps;
- A2  (unified) real crops: any predicted edited pixel <= 0.20 of crops, mean area <= 0.01;
- A3  (unified) fully generated crops: the same two limits;
- A4  (unified) per-image mean mask F1 on the validation edited crops >= 0.50;
- R1  (unified, report) per source: a real DiffSeg30k crop called edited more often than a real COCO
      crop by more than 0.10 is a stop-and-ask (the single-source risk);
- R2  (unified, report) binary mIoU on validation crops: a guide only, not CP4's P2 number, which
      is measured on whole tiled images from held-out editors.

The masks are the raw decoder output at threshold 0.5 (no class-gating). Every value is compared
with the untrained model of CP2.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from eviground.eval.metrics import PLATEAU_GAIN, plateau_gain

THRESHOLDS = {
    "S2a_macro_auc_min": 0.90,
    "S2a_class_auc_stop_below": 0.80,
    "S2a_det_auc_min": 0.90,
    "S2b_plateau_gain": PLATEAU_GAIN,
    "S3_det_acc_min": 0.95,
    "A1_acc_min": 0.99,
    "A1_acc_steps": 300,
    "A1_mask_f1_min": 0.80,
    "A1_mask_steps": 1000,
    "A2_A3_any_pred_max": 0.20,
    "A2_A3_pred_area_max": 0.01,
    "A4_mask_f1_min": 0.50,
    "R1_called_edited_gap_stop": 0.10,
}
UNTRAINED_CP2 = {  # notebook 05, 15 Nov 2026, untrained model on the same validation crops
    "val_macro_auc": 0.5234,
    "val_auc_real": 0.5131,
    "val_auc_generated": 0.5451,
    "val_auc_edited": 0.5120,
    "val_mask_f1": 0.2162,
    "val_real_any_pred": 1.0,
    "val_real_pred_area": 0.1405,
}
PASS, FAIL, STOP, REPORT = "PASS", "FAIL", "STOP_AND_ASK", "REPORT"


def _num(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _ok(cond: bool) -> str:
    return PASS if cond else FAIL


def _at_least(value: float | None, floor: float) -> str:
    return _ok(value is not None and value >= floor)


def _at_most(value: float | None, ceiling: float) -> str:
    return _ok(value is not None and value <= ceiling)


def best_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    scored = [r for r in rows if _num(r.get("val_selection")) is not None]
    return max(scored, key=lambda r: _num(r["val_selection"])) if scored else None


def check_s1(rows: list[dict[str, str]], steps: list[dict[str, str]], run_dir: Path) -> dict:
    cols = [k for k in (steps[0] if steps else {}) if k.startswith("loss_") or k == "grad_norm"]
    bad = [
        (r.get("global_step"), k)
        for r in steps
        for k in cols
        if (v := _num(r.get(k))) is None or not math.isfinite(v)
    ]
    first = _num(rows[0].get("train_total")) if rows else None
    best = best_row(rows)
    best_loss = _num(best.get("train_total")) if best else None
    loss_fell = first is not None and best_loss is not None and best_loss < first
    nonfinite = (run_dir / "nonfinite.json").is_file()
    return {
        "status": _ok(bool(steps) and not bad and not nonfinite and loss_fell),
        "steps_logged": len(steps),
        "non_finite_values": bad[:10],
        "nonfinite_record": nonfinite,
        "epoch0_train_loss": first,
        "best_epoch_train_loss": best_loss,
    }


def check_s2(profile: str, rows: list[dict[str, str]], best: dict[str, str]) -> dict:
    t = THRESHOLDS
    if profile == "detection":
        auc = _num(best.get("val_det_auc"))
        s2a = {"status": _at_least(auc, t["S2a_det_auc_min"]), "det_auc": auc}
    else:
        auc = _num(best.get("val_macro_auc"))
        per_class = {c: _num(best.get(f"val_auc_{c}")) for c in ("real", "generated", "edited")}
        weak = [c for c, v in per_class.items() if v is None or v < t["S2a_class_auc_stop_below"]]
        status = _at_least(auc, t["S2a_macro_auc_min"])
        if status == PASS and weak:
            status = STOP
        s2a = {"status": status, "macro_auc": auc, "per_class_auc": per_class, "weak": weak}
    history = [_num(r.get("val_selection")) for r in rows]
    gain = plateau_gain(history)
    early_stop = bool(rows) and int(_num(rows[-1].get("bad_epochs")) or 0) >= 4
    plateaued = early_stop or (gain is not None and gain < t["S2b_plateau_gain"])
    s2b = {
        "status": PASS if plateaued else STOP,
        "early_stop": early_stop,
        "plateau_gain": gain,
        "epochs": len(rows),
    }
    return {"S2a": s2a, "S2b": s2b}


def check_s3(profile: str, best: dict[str, str]) -> dict:
    if profile == "detection":
        acc = _num(best.get("val_det_acc"))
        return {"status": _at_least(acc, THRESHOLDS["S3_det_acc_min"]), "det_acc": acc}
    return {"status": REPORT, "genimage_acc_3class": _num(best.get("val_acc_genimage"))}


def check_a1(overfit: dict[str, Any] | None) -> dict:
    if not overfit:
        return {"status": FAIL, "reason": "no overfit record"}
    ok = bool(overfit.get("acc_pass")) and bool(overfit.get("mask_pass"))
    return {
        "status": _ok(ok),
        "acc_reached_at_step": overfit.get("acc_reached_at_step"),
        "mask_f1_reached_at_step": overfit.get("mask_f1_reached_at_step"),
        "final_acc": overfit.get("final_acc"),
        "final_mask_f1": overfit.get("final_mask_f1"),
        "by_class": overfit.get("by_class"),
    }


def check_unmarked(best: dict[str, str], name: str) -> dict:
    t = THRESHOLDS
    any_pred = _num(best.get(f"val_{name}_any_pred"))
    area = _num(best.get(f"val_{name}_pred_area"))
    ok = (
        _at_most(any_pred, t["A2_A3_any_pred_max"]) == PASS
        and _at_most(area, t["A2_A3_pred_area_max"]) == PASS
    )
    return {"status": _ok(ok), "any_pred": any_pred, "pred_area": area}


def check_r1(best: dict[str, str]) -> dict:
    keys = ("called_edited", "any_pred", "pred_area")
    diffseg = {k: _num(best.get(f"val_diffseg30k_real_{k}")) for k in keys}
    coco = {k: _num(best.get(f"val_coco_real_{k}")) for k in keys}
    gap = (
        None
        if diffseg["called_edited"] is None or coco["called_edited"] is None
        else diffseg["called_edited"] - coco["called_edited"]
    )
    stop = gap is not None and gap > THRESHOLDS["R1_called_edited_gap_stop"]
    return {
        "status": STOP if stop else REPORT,
        "called_edited_gap": gap,
        "diffseg30k_real": diffseg,
        "coco_real": coco,
    }


def evaluate_run(
    run_dir: Path, profile: str, overfit: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score one run folder; ``overfit`` is the A1 record (unified model)."""
    run_dir = Path(run_dir)
    rows = read_csv(run_dir / "metrics.csv")
    steps = read_csv(run_dir / "steps.csv")
    best = best_row(rows)
    if best is None:
        reason = "no scored epoch in metrics.csv"
        return {"verdict": FAIL, "reason": reason, "run_dir": str(run_dir)}
    checks: dict[str, Any] = {"S1": check_s1(rows, steps, run_dir)}
    checks.update(check_s2(profile, rows, best))
    checks["S3"] = check_s3(profile, best)
    if profile == "unified":
        checks["A1"] = check_a1(overfit)
        checks["A2"] = check_unmarked(best, "real")
        checks["A3"] = check_unmarked(best, "gen")
        f1 = _num(best.get("val_mask_f1"))
        checks["A4"] = {"status": _at_least(f1, THRESHOLDS["A4_mask_f1_min"]), "mask_f1": f1}
        checks["R1"] = check_r1(best)
        miou = _num(best.get("val_mask_miou_binary"))
        checks["R2"] = {"status": REPORT, "mask_miou_binary": miou}
    statuses = [c["status"] for c in checks.values()]
    verdict = FAIL if FAIL in statuses else STOP if STOP in statuses else PASS
    vs_untrained = {
        k: {"untrained": v, "best": _num(best.get(k))}
        for k, v in UNTRAINED_CP2.items()
        if profile == "unified" or k == "val_macro_auc"
    }
    return {
        "verdict": verdict,
        "profile": profile,
        "run_dir": str(run_dir),
        "best_epoch": int(_num(best.get("epoch")) or 0),
        "epochs_run": len(rows),
        "checks": checks,
        "vs_untrained_cp2": vs_untrained,
        "thresholds": THRESHOLDS,
        "versions": {k: best.get(k) for k in ("index_sha256", "mask_rule", "clip_revision")},
    }


def write_report(report: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str))
    return path
