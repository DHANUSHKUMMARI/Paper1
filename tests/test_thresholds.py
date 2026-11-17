"""Validation-only threshold sweep of the localization diagnosis (16 Nov 2026)."""

from __future__ import annotations

import numpy as np
import pytest

from eviground.eval import locdiag, thresholds
from eviground.eval.tiled import boundary_counts, mask_record, pooled


def _image(seed: int, h: int = 40, w: int = 56):
    rng = np.random.default_rng(seed)
    gt = np.zeros((h, w), bool)
    gt[8 : 8 + rng.integers(5, 20), 10 : 10 + rng.integers(5, 30)] = True
    # quantised scores, away from the 1/1000 bin edges, so direct thresholding is unambiguous
    prob = (rng.integers(0, 1000, size=(h, w)) + 0.5) / 1000
    prob = np.where(gt, np.minimum(prob + 0.3, 0.9995), prob).astype(np.float32)
    return prob, gt


def _records(n: int = 6):
    out = []
    for i in range(n):
        prob, gt = _image(i)
        out.append(
            {
                "prob": prob,
                "gt": gt,
                "heads": {"model": mask_record(prob, gt)},
                "boundary_grid": {"model": thresholds.boundary_grid(prob, gt)},
            }
        )
    return out


@pytest.mark.parametrize("t", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_histogram_sweep_equals_direct_thresholding(t):
    recs = _records()
    row = next(r for r in thresholds.pixel_sweep(recs, "model", grid=(t,)))
    direct = pooled([mask_record(r["prob"], r["gt"], threshold=t) for r in recs])
    for key in ("iou_edited", "iou_background", "miou_diffseg30k", "miou_two_class", "f1_pooled"):
        assert row[key] == pytest.approx(direct[key], abs=1e-12), key


def test_sweep_at_half_matches_the_recorded_counts():
    recs = _records()
    row = thresholds.pixel_sweep(recs, "model", grid=(0.5,))[0]
    assert row["miou_diffseg30k"] == pytest.approx(
        pooled([r["heads"]["model"] for r in recs])["miou_diffseg30k"], abs=1e-12
    )


def test_confusion_at_splits_the_histograms():
    pos = np.array([1, 2, 3, 4])
    neg = np.array([5, 6, 7, 8])
    assert thresholds.confusion_at(pos, neg, 2) == {"tp": 7, "fn": 3, "fp": 15, "tn": 11}


def test_boundary_counts_at_half_equal_mask_record():
    prob, gt = _image(3)
    rec = mask_record(prob, gt)
    counts = boundary_counts(prob >= 0.5, gt)
    assert counts == {k: rec[k] for k in ("b_pred", "b_gt", "b_pred_hit", "b_gt_hit")}
    assert set(thresholds.boundary_grid(prob, gt)) == set(thresholds.BOUNDARY_GRID)


def test_boundary_f1_is_swept_only_on_the_boundary_grid():
    recs = _records(3)
    rows = {r["threshold"]: r for r in thresholds.pixel_sweep(recs, "model")}
    assert "mean_boundary_f1" in rows[0.5]
    assert "mean_boundary_f1" not in rows[0.55]


def test_image_sweep_rates_and_tile_bins():
    edited = [{"p_edited": {"m": p}, "tiles": 1} for p in (0.2, 0.6, 0.9)]
    real = [
        {"p_edited": {"m": 0.1}, "tiles": 1},
        {"p_edited": {"m": 0.55}, "tiles": 1},
        {"p_edited": {"m": 0.7}, "tiles": 16},
        {"p_edited": {"m": 0.3}, "tiles": 16},
    ]
    rows = thresholds.image_sweep(edited, real, "m", grid=(0.5,), tile_bin=locdiag.tile_bin)
    r = rows[0]
    assert r["edited_called_edited"] == pytest.approx(2 / 3)
    assert r["real_called_edited"] == pytest.approx(2 / 4)
    assert r["real_called_edited_by_tiles"] == {"1": 0.5, "10-16": 0.5}


def test_best_reports_the_argmax_and_the_default():
    sweep = [
        {"threshold": 0.3, "miou_diffseg30k": 0.6},
        {"threshold": 0.5, "miou_diffseg30k": 0.65},
        {"threshold": 0.7, "miou_diffseg30k": 0.7},
        {"threshold": 0.9, "miou_diffseg30k": None},
    ]
    b = thresholds.best(sweep)
    assert (b["threshold"], b["value"], b["at_default"]) == (0.7, 0.7, 0.65)
    assert thresholds.best([])["threshold"] is None


def test_diagnosis_sweeps_block():
    recs = _records(2)
    for r in recs:
        r["p_edited"] = {"model": 0.8}
        r["tiles"] = 1
    real = [{"p_edited": {"model": 0.2}, "tiles": 4}]
    out = thresholds.diagnosis_sweeps(recs, real, ["model"], tile_bin=locdiag.tile_bin)
    assert len(out["model"]["pixel"]) == len(thresholds.PIXEL_GRID)
    assert out["model"]["best"]["miou_diffseg30k"]["default_threshold"] == 0.5
    rows = {r["threshold"]: r for r in out["model"]["image"]}
    assert rows[0.05]["real_called_edited_by_tiles"] == {"2-4": 1.0}
    assert rows[0.5]["real_called_edited_by_tiles"] == {"2-4": 0.0}
