"""CP4 fix: interior-tile metrics and the comparison with the CP3 model (17 Nov 2026)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from eviground.eval import fixcompare, locdiag  # noqa: E402


def _image():
    gt = np.zeros((8, 12), bool)
    gt[:, :4] = True  # tile 0 fully edited
    gt[:4, 4:8] = True  # tile 1 half edited
    boxes = [(0, 0, 4, 8), (4, 0, 8, 8), (8, 0, 12, 8)]
    probs = np.array([[0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    pred = np.zeros_like(gt, float)
    pred[:4, :4] = 0.9  # half of the interior tile found
    pred[:4, 4:8] = 0.9  # all of the rest found
    return pred, gt, boxes, probs


def test_interior_record_on_a_hand_made_map():
    pred, gt, boxes, probs = _image()
    r = locdiag.interior_record(pred, gt, boxes, probs)
    assert r["interior_tiles"] == 1
    assert r["interior_called_generated"] == 1  # argmax of tile 0 is "generated"
    assert r["interior_p_generated_sum"] == pytest.approx(0.7)
    assert r["edited_interior"] == 32
    assert r["edited_rest"] == 16
    assert r["missed_interior"] == 16
    assert r["missed_rest"] == 0


def test_interior_summary_pools_images():
    pred, gt, boxes, probs = _image()
    rec = {"interior": {"model": locdiag.interior_record(pred, gt, boxes, probs)}}
    empty = {"interior": {"model": locdiag.interior_record(pred * 0, gt * False, boxes, probs)}}
    s = locdiag.interior_summary([rec, empty], "model")
    assert s["images"] == 2
    assert s["interior_tiles"] == 1
    assert s["interior_called_generated"] == 1.0
    assert s["miss_share_interior"] == 0.5
    assert s["miss_share_rest"] == 0.0
    assert s["edited_pixels_interior_share"] == pytest.approx(32 / 48)
    assert locdiag.interior_summary([], "model")["interior_tiles"] == 0


def _rows(sel, auc):
    return [
        {"epoch": "0", "val_selection": "0.5", "val_macro_auc": "0.9"},
        {
            "epoch": "1",
            "val_selection": sel,
            "val_macro_auc": auc,
            "val_sidset_generated_called_edited": "0.1",
        },
    ]


def test_compare_runs_takes_each_best_epoch():
    out = fixcompare.compare_runs(_rows("0.8", "0.99"), _rows("0.9", "0.98"))
    assert out["best_epoch"] == {"cp3": 1.0, "fix": 1.0}
    auc = out["metrics"]["macro AUC (3 classes)"]
    assert auc == {"cp3": 0.99, "fix": 0.98, "change": pytest.approx(-0.01)}
    assert out["metrics"]["pooled edited IoU (crops)"]["change"] is None


def test_compare_diagnoses_by_base():
    def blk(fp):
        return {"miou_diffseg30k": 0.4, "fp_share_of_pixels": fp, "fn_share_of_edited": 0.19}

    def diag(fp_ai, fp_real):
        by_base = {"ai": blk(fp_ai), "real": blk(fp_real)}
        return {"edited": {"model": {"all": blk(0.29), "by_base_type": by_base}}}

    out = fixcompare.compare_diagnoses(diag(0.359, 0.049), diag(0.12, 0.05))
    over = "over-marking (share of all pixels)"
    assert out["ai_base"][over]["change"] == pytest.approx(-0.239)
    assert out["real_base"][over]["change"] == pytest.approx(0.001)
    assert out["all"]["misses (share of edited pixels)"]["change"] == 0
