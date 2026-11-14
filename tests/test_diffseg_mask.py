"""DiffSeg30k mask reading and CP1's crop replacement (decision of 14 Nov 2026)."""

from __future__ import annotations

import numpy as np
import pytest

from eviground.data import cp1, crops, diffseg_mask

SHARE_BIN_TOTAL = crops.TRIES


def _ai(base=7, edit=3, size=512, box=(100, 100, 300, 300)):
    m = np.full((size, size), base, np.uint8)
    t, left, b, r = box
    m[t:b, left:r] = edit
    return m


def test_ai_base_read_by_border_and_count():
    info = diffseg_mask.read_base(_ai())
    assert info["base_type"] == "ai"
    assert info["base_reading"] == "agree"
    assert info["mask_base"] == 7
    assert info["edit_values"] == "3"
    edit = diffseg_mask.edit_mask(_ai(), info["mask_base"])
    assert set(np.unique(edit)) == {0, 3}
    assert int((edit > 0).sum()) == 200 * 200


def test_real_base_unedited_no_edit_and_disagreement():
    real = np.zeros((300, 300), np.uint8)
    real[10:50, 10:50] = 5
    r = diffseg_mask.read_base(real)
    assert (r["base_type"], r["base_reading"], r["mask_base"], r["edit_values"]) == (
        "real",
        "real",
        None,
        "5",
    )
    assert diffseg_mask.edit_mask(real, None) is real
    empty = diffseg_mask.read_base(np.zeros((300, 300), np.uint8))
    assert (empty["base_type"], empty["base_reading"], empty["edit_values"]) == (
        "real",
        "unedited",
        "",
    )
    single = diffseg_mask.read_base(np.full((300, 300), 8, np.uint8))
    assert (single["base_reading"], single["mask_base"], single["edit_values"]) == (
        "no_edit",
        8,
        "",
    )
    # an edit of 4 over 64% of the image: the border says 7, the pixel count says 4
    dis = diffseg_mask.read_base(_ai(base=7, edit=4, size=500, box=(50, 50, 450, 450)))
    assert dis["base_reading"] == "disagree"
    assert dis["mask_base"] is None
    assert (dis["dominant_value"], dis["border_value"]) == (4, 7)


def test_border_counts_each_border_pixel_once():
    m = np.zeros((4, 5), np.uint8)
    assert int(diffseg_mask.border_counts(m).sum()) == 2 * 5 + 2 * 2


def test_base_reading_report_and_stop_share():
    rows = [
        {
            "role": "train_pool",
            "label": "edited",
            "dataset_split": "train",
            "base_reading": "agree",
            "base_type": "ai",
            "valid_crop": True,
        },
        {
            "role": "excluded_base_unclear",
            "label": "edited",
            "dataset_split": "train",
            "base_reading": "disagree",
            "base_type": "ai",
            "valid_crop": False,
        },
        {
            "role": "test_P2",
            "label": "edited",
            "dataset_split": "validation",
            "base_reading": "real",
            "base_type": "real",
        },
        {
            "role": "test_P2",
            "label": "edited",
            "dataset_split": "validation",
            "base_reading": "agree",
            "base_type": "ai",
        },
    ]
    rep = cp1.base_reading_report(rows)
    assert rep["full_masks"] == 3
    assert rep["disagree_share_of_full"] == pytest.approx(0.3333, abs=1e-4)
    assert rep["p2"] == {"all": 2, "real_base": 1, "ai_base": 1}
    assert rep["valid_crop"] == {
        "train_split_checked": 2,
        "train_split_valid": 1,
        "train_pool_edited": 1,
        "train_pool_edited_valid": 1,
    }


def test_draw_crops_replaces_failed_edited_crops_and_describes_them(monkeypatch):
    """Edited images e0-e4 never crop (all tries 100% edited); the rest always crop."""
    eligible = [{"uid": f"g{i}", "source": "genimage", "label": "real"} for i in range(4)]
    eligible += [{"uid": f"s{i}", "source": "sidset", "label": "real"} for i in range(3)]
    eligible += [{"uid": f"c{i}", "source": "coco", "label": "real"} for i in range(3)]
    eligible += [{"uid": f"d{i}", "source": "diffseg30k", "label": "real"} for i in range(3)]
    eligible += [{"uid": f"G{i}", "source": "genimage", "label": "generated"} for i in range(4)]
    eligible += [{"uid": f"S{i}", "source": "sidset", "label": "generated"} for i in range(3)]
    eligible += [{"uid": f"e{i}", "source": "diffseg30k", "label": "edited"} for i in range(10)]
    bad = {f"e{i}" for i in range(5)}

    def fake_sample_row(task):
        r = task[0]
        out = {"uid": r["uid"], "source": r["source"], "label": r["label"]}
        if r["uid"] in bad:
            out.update(error="crop_failed", crop_shares=[1.0] * crops.TRIES)
        return out

    monkeypatch.setattr(cp1, "sample_row", fake_sample_row)
    cfg = cp1.CP1Config(workers=1, crops_per_class=40, blockiness_per_group=5)
    samples, log = cp1.draw_crops(eligible, cfg, lambda _m: None)
    ok = [s for s in samples if "error" not in s]
    for label in ("real", "generated", "edited"):
        assert sum(1 for s in ok if s["label"] == label) == 40
    assert log["first_pass_edited_drawn"] == 40
    assert log["first_pass_failed"] > 0
    assert log["failed_by_kind"] == {"all_tries_above_90pct": log["first_pass_failed"]}
    hist = log["failed_tries_share_histogram"]
    assert hist["100%"] == log["first_pass_failed"] * crops.TRIES
    assert sum(hist.values()) == log["first_pass_failed"] * crops.TRIES
    assert log["unresolved"] == 0
    assert set(log["replacements"]) == {"edited/diffseg30k"}
