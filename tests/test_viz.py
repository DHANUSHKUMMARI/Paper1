"""Seaborn figures drawn from saved result files (Dhanush, 17 Nov 2026)."""

from __future__ import annotations

import csv
import json

import pytest

from eviground import viz
from eviground.eval import cp4, locdiag


def _edited_csv(path):
    cols = [
        "uid",
        "share",
        "editors",
        "base_type",
        "tiles",
        "model_iou",
        "model_f1",
        "model_boundary_f1",
        "probe_iou",
        "probe_f1",
        "ceiling16_iou",
        "ceiling64_iou",
    ]
    rows = [
        ["a", 0.005, "sdxl", "real", 4, 0.1, 0.2, 0.1, 0.05, 0.1, 0.3, 0.6],
        ["b", 0.03, "sdxl|flux", "ai", 4, 0.4, 0.5, 0.3, 0.2, 0.3, 0.7, 0.9],
        ["c", 0.5, "kandinsky", "real", 16, 0.8, 0.9, 0.7, 0.6, 0.7, 0.95, 0.99],
        ["d", 0.2, "", "ai", 9, 0.5, 0.6, "", "", "", 0.9, 0.97],  # probe IoU missing
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    return path


def _result():
    curves = {
        "val_mask_f1": [0.3, None, 0.5],
        "val_mask_iou_pooled": [0.2, 0.3, 0.4],
        "val_real_any_pred": [0.5, 0.2, 0.1],
    }
    pixel = [
        {"threshold": t, "miou_diffseg30k": t / 2, "f1_pooled": t / 3} for t in (0.3, 0.5, 0.7)
    ]
    pixel[1]["mean_boundary_f1"] = 0.2
    image = [
        {"threshold": t, "edited_called_edited": 1 - t, "real_called_edited": t / 4}
        for t in (0.3, 0.5, 0.7)
    ]
    sweep = {h: {"pixel": pixel, "image": image} for h in ("model", "probe")}
    return {"curves": curves, "threshold_sweep": sweep}


def test_curve_rows_split_panels_and_skip_missing_epochs():
    rows = viz.curve_rows(_result()["curves"])
    f1 = [r for r in rows if r["metric"] == viz.MASK_CURVES["val_mask_f1"]]
    assert [(r["epoch"], r["value"]) for r in f1] == [(1, 0.3), (3, 0.5)]
    assert {r["panel"] for r in rows if r["metric"] == viz.REAL_CURVES["val_real_any_pred"]} == {
        "real"
    }
    assert len(rows) == 2 + 3 + 3


def test_sweep_rows_name_heads_and_keep_boundary_f1_where_measured():
    rows = viz.sweep_rows(_result()["threshold_sweep"], ["model", "probe"])
    assert {r["head"] for r in rows} == {"EviGround", "CLIP patch probe"}
    bf1 = [r for r in rows if r["metric"] == viz.PIXEL_SWEEP["mean_boundary_f1"]]
    assert [r["threshold"] for r in bf1] == [0.5, 0.5]
    assert len([r for r in rows if r["kind"] == "image"]) == 2 * 3 * 2


def test_per_image_rows_explode_editors_and_skip_missing_values(tmp_path):
    rows = viz.per_image_rows(_edited_csv(tmp_path / "e.csv"))
    b_model = [r for r in rows if r["uid"] == "b" and r["predictor"] == "EviGround"]
    assert sorted(r["editor"] for r in b_model) == ["flux", "sdxl"]
    assert not [r for r in rows if r["uid"] == "d" and r["predictor"] == "CLIP patch probe"]
    assert {r["editor"] for r in rows if r["uid"] == "d"} == {"unknown"}
    assert {r["size_bin"] for r in rows} == {
        locdiag.size_bin(0.005),
        locdiag.size_bin(0.03),
        locdiag.size_bin(0.5),
        locdiag.size_bin(0.2),
    }
    assert viz.SIZE_ORDER[0] == locdiag.size_bin(0.0)


def _p1_table():
    recs = []
    for head, shift in (("detection", 0.3), ("detection_probe", 0.1)):
        for version in ("raw", "harmonized"):
            for gen in ("adm", "midjourney"):
                for i in range(10):
                    fake = i % 2 == 0
                    p_real = 0.5 - shift if fake else 0.5 + shift
                    recs.append(
                        {
                            "head": head,
                            "version": version,
                            "generator": gen,
                            "label": "generated" if fake else "real",
                            "p_real": p_real,
                        }
                    )
    return cp4.p1_table(recs)


def test_p1_rows_one_per_head_version_generator():
    rows = viz.p1_rows(_p1_table())
    assert len(rows) == 2 * 2 * 2
    assert {r["acc"] for r in rows} == {1.0}


def test_figures_are_drawn_from_saved_files(tmp_path):
    pytest.importorskip("seaborn")
    result_file = tmp_path / "localization_diagnosis_seed42.json"
    result_file.write_text(json.dumps(_result()))  # as the notebook saves it, read back
    saved = json.loads(result_file.read_text())
    out = tmp_path / "figures"
    figs = viz.diagnosis_figures(saved, _edited_csv(tmp_path / "e.csv"), out, show=False)
    assert [f.name for f in figs] == [
        "diagnosis_curves_seed42.png",
        "diagnosis_threshold_sweep_seed42.png",
        "diagnosis_iou_by_size_seed42.png",
        "diagnosis_iou_by_editor_seed42.png",
    ]
    assert all(f.stat().st_size > 0 and f.with_suffix(".pdf").is_file() for f in figs)
    p1 = viz.p1_figure(_p1_table(), out, show=False)
    assert p1 == out / "p1_dev_accuracy_seed42.png"
    assert p1.is_file()
    assert p1.with_suffix(".pdf").is_file()
    assert viz.p1_figure({}, out, show=False) is None
