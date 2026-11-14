"""DiffSeg30k mask-label diagnostic (read-only, 14 Nov 2026)."""

from __future__ import annotations

import csv
import gzip

import numpy as np
from PIL import Image

from eviground.data import mask_diagnostic as md


def _full(base=7, edit=3, size=1024, box=(100, 100, 400, 400)):
    m = np.full((size, size), base, np.uint8)
    t, left, b, r = box
    m[t:b, left:r] = edit
    return m


def _partial(h=480, w=640, value=5):
    m = np.zeros((h, w), np.uint8)
    m[100:200, 100:300] = value
    return m


def test_mask_facts_full_and_partial():
    f = md.mask_facts(_full())
    assert f["full"] is True
    assert f["size_class"] == "1024x1024"
    assert f["dominant"] == 7
    assert f["second"] == 3
    assert f["n_values"] == 2
    assert abs(f["second_share"] - 300 * 300 / 1024**2) < 1e-6
    p = md.mask_facts(_partial())
    assert p["full"] is False
    assert p["size_class"] == "both_sides_le_640"
    assert p["dominant"] == 5
    assert p["second"] is None
    e = md.mask_facts(np.zeros((300, 300), np.uint8))
    assert e["n_values"] == 0
    assert e["dominant"] is None


def test_whatif_reads_the_dominant_value_of_a_full_mask_as_background():
    m = _full()
    edited = md.whatif_edited(m, md.mask_facts(m))
    assert edited.sum() == 300 * 300
    p = _partial()
    assert (md.whatif_edited(p, md.mask_facts(p)) == (p > 0)).all()


def _manifest(tmp_path, masks):
    rows = []
    for uid, (split, role, mask) in masks.items():
        mp = tmp_path / f"{uid}.mask.png"
        ip = tmp_path / f"{uid}.image.png"
        Image.fromarray(mask).save(mp)
        Image.fromarray(np.full((*mask.shape, 3), 128, np.uint8)).save(ip)
        rows.append(
            {
                "uid": uid,
                "source": "diffseg30k",
                "dataset_split": split,
                "role": role,
                "path": ip,
                "mask_path": mp,
            }
        )
    path = tmp_path / "manifest.csv.gz"
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_diagnostic_on_full_partial_and_holdout_masks(tmp_path):
    masks = {
        "a_full_7_3": ("train", "train_pool", _full(7, 3)),
        "b_full_kolors_base": ("train", "excluded_holdout", _full(2, 5)),
        "c_full_single": ("train", "train_pool", np.full((1024, 1024), 8, np.uint8)),
        "d_partial_kolors_edit": ("validation", "test_P2", _partial(value=2)),
        "e_partial": ("train", "train_pool", _partial(value=1)),
        "f_empty": ("train", "excluded_unedited", np.zeros((480, 640), np.uint8)),
    }
    out = md.run_diagnostic(_manifest(tmp_path, masks), out_dir=tmp_path / "out")
    assert out["masks"] == 6
    full = out["full_masks"]
    assert full["count"] == 3
    assert full["n_values"] == {1: 1, 2: 2}
    assert full["dominant_value"] == {2: 1, 7: 1, 8: 1}
    assert out["partial_masks"]["count"] == 2
    assert out["empty_masks"] == 1
    assert out["per_value_dominant_vs_minority"]["7 Tencent-Hunyuan/HunyuanDiT-Diffusers"] == {
        "dominant_1024x1024": 1
    }
    kolors = out["holdout_values"]["2 kolors"]
    assert kolors == {
        "excluded_holdout | dominant_background_of_full_mask": 1,
        "test_P2 | edit_region": 1,
    }
    w = out["whatif_dominant_as_background"]
    assert w["full_masks_with_no_edit_left"] == 1  # the single-value full mask
    assert w["role_changes"] == {
        "excluded_holdout -> train_pool": 1,
        "train_pool -> excluded_unedited": 1,
    }
    assert w["train_images_checked"] == 5
    assert w["valid_crop_now"] == 1  # only the partial train mask has a 1-90% window now
    assert w["valid_crop_whatif"] == 3  # the two full masks with an edit region become valid
    assert (tmp_path / "out" / "mask_examples.png").is_file()
    assert len(out["examples"]["examples"]) == 5
