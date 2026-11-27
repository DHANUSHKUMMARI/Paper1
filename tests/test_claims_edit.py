"""CP8 Claims-Edit plan and helpers (Dhanush, 27 Nov): frozen by hash before any generation."""

import numpy as np
import pytest

from eviground.data import claims_edit as ce
from eviground.data import inpaint

_NAMES = {3: "car", 63: "couch", 62: "chair", 67: "dining table", 73: "laptop", 1: "person"}
CATS = [{"id": k, "name": v} for k, v in _NAMES.items()]


def _square(x, y, s):
    return [[x, y, x + s, y, x + s, y + s, x, y + s]]


def _world():
    images, anns = [], []
    for iid in (139, 285, 632, 724):
        images.append({"id": iid, "width": 100, "height": 100})
    # 139: a car at 25% and a chair at 9%: the car wins
    anns += [
        {
            "id": 1,
            "image_id": 139,
            "category_id": 3,
            "area": 2500,
            "segmentation": _square(10, 10, 50),
        },
        {
            "id": 2,
            "image_id": 139,
            "category_id": 62,
            "area": 900,
            "segmentation": _square(70, 70, 30),
        },
    ]
    # 285: only a couch at 50% (too large) and a person: dropped
    anns += [
        {
            "id": 3,
            "image_id": 285,
            "category_id": 63,
            "area": 5000,
            "segmentation": _square(0, 0, 70),
        },
        {
            "id": 4,
            "image_id": 285,
            "category_id": 1,
            "area": 900,
            "segmentation": _square(0, 0, 30),
        },
    ]
    # 632: two laptops of equal area: a seeded tie-break; one crowd car is ignored
    anns += [
        {
            "id": 5,
            "image_id": 632,
            "category_id": 73,
            "area": 400,
            "segmentation": _square(0, 0, 20),
        },
        {
            "id": 6,
            "image_id": 632,
            "category_id": 73,
            "area": 400,
            "segmentation": _square(50, 50, 20),
        },
        {
            "id": 7,
            "image_id": 632,
            "category_id": 3,
            "area": 3000,
            "iscrowd": 1,
            "bbox": [0, 0, 60, 50],
        },
    ]
    # 724: a chair with only a box
    anns += [{"id": 8, "image_id": 724, "category_id": 62, "area": 1200, "bbox": [5, 5, 30, 40]}]
    rows = [{"uid": f"coco/val2017/{iid:012d}.jpg"} for iid in (139, 285, 632, 724)]
    return rows, {"images": images, "annotations": anns, "categories": CATS}  # fmt: skip


def test_the_plan_follows_the_rules_and_is_deterministic():
    rows, inst = _world()
    plan = ce.plan_edits(rows, inst)
    by = {p["image_id"]: p for p in plan["plans"]}
    assert set(by) == {139, 632, 724}
    reason = "no instance of the categories with 2-40% area"
    assert plan["dropped"] == [{"uid": "coco/val2017/000000000285.jpg", "reason": reason}]
    assert by[139]["category"] == "car"
    assert by[139]["area_share"] == 0.25
    assert by[632]["ann_id"] in (5, 6)  # the crowd car is ignored
    assert by[724]["category"] == "chair"
    for p in plan["plans"]:
        assert p["prompt"] == f"realistic {p['damage']} on the {p['category']}, photo"
        if p["category"] not in ce.GLASS_OK:
            assert p["damage"] != "broken glass"
    assert ce.plan_edits(list(reversed(rows)), inst) == plan


def test_freeze_hash_detects_any_change():
    rows, inst = _world()
    frozen = ce.freeze(ce.plan_edits(rows, inst))
    assert frozen["images"] == 3
    assert frozen["dropped_count"] == 1
    assert ce.intact(frozen)
    frozen["plans"][0]["damage"] = "dents" if frozen["plans"][0]["damage"] != "dents" else "stains"
    assert not ce.intact(frozen)


def test_instance_mask_from_polygon_or_box():
    poly = ce.instance_mask({"segmentation": _square(10, 10, 50)}, 100, 100)
    assert poly.shape == (100, 100)
    assert 2400 <= poly.sum() <= 2700  # PIL fills the polygon including its border
    box = ce.instance_mask({"bbox": [5, 5, 30, 40]}, 100, 100)
    assert box.sum() == 30 * 40


def test_composite_only_changes_the_mask_and_the_drop_rule():
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), bool)
    mask[10:20, 10:30] = True
    gen = inpaint.MockInpainter().inpaint(orig, mask, "x", negative="", seed=1, steps=1, guidance=1)
    out = ce.composite(orig, gen, mask)
    assert np.array_equal(out[~mask], orig[~mask])
    assert np.array_equal(out[mask], gen[mask])
    assert ce.keep_pair(orig, out, mask) == (True, "kept")
    same = ce.keep_pair(orig, orig.copy(), mask)
    assert same[0] is False
    assert same[1].startswith("edit not visible")
    assert ce.keep_pair(orig, None, mask) == (False, "generation failed")
    with pytest.raises(ValueError, match="shapes differ"):
        ce.composite(orig, gen[:-1], mask)


def test_band_iou_is_high_for_a_border_only_prediction():
    gt = np.zeros((60, 60), bool)
    gt[15:45, 15:45] = True
    band = ce.boundary_band(gt, 4)
    assert band[15, 30]
    assert not band[30, 30]
    assert ce.band_iou(band, gt, 4) == 1.0
    assert ce.band_iou(np.zeros_like(gt), gt, 4) == 0.0


def test_eye_check_sample_is_seeded_and_sized():
    plan = {"plans": [{"uid": f"u{i:03d}"} for i in range(150)]}
    a = ce.eye_check_sample(plan, 30)
    assert len(a) == 30
    assert a == ce.eye_check_sample(plan, 30)
    assert a == sorted(a)


def test_claims_and_the_training_editor_guard():
    rows, inst = _world()
    p = ce.plan_edits(rows, inst)["plans"][0]
    m = np.ones((2, 2), bool)
    c = ce.claims(p, m)
    assert c["real"]["label"] == "real"
    assert c["edited"]["uid"] == ce.edited_uid(p["uid"])
    assert c["edited"]["label"] == "edited"
    assert "not a training editor" in inpaint.assert_not_training_editor(
        inpaint.KolorsInpainter.name
    )
    with pytest.raises(ValueError, match="training editor"):
        inpaint.assert_not_training_editor("SDXL-inpainting")


def test_kolors_size_fit():
    assert inpaint._fit(480, 640) == (768, 1024)
    assert inpaint._fit(640, 427) == (1024, 704)


def test_cp8_summary_counts_accuracy_pixels_band_and_cc():
    from eviground.eval import cp8
    from eviground.eval.tiled import mask_record

    gt = np.zeros((20, 20), bool)
    gt[5:15, 5:15] = True
    good = mask_record(gt.astype(float), gt)
    band = {k: v for k, v in mask_record(gt.astype(float), ce.boundary_band(gt, 2)).items()
            if not isinstance(v, np.ndarray)}  # fmt: skip
    recs = [
        {"uid": "r1", "label": "real", "label_pred": "real"},
        {"uid": "r2", "label": "real", "label_pred": "edited"},
        {"uid": "e1", "label": "edited", "label_pred": "edited", "mask": good, "band": band},
        {"uid": "e2", "label": "edited", "label_pred": "real", "mask": good, "band": band},
    ]
    s = cp8.cp8_summary(
        recs,
        [
            {"scored": True, "passed": True, "drop": 0.3, "random_drop": 0.1},
            {"scored": True, "passed": False, "drop": 0.0, "random_drop": 0.1},
        ],
    )
    assert s["images"] == 4
    assert s["acc"] == 0.5
    assert s["acc_real"] == 0.5
    assert s["acc_edited"] == 0.5
    assert s["confusion"]["real"] == {"real": 1, "generated": 0, "edited": 1}
    assert s["pixel"]["images"] == 2
    assert s["pixel"]["iou_edited"] == 1.0
    assert 0 < s["band_iou_mean"] < 1
    assert s["cc"]["scored"] == 2


def test_mock_inpainter_edits_only_inside_mask_and_records_request():
    from eviground.data.inpaint import MockInpainter

    image = np.full((8, 8, 3), 200, np.uint8)
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    inp = MockInpainter()
    out = inp.inpaint(image, mask, "p", negative="n", seed=3, steps=7, guidance=5.0)
    assert np.array_equal(out[~mask], image[~mask])
    assert (out[mask] != image[mask]).any()
    assert inp.last_call == {"prompt": "p", "negative": "n", "seed": 3, "steps": 7, "guidance": 5.0}


def test_generation_fingerprint_and_resume_refusal():
    fp = ce.generation_fingerprint("kolors@a", "s" * 64, {"src": "t1"})
    assert fp == ce.generation_fingerprint("kolors@a", "s" * 64, {"src": "t1"})
    assert fp != ce.generation_fingerprint("kolors@b", "s" * 64, {"src": "t1"})
    assert fp != ce.generation_fingerprint("kolors@a", "s" * 64, {"src": "t2"})
    ce.check_resume([{"uid": "a", "fingerprint": fp}], fp)
    with pytest.raises(ValueError, match="another generation configuration"):
        ce.check_resume([{"uid": "a", "fingerprint": fp}, {"uid": "b", "fingerprint": "x"}], fp)
    with pytest.raises(ValueError):
        ce.check_resume([{"uid": "c"}], fp)   # an old record without a fingerprint


def test_generation_code_covers_only_pixel_deciding_files(tmp_path):
    """An unrelated src change keeps the fingerprint (resume allowed); a claims_edit.py change
    alters it (resume refused). Dhanush, 27 Nov."""
    import subprocess

    from eviground.eval import phasec

    repo = tmp_path

    def git(*a):
        cmd = ["git", "-C", str(repo), *a]
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def commit(path, text):
        f = repo / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        git("add", "-A")
        git("commit", "-q", "-m", path)

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    for p in ce.GEN_CODE_PATHS:
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(f"# {p}\n")
    commit("src/eviground/eval/other.py", "x = 1\n")

    def fp():
        code = phasec.code_tree(repo, ce.GEN_CODE_PATHS)
        return ce.generation_fingerprint("kolors@a", "s" * 64, code)

    fp1 = fp()
    records = [{"uid": "a", "fingerprint": fp1}]
    commit("src/eviground/eval/cp9.py", "y = 2\n")   # unrelated code merged
    assert fp() == fp1
    ce.check_resume(records, fp())
    commit("src/eviground/data/claims_edit.py", "# changed\n")
    assert fp() != fp1
    with pytest.raises(ValueError):
        ce.check_resume(records, fp())
