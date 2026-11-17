"""CP4 evaluation: tiled evaluator, pixel metrics, ceilings, probes, test-file readers, the gate."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from eviground.baselines import clip_probe
from eviground.data import harmonize
from eviground.eval import cp4, locdiag, testsets, tiled

# ------------------------------------------------------------------ tiled evaluator


def _echo(x):
    """Predictor: tile classified real; mask = the red channel (so the stitched map is known)."""
    n = x.shape[0]
    return {
        "a": {
            "logits": torch.tensor([[3.0, 0.0, -3.0]]).repeat(n, 1),
            "mask_logits": (x[:, :1] - 0.5) * 40,
        },
        "b": {
            "logits": torch.tensor([[-3.0, 0.0, 3.0]]).repeat(n, 1),
            "mask_logits": torch.zeros(n, 1, *x.shape[-2:]),
        },
    }


def test_tiled_evaluator_batches_across_images_and_stitches():
    got, sizes = {}, []

    def predict(x):
        sizes.append(tuple(x.shape))
        return _echo(x)

    ev = tiled.TiledEvaluator(predict, lambda it, res: got.__setitem__(it.uid, (it, res)), batch=4)
    rng = np.random.default_rng(0)
    for i, (h, w) in enumerate([(1024, 1024), (128, 128), (300, 520), (128, 128), (100, 300)]):
        img = (rng.random((h, w, 3)) > 0.5).astype(np.uint8) * 255
        ev.add(tiled.Item(f"u{i}", img, meta={"k": i}))
    ev.close()
    assert ev.images == 5
    assert ev.max_tiles == 16
    assert ev.tiles == 16 + 1 + 6 + 1 + 2
    assert all(s[0] <= 4 for s in sizes)
    assert {s[2:] for s in sizes} == {(256, 256), (128, 128), (100, 256)}  # one queue per shape
    for it, res in got.values():
        assert res["a"]["tiles"] == len(tiled.tiling.tile_boxes(*it.image.shape[:2]))
        assert res["a"]["label"] == "real"
        assert res["b"]["label"] == "edited"
        assert np.array_equal(res["a"]["prob_map"] >= 0.5, it.image[..., 0] > 0)


def test_mask_record_counts_and_pooled_metrics_by_hand():
    gt = np.zeros((4, 4), bool)
    gt[0, :2] = True  # 2 edited pixels
    prob = np.zeros((4, 4))
    prob[0, 0] = 0.9  # tp
    prob[3, 3] = 0.7  # fp
    editors = np.zeros((4, 4), int)
    editors[0, 0] = 6  # Glide (held out)
    editors[0, 1] = 3  # a training editor
    r = tiled.mask_record(prob, gt, editors=editors, tolerance=0)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 13)
    p = tiled.pooled([r])
    assert p["iou_edited"] == pytest.approx(1 / 3)
    assert p["iou_background"] == pytest.approx(13 / 15)
    assert p["miou_two_class"] == pytest.approx((1 / 3 + 13 / 15) / 2)
    assert p["miou_diffseg30k"] == pytest.approx(1 / 3)  # background excluded
    assert p["fn_share_of_edited"] == 0.5
    # held-out-editor pixels: only (0, 0) counts as edited; (0, 1) is ignored
    assert (r["ho_tp"], r["ho_fn"], r["ho_fp"], r["ho_tn"]) == (1, 0, 1, 13)
    assert tiled.pooled([r], "ho_")["iou_edited"] == 0.5


def test_boundary_f1_and_histogram_auc():
    gt = np.zeros((32, 32), bool)
    gt[8:24, 8:24] = True
    same = tiled.per_image(tiled.mask_record(gt.astype(float), gt))
    assert same["boundary_f1"] == 1.0
    assert same["iou"] == 1.0
    shifted = np.roll(gt, 4, axis=1).astype(float)
    far = tiled.per_image(tiled.mask_record(shifted, gt, tolerance=1))
    near = tiled.per_image(tiled.mask_record(shifted, gt, tolerance=4))
    assert far["boundary_f1"] < near["boundary_f1"] <= 1.0
    assert tiled.hist_auc([0, 0, 5], [5, 0, 0]) == 1.0
    assert tiled.hist_auc([1, 1], [1, 1]) == pytest.approx(0.5)


# ------------------------------------------------------------------ diagnosis


def test_grid_ceilings():
    aligned = np.zeros((256, 256), bool)
    aligned[32:96, 64:128] = True  # on the 16-pixel cell grid
    for g, r in locdiag.ceiling_records(aligned).items():
        assert tiled.pooled([r])["miou_two_class"] == 1.0, g
    small = np.zeros((1024, 1024), bool)
    small[100:107, 200:207] = True  # 7 x 7 pixels: below half a 16-pixel cell
    c = locdiag.ceiling_records(small)
    assert tiled.pooled([c[16]])["iou_edited"] == 0.0
    assert tiled.pooled([c[64]])["iou_edited"] > 0.4


def test_bins_and_summaries():
    assert locdiag.size_bin(0.005) == "0%-1%"
    assert locdiag.size_bin(0.99) == "40%-100%"
    assert [locdiag.tile_bin(n) for n in (1, 3, 9, 16, 54)] == ["1", "2-4", "5-9", "10-16", "17+"]
    gt = np.zeros((8, 8), bool)
    gt[:2, :2] = True
    rec = tiled.mask_record(gt.astype(float), gt)
    edited = [
        {
            "share": 0.0625,
            "editors": ["1", "3"],
            "base_type": "real",
            "heads": {"m": rec},
            "ceilings": None,
        },
        {"share": 0.5, "editors": ["3"], "base_type": "ai", "heads": {"m": rec}, "ceilings": None},
    ]
    s = locdiag.summarize_edited(edited, ["m"])
    assert s["m"]["all"]["miou_two_class"] == 1.0
    assert set(s["m"]["by_editor"]) == {"1", "3"}
    assert s["m"]["by_editor"]["3"]["images"] == 2
    real = [
        {"tiles": 1, "source": "coco", "heads": {"m": {"label": "edited", "marked_share": 0.1}}},
        {"tiles": 16, "source": "coco", "heads": {"m": {"label": "real", "marked_share": 0.0}}},
    ]
    r = locdiag.summarize_real(real, ["m"])
    assert r["m"]["all"]["called_edited"] == 0.5
    assert r["m"]["by_tiles"]["10-16"]["any_marked"] == 0.0


# ------------------------------------------------------------------ probes


def test_logistic_probe_separates_and_picks_l2_on_validation():
    torch.manual_seed(0)
    x = torch.randn(400, 16)
    y = (x[:, 0] > 0).long()
    xv = torch.randn(200, 16)
    yv = (xv[:, 0] > 0).long()
    fit = clip_probe.choose_logreg(x, y, xv, yv, 2, grid=(1e-3, 10.0))
    assert fit["val_auc"] > 0.99
    assert {t["l2"] for t in fit["tried"]} == {1e-3, 10.0}
    head = clip_probe.ProbeHead(fit["w"], fit["b"], [0, 1])
    out = head(xv, torch.zeros(200, 256, 16), (64, 64))
    assert out["logits"].shape == (200, 3)
    assert float(out["logits"][:, 2].max()) == clip_probe.NEG
    assert torch.all(out["mask_logits"] == clip_probe.NEG)
    again = clip_probe.ProbeHead.from_state(head.state())
    assert torch.equal(
        again(xv, torch.zeros(200, 256, 16), (8, 8))["logits"],
        head(xv, torch.zeros(200, 256, 16), (8, 8))["logits"],
    )


def test_patch_probes_learn_and_targets_follow_the_half_rule():
    m = torch.zeros(1, 1, 256, 256)
    m[..., :8, :16] = 1  # half of the first cell -> positive
    m[..., 16:23, 16:32] = 1  # 7/16 of a cell -> negative
    t = clip_probe.patch_targets(m)
    assert t.shape == (1, 256)
    assert t[0, 0] == 1
    assert t[0, 17] == 0
    torch.manual_seed(0)
    probes = clip_probe.PatchProbes(8, l2s=(0.0, 1.0))
    opt = torch.optim.AdamW(probes.parameters(), lr=0.05, weight_decay=0)
    for _ in range(200):
        feats = torch.randn(4, 64, 8)
        tgt = (feats[..., 0] > 0).float()
        opt.zero_grad()
        probes.loss(feats, tgt).backward()
        opt.step()
    feats = torch.randn(4, 64, 8)
    acc = ((probes(feats)[0] > 0) == (feats[..., 0] > 0)).float().mean()
    assert acc > 0.95
    head = clip_probe.ProbeHead(
        torch.zeros(8, 3), torch.zeros(3), [0, 1, 2], probes.w[0][:, None], probes.b[0][None]
    )
    assert head(torch.zeros(4, 8), feats, (256, 256))["mask_logits"].shape == (4, 1, 256, 256)


# ------------------------------------------------------------------ test files


def _png(h, w, seed=0):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(
        buf, "PNG"
    )
    return buf.getvalue()


def _jpeg(h, w):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_p1_harmonized_compresses_generated_pngs_only_and_skips_nothing():
    dist = harmonize.QualityDistribution.measure(
        [{"format": "jpeg", "jpeg_quality": 96, "subsampling": "4:2:0"}] * 10
    )
    small = {"uid": "genimage_val/biggan/ai/1.png", "source": "genimage_val", "label": "generated"}
    out = testsets.p1_harmonized_bytes(small, _png(128, 128), dist)  # under 256: still compressed
    assert out[:2] == b"\xff\xd8"
    real = {"uid": "genimage_val/biggan/nature/1.jpeg", "source": "genimage_val", "label": "real"}
    raw = _jpeg(375, 500)
    assert testsets.p1_harmonized_bytes(real, raw, dist) == raw


def test_file_items_read_p2_masks_with_editor_values_and_stage_files(tmp_path):
    img = tmp_path / "a.image.png"
    img.write_bytes(_png(40, 60))
    mask = np.zeros((40, 60), np.uint8)
    mask[:10, :10] = 6  # Glide edit
    mask[20:, :] = 8  # AI base value, read as 0
    mask[30:, 50:] = 3  # a training editor's edit on that base
    Image.fromarray(mask).save(tmp_path / "a.mask.png")
    row = {
        "uid": "diffseg30k/validation/a",
        "role": "test_P2",
        "label": "edited",
        "source": "diffseg30k",
        "path": str(img),
        "mask_path": str(tmp_path / "a.mask.png"),
        "mask_base": "8",
        "editors": "3,6",
    }
    staged = testsets.stage_files([row], tmp_path / "local")
    assert staged[0]["path"].startswith(str(tmp_path / "local"))
    assert testsets.stage_files([row], tmp_path / "local")[0]["path"] == staged[0]["path"]
    (item,) = testsets.file_items(staged)
    assert item.image.shape == (40, 60, 3)
    assert item.mask.sum() == 100 + 100
    assert set(np.unique(item.editors)) == {0, 3, 6}
    assert item.meta["editors"] == ["3", "6"]


# ------------------------------------------------------------------ the gate


def _p1(head, version, acc_by_gen):
    recs = []
    for gen, acc in acc_by_gen.items():
        n = 20
        right = int(acc * n)
        for i in range(n):
            label = "generated" if i % 2 else "real"
            correct = i < right
            p_real = (
                (0.9 if label == "real" else 0.1) if correct else (0.1 if label == "real" else 0.9)
            )
            recs.append(
                {
                    "head": head,
                    "version": version,
                    "generator": gen,
                    "label": label,
                    "p_real": p_real,
                    "label_pred": "real",
                    "tiles": 4,
                }
            )
    return recs


GENS = ["adm", "biggan", "glide", "midjourney", "stable_diffusion_v_1_4", "sd15", "vqdm", "wukong"]


def _table(model_acc, probe_acc, model_harm=None):
    recs = []
    for v, ma, pa in (
        ("raw", model_acc, probe_acc),
        ("harmonized", model_harm or model_acc, probe_acc),
    ):
        recs += _p1("detection", v, dict.fromkeys(GENS, ma))
        recs += _p1("detection_probe", v, dict.fromkeys(GENS, pa))
    return cp4.p1_table(recs)


def test_gate_pass_fail_and_red_flag():
    ok = cp4.evaluate(_table(0.85, 0.75))
    assert ok["checks"]["G1"]["status"] == "PASS"
    assert ok["verdict"] == "PENDING_P2"
    assert ok["checks"]["G2"]["status"] == "NOT_RUN"
    assert cp4.evaluate(_table(0.85, 0.75), p2={"miou_diffseg30k": 0.81})["verdict"] == "PASS"
    assert cp4.evaluate(_table(0.85, 0.75), p2={"miou_diffseg30k": 0.66})["verdict"] == "FAIL"
    # G2 reads the DiffSeg30k mIoU (edited class only), not the two-class mean
    two_class_only = {"miou_two_class": 0.9, "miou_diffseg30k": 0.7}
    g2 = cp4.evaluate(_table(0.85, 0.75), p2=two_class_only)
    assert g2["verdict"] == "FAIL"
    assert cp4.evaluate(_table(0.70, 0.60))["checks"]["G1"]["status"] == "FAIL"  # under 0.721
    harm = cp4.evaluate(_table(0.85, 0.75, model_harm=0.70))  # loses to the probe on harmonized
    assert harm["checks"]["G1"]["status"] == "FAIL"
    rf = cp4.evaluate(_table(1.0, 0.75))
    assert rf["checks"]["RF"]["raised"]
    assert rf["verdict"] == "STOP_AND_ASK"
    assert "stable_diffusion_v_1_4" not in rf["checks"]["RF"]["unseen_acc"]


def test_average_precision_and_false_edited_bins():
    assert cp4.average_precision([True, False, True], [0.9, 0.8, 0.7]) == pytest.approx(
        (1 + 2 / 3) / 2
    )
    recs = [
        {"head": "unified", "label": "real", "label_pred": "edited", "tiles": 16},
        {"head": "unified", "label": "real", "label_pred": "real", "tiles": 16},
        {"head": "unified", "label": "real", "label_pred": "real", "tiles": 1},
        {"head": "unified", "label": "generated", "label_pred": "edited", "tiles": 1},
    ]
    fe = cp4.false_edited(recs, "unified")
    assert fe["images"] == 3
    assert fe["by_tiles"]["10-16"]["rate"] == 0.5


def test_boundary_uses_four_neighbours_and_not_the_image_frame():
    from eviground.eval.tiled import _boundary

    plus = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    plus[..., 2, 1:4] = True
    plus[..., 1:4, 2] = True
    b = _boundary(plus)[0, 0]
    assert not b[2, 2]  # its outside neighbours are diagonal only
    assert int(b.sum()) == 4
    full = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    assert not _boundary(full).any()
    half = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    half[..., :, :2] = True
    assert _boundary(half)[0, 0].nonzero()[:, 1].unique().tolist() == [1]


def test_diffseg30k_area_rule_in_the_summaries():
    """DiffSeg30k image rule: edited when more than 1% of the image is predicted edited."""
    from eviground.eval import locdiag, runner

    assert not tiled.area_rule(0.01) and tiled.area_rule(0.0101)
    gt = np.zeros((10, 10), bool)
    gt[:5, :5] = True
    small = np.zeros((10, 10), np.float32)
    small[0, 0] = 0.9  # 1 pixel of 100 predicted: exactly 1%, not more
    big = np.zeros((10, 10), np.float32)
    big[:2, :2] = 0.9  # 4%
    recs = [
        {"heads": {"m": tiled.mask_record(p, gt)}, "labels": {"m": "edited"}, "share": 0.25}
        for p in (small, big)
    ]
    assert tiled.per_image(recs[1]["heads"]["m"])["pred_share"] == pytest.approx(0.04)
    assert runner.p2_summary(recs, "m")["all"]["image_acc_edited_area_rule"] == 0.5
    edited = locdiag.summarize_edited(
        [{**r, "editors": ["x"], "base_type": "real"} for r in recs], ["m"]
    )
    assert edited["m"]["all"]["called_edited_area_rule"] == 0.5
    real = [
        {"tiles": 1, "source": "coco", "heads": {"m": {"label": "real", "marked_share": s}}}
        for s in (0.0, 0.005, 0.02, 0.5)
    ]
    assert locdiag.summarize_real(real, ["m"])["m"]["all"]["called_edited_area_rule"] == 0.5
