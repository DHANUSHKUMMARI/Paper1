"""CP5 explanation code: regions, evidence bundles, faithfulness (hand-made maps) and Grad-CAM++
on the tiny model. CPU only."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from eviground.explain import evidence, faithfulness, regions  # noqa: E402


def _square(h=64, w=64, box=(16, 16, 32, 32)):
    m = np.zeros((h, w), dtype=bool)
    m[box[1] : box[3], box[0] : box[2]] = True
    return m


def test_components_drop_tiny_ones_and_report_boxes():
    m = _square()
    m[0, 0] = True  # 1 pixel: under 0.5% of 64*64
    comps = regions.components(m)
    assert len(comps) == 1
    assert regions.bbox(comps[0]) == [16, 16, 16, 16]
    stats = regions.region_stats(
        comps[0], m.astype(float), m.astype(float), regions.image_maps(np.random.rand(64, 64, 3))
    )
    assert stats["area_pct"] == pytest.approx(100 * 256 / 4096)
    assert stats["mask_conf"] == 1.0
    assert stats["attr_mass"] == pytest.approx(256 / 257)


def test_a_sparse_map_gives_regions_without_the_zero_background():
    attr = np.zeros((64, 64))
    attr[10:20, 10:20] = 1.0  # 2.4% of the pixels: the 90th percentile is 0
    comps = regions.top_attribution_regions(attr)
    assert len(comps) == 1
    assert comps[0].sum() == 100


def test_rle_round_trip():
    for m in (_square(), ~_square(), np.zeros((5, 7), bool), np.ones((3, 3), bool)):
        assert (regions.rle_decode(regions.rle_encode(m)) == m).all()


def test_noise_anomaly_sees_a_noisy_region():
    rng = np.random.default_rng(0)
    img = np.full((64, 64), 0.5)
    m = _square()
    img[m] += rng.normal(0, 0.2, m.sum())
    maps = regions.image_maps(np.clip(img, 0, 1))
    assert regions.region_stats(m, m.astype(float), m.astype(float), maps)["noise_anomaly"] > 10


def test_epg_and_area_fraction():
    gt = _square()
    assert faithfulness.epg(gt.astype(float), gt) == 1.0
    assert faithfulness.epg(np.ones((64, 64)), gt) == pytest.approx(faithfulness.area_fraction(gt))


def test_remove_region_changes_only_the_region():
    img = torch.rand(3, 64, 64)
    m = _square()
    out = faithfulness.remove_region(img, m)
    assert torch.equal(out[:, ~torch.as_tensor(m)], img[:, ~torch.as_tensor(m)])
    assert not torch.equal(out[:, torch.as_tensor(m)], img[:, torch.as_tensor(m)])


def test_deletion_beats_a_random_order_for_a_perfect_map():
    gt = _square()
    img = torch.rand(3, 64, 64)
    img[:, torch.as_tensor(gt)] = 1.0  # the "edit": a bright square

    def score(x):  # the edited-class probability: how much of the square is still bright
        return (x[:, :, 16:32, 16:32] > 0.99).float().mean(dim=(1, 2, 3))

    good = faithfulness.deletion_insertion(score, img, gt.astype(float), steps=10)
    rand = faithfulness.random_order_auc(score, img, gt.shape, steps=10)
    assert good["deletion_auc"] < rand["deletion_auc"]
    assert good["insertion_auc"] > rand["insertion_auc"]


def test_best_tile_and_ssim():
    m = np.zeros((256, 512), bool)
    m[:, 300:] = True
    assert faithfulness.best_tile(m, [(0, 0, 256, 256), (256, 0, 512, 256)]) == (256, 0, 512, 256)
    a = np.random.default_rng(0).random((32, 32))
    assert faithfulness.ssim(a, a) == pytest.approx(1.0)
    assert faithfulness.ssim(a, np.random.default_rng(1).random((32, 32))) < 0.5


def test_evidence_record_by_verdict():
    img = np.random.default_rng(0).random((64, 64, 3))
    mask = _square().astype(float)
    attr = mask * 0.9 + 0.01
    edited, rles = evidence.evidence_record(
        {"real": 0.1, "generated": 0.1, "edited": 0.8}, 0.4, mask, attr, img
    )
    assert edited["verdict"] == "locally_edited"
    assert [r["id"] for r in edited["regions"]] == ["E1"]
    assert set(rles) == {"E1"}
    assert edited["global"][0] == {
        "id": "G1",
        "type": "branch_reliance",
        "semantic": 0.4,
        "trace": 0.6,
    }
    real, _ = evidence.evidence_record(
        {"real": 0.9, "generated": 0.05, "edited": 0.05}, 0.5, mask, attr, img
    )
    assert real["regions"] == []
    gen, _ = evidence.evidence_record(
        {"real": 0.1, "generated": 0.8, "edited": 0.1}, 0.5, mask, attr, img
    )
    assert 1 <= len(gen["regions"]) <= 3
    assert all(r["id"].startswith("A") for r in gen["regions"])
    json.dumps(edited)  # the LLM input is plain JSON
    assert "counts" not in json.dumps(edited)  # no region masks in it


def test_bundle_round_trip(tmp_path):
    img = np.random.default_rng(0).random((64, 64, 3))
    mask = _square().astype(float)
    probs = {"real": 0.1, "generated": 0.1, "edited": 0.8}
    rec, rles = evidence.evidence_record(probs, 0.4, mask, mask, img)
    evidence.save_bundle(
        tmp_path, "diffseg30k/val/1.png", rec, rles, {"attr": mask, "mask": mask}, probs, 0.4
    )
    got = evidence.load_bundle(tmp_path, "diffseg30k/val/1.png")
    assert got["uid"] == "diffseg30k/val/1.png"
    assert got["evidence"] == rec
    assert (got["regions"]["E1"] == _square()).all()
    assert got["maps"]["attr"].shape == (16, 16)
    assert got["maps"]["attr"].dtype == np.float16


# --------------------------------------------------------------------------------------
# the tiny model
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("timm")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from eviground.config import load_config
    from eviground.models.eviground import build_model

    torch.manual_seed(0)
    m = build_model(load_config("debug").model)
    m.eval()
    return m


def test_channel_weights_follow_the_gradcam_pp_formula():
    from eviground.explain import gradcam

    gen = torch.Generator().manual_seed(0)
    a, g = torch.rand(2, 5, 3, generator=gen), torch.randn(2, 5, 3, generator=gen)
    want = torch.zeros(2, 3)
    for b in range(2):
        for k in range(3):
            total = a[b, :, k].sum()
            for i in range(5):
                gi = g[b, i, k]
                alpha = gi**2 / (2 * gi**2 + total * gi**3)
                want[b, k] += alpha * torch.relu(gi)
    assert torch.allclose(gradcam.channel_weights(a, g), want, atol=1e-6)


def test_gradcam_is_raw_and_non_negative(model):
    from eviground.explain import gradcam

    cam = gradcam.gradcam_pp(model, torch.rand(2, 3, 256, 256), 2)
    assert cam.shape == (2, 256, 256)
    assert float(cam.min()) >= 0
    tiled = gradcam.tiled_map(model, torch.rand(3, 300, 520), 2)
    assert tiled.shape == (300, 520)
    assert float(tiled.max()) == pytest.approx(1.0)


def test_a_blank_tile_stays_near_zero_after_stitching(monkeypatch):
    from eviground.explain import gradcam

    def fake(model, crops, cls):  # tile 0: strong attribution; tile 1: almost none
        out = torch.zeros(crops.shape[0], crops.shape[2], crops.shape[3])
        out[0] = 5.0
        if crops.shape[0] > 1:
            out[1] = 0.01
        return out

    monkeypatch.setattr(gradcam, "gradcam_pp", fake)
    m = gradcam.tiled_map(None, torch.rand(3, 256, 512), 2)
    assert float(m[:, :256].mean()) == pytest.approx(1.0)
    assert float(m[:, 256:].max()) < 0.01  # per-tile normalization would have made it 1


def test_randomization_changes_the_map(model):
    from eviground.explain import gradcam

    x = torch.rand(1, 3, 256, 256)
    steps = faithfulness.randomization_check(model, lambda m: gradcam.gradcam_pp(m, x, 2)[0])
    assert [s["step"] for s in steps] == ["heads", "fusion", "trace", "semantic_lora"]
    assert all(s["ssim"] < 1.0 for s in steps)
