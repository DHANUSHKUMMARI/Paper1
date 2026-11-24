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


# --------------------------------------------------------------------------------------
# CP5 (notebook 14)
# --------------------------------------------------------------------------------------


def _records(n=20, epg=0.6, area=0.2, dele=0.2, rnd=0.5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = {"area_fraction": area + rng.normal(0, 0.01)}
        for m in ("gradcam", "decoder"):
            r[f"epg_{m}"] = epg + rng.normal(0, 0.02)
            r[f"deletion_{m}"] = dele + rng.normal(0, 0.02)
            r[f"insertion_{m}"] = 0.7
        r["deletion_random"] = rnd + rng.normal(0, 0.02)
        r["insertion_random"] = 0.5
        out.append(r)
    return out


def _rand(ssim_last):
    steps = [
        {"step": s, "ssim_mean": v, "ssim": [v] * 3}
        for s, v in (("heads", 0.8), ("all", ssim_last))
    ]
    return {"gradcam": steps, "decoder": steps}


def test_cp5_verdict_rules():
    from eviground.explain import cp5

    ok = cp5.cp5_verdict(_records(), _rand(0.2))
    assert ok["verdict"] == "PASS"
    assert ok["primary_map"] == "gradcam"
    assert ok["secondary_map"] == "decoder"
    assert ok["maps"]["gradcam"]["epg"]["ratio"] == pytest.approx(3.0, rel=0.1)
    # EPG only 1.25x the area fraction: Grad-CAM++ fails the ratio; the decoder takes over
    recs = _records()
    for r in recs:
        r["epg_gradcam"] = 1.25 * r["area_fraction"] + 0.001
    fb = cp5.cp5_verdict(recs, _rand(0.2))
    assert fb["verdict"] == "FALLBACK"
    assert fb["primary_map"] == "decoder"
    assert fb["secondary_map"] == "gradcam"
    assert not fb["maps"]["gradcam"]["epg"]["pass"]
    # too few images for p < 0.01 although the ratio is high
    few = cp5.cp5_verdict(_records(n=4), _rand(0.2))
    assert few["maps"]["gradcam"]["epg"]["wilcoxon_p"] >= 0.01
    assert few["verdict"] == "FAIL"
    # maps that do not change under randomization fail for both maps
    assert cp5.cp5_verdict(_records(), _rand(0.9))["verdict"] == "FAIL"
    # deletion no better than random
    assert not cp5.cp5_verdict(_records(dele=0.6), _rand(0.2))["maps"]["gradcam"]["deletion"][
        "pass"
    ]


def test_cp5_summarize_and_overlay(tmp_path):
    from eviground.explain import cp5

    s = cp5.summarize([{"part": "a", "x": 1.0}, {"part": "a", "x": 3.0}, {"part": "b", "x": 5.0}])
    assert s["a"] == {"images": 2, "x": 2.0}
    assert s["all"]["x"] == 3.0
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    rec = {"regions": [{"id": "E1", "bbox": [16, 16, 16, 16]}]}
    maps = {"decoder": _square().astype(float), "gradcam_pred": _square().astype(float)}
    path = cp5.overlay_figure(
        [{"image": img, "mask": _square(), "maps": maps, "record": rec}], tmp_path / "o.png"
    )
    assert path.stat().st_size > 1000


def test_cp5_on_the_tiny_model(model):
    from eviground.explain import cp5

    img = (np.random.default_rng(0).random((300, 400, 3)) * 255).astype(np.uint8)
    out = cp5.explain_image(model, img, torch.device("cpu"))
    assert 0.0 <= out["gate"] <= 1.0
    assert out["verdict"] in ("real", "generated", "edited")
    for k in ("decoder", "gradcam", "gradcam_pred"):
        assert out[k].shape == (300, 400)
    mask = np.zeros((300, 400), bool)
    mask[50:200, 100:300] = True
    rec = cp5.faithfulness_record(
        model,
        img,
        mask,
        {"gradcam": out["gradcam"], "decoder": out["decoder"]},
        torch.device("cpu"),
        steps=4,
    )
    assert rec["area_fraction"] == pytest.approx(mask.mean())
    assert {"epg_gradcam", "deletion_decoder", "insertion_random"} <= set(rec)
    tiles_x = torch.rand(2, 3, 256, 256)
    rnd = cp5.randomization(model, tiles_x)
    assert [s["step"] for s in rnd["gradcam"]] == ["heads", "fusion", "trace", "semantic_lora"]
    assert len(rnd["decoder"][-1]["ssim"]) == 2


# --------------------------------------------------------------------------------------
# Token-level deletion (notebook 15)
# --------------------------------------------------------------------------------------


class _TokenModel(torch.nn.Module):
    """P(edited) rises with the number of untouched 'edited' trace tokens (value 1.0)."""

    def __init__(self, edited, flat=False):
        super().__init__()
        self.edited, self.flat = edited, flat

    def branch_tokens(self, x):
        t = torch.zeros(x.shape[0], 256, 4)
        t[:, self.edited, 0] = 1.0
        return torch.zeros(x.shape[0], 257, 4), t

    def logits_from_tokens(self, s, t):
        if self.flat:
            return torch.zeros(s.shape[0], 3)
        k = (t[:, :, 0] > 0.99).float().sum(dim=1)
        return torch.stack([torch.zeros_like(k), torch.zeros_like(k), k / 10.0], dim=1)


def test_token_deletion_oracle_beats_random_and_flat_head_is_uninformative():
    from eviground.explain import cp5

    edited = np.arange(100)  # the first 100 tokens hold the edit
    oracle = np.zeros(256)
    oracle[edited] = 1.0
    recs = []
    for seed in range(12):
        recs.append(
            cp5.token_deletion(
                _TokenModel(edited),
                torch.zeros(3, 256, 256),
                {
                    "oracle": oracle,
                    "decoder": oracle + np.random.default_rng(seed).random(256) * 0.1,
                    "gradcam": np.random.default_rng(seed).random(256),
                },
                steps=10,
                seed=seed,
            )
        )
    assert recs[0]["token_oracle"] < recs[0]["token_random"]
    v = cp5.token_deletion_verdict(recs)
    assert v["validity"]["valid"]
    assert v["maps"]["decoder"]["verdict"] == "PASS"
    assert v["maps"]["gradcam"]["verdict"] == "FAIL"
    flat = [
        cp5.token_deletion(
            _TokenModel(edited, flat=True),
            torch.zeros(3, 256, 256),
            {"oracle": oracle, "decoder": oracle, "gradcam": oracle},
            steps=4,
            seed=s,
        )
        for s in range(5)
    ]
    vf = cp5.token_deletion_verdict(flat)
    assert not vf["validity"]["valid"]
    assert {m["verdict"] for m in vf["maps"].values()} == {"UNINFORMATIVE"}


def test_token_scores_and_spearman():
    from eviground.explain import cp5

    m = np.zeros((256, 256))
    m[:16, :16] = 1.0
    sc = cp5.token_scores(m)
    assert sc.shape == (256,)
    assert sc[0] == 1.0
    assert sc[1:].sum() == 0.0
    a = np.random.default_rng(0).random((32, 32))
    assert cp5.spearman(a, a) == pytest.approx(1.0)
    assert np.isnan(cp5.spearman(a, np.ones((32, 32))))


def test_bundle_tile_restores_the_window():
    from eviground.explain import cp5

    full = np.zeros((300, 520))
    full[:, 300:] = 1.0
    small = full[:300, :520].reshape(75, 4, 130, 4).mean(axis=(1, 3))
    tile = cp5.bundle_tile(small, (264, 44, 520, 300))
    assert tile.shape == (256, 256)
    assert tile[:, :36].max() == 0.0
    assert tile[:, 36:].min() == 1.0


def test_aspect_sensitivity_reports_as_run_and_without_the_excluded():
    rng = np.random.default_rng(0)
    p6 = [
        {
            "uid": f"u{i}",
            "part": "sidset_tampered" if i < 4 else "brgen",
            "area_fraction": "0.2",
            "epg_gradcam": str(0.6 if i else 0.0),
            "epg_decoder": "0.4",
        }
        for i in range(8)
    ] + [{"uid": "real", "part": "real", "area_fraction": "", "epg_gradcam": ""}]
    tok = [
        {
            "uid": f"u{i}",
            "part": "sidset_tampered" if i < 4 else "brgen",
            "token_oracle": 0.3 + 0.01 * rng.random(),
            "token_random": 0.6,
            "token_gradcam": 0.4,
            "token_decoder": 0.5,
        }
        for i in range(8)
    ]
    from eviground.explain import cp5

    out = cp5.aspect_sensitivity(p6, tok, ["u0"])
    sid = out["parts"]["sidset_tampered"]
    assert sid["as_run"]["edited_images"] == 4
    assert sid["excl_mismatched"]["edited_images"] == 3
    assert sid["as_run"]["epg_ratio"]["gradcam"] == pytest.approx(0.45 / 0.2)
    assert sid["excl_mismatched"]["epg_ratio"]["gradcam"] == pytest.approx(3.0)
    assert out["parts"]["brgen"]["as_run"] == out["parts"]["brgen"]["excl_mismatched"]
    assert out["parts"]["all"]["as_run"]["token_images"] == 8
    assert out["parts"]["all"]["excl_mismatched"]["token_images"] == 7
