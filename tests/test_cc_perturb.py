"""PR Z: counterfactual consistency (explain/cc.py) and the P5 perturbations (eval/perturb.py)."""

import numpy as np
import pytest
import torch

from eviground.eval import perturb
from eviground.explain import cc
from eviground.models.eviground import aggregate_tiles

# ---- a tiny stand-in for EviGround's token interface -----------------------------------------


class TokenModel(torch.nn.Module):
    """Branch tokens: d = 2, a 16 x 16 grid per tile. A trace token's first feature is the mean
    of its cell's red channel. The class head calls a tile edited in proportion to how many trace
    tokens are bright, so removing bright tokens (replacing them by the tile mean) lowers
    P(edited) and removing dark ones does not."""

    grid = 16

    def branch_tokens(self, images):
        b, _, h, w = images.shape
        red = torch.nn.functional.adaptive_avg_pool2d(images[:, :1], self.grid)  # (b, 1, g, g)
        t = torch.cat([red, torch.zeros_like(red)], dim=1).flatten(2).transpose(1, 2)
        s = torch.zeros(b, 1 + self.grid**2, 2)
        return s, t

    def logits_from_tokens(self, s, t):
        bright = (t[..., 0] > 0.5).float().mean(dim=1)  # share of bright tokens
        edited = 8 * bright - 1
        return torch.stack([torch.zeros_like(edited), torch.full_like(edited, -3.0), edited], 1)


def _image(h=256, w=512, box=(0, 0, 128, 128)):
    img = np.zeros((h, w, 3), np.uint8)
    y0, x0, y1, x1 = box
    img[y0:y1, x0:x1, 0] = 255
    region = np.zeros((h, w), bool)
    region[y0:y1, x0:x1] = True
    return img, region


def test_token_membership_uses_the_25_percent_share():
    m = np.zeros((256, 256), bool)
    m[:4, :16] = True  # a quarter of token (0, 0)
    m[16:19, 16:32] = True  # 3/16 of token (1, 1): below the share
    sel = cc.token_membership(m)
    assert sel.shape == (256,)
    assert sel[0]
    assert not sel[17]
    assert sel.sum() == 1


def test_token_membership_on_a_small_tile():
    m = np.ones((64, 64), bool)  # a tile narrower than 256: cells of 4 x 4 pixels
    assert cc.token_membership(m).all()


def test_remove_tokens_replaces_with_the_tile_mean_and_keeps_the_class_token():
    s = torch.arange(2 * 5 * 1, dtype=torch.float32).reshape(2, 5, 1)
    t = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    sel = torch.tensor([[True, False, False, False], [False, False, False, False]])
    s2, t2 = cc.remove_tokens(s, t, sel)
    assert s2[0, 0, 0] == s[0, 0, 0]  # class token kept
    assert s2[0, 1, 0] == s[0, 1:].mean()
    assert t2[0, 0, 0] == t[0].mean()
    assert torch.equal(s2[1], s[1])
    assert torch.equal(t2[1], t[1])
    assert s[0, 1, 0] == 1  # inputs untouched


def test_random_selection_matches_the_counts_per_tile():
    sel = cc.random_selection([3, 0, 5], 256, np.random.default_rng(0))
    assert sel.sum(dim=1).tolist() == [3, 0, 5]


def test_a_region_carrying_the_evidence_passes_and_a_blank_one_does_not():
    model = TokenModel()
    img, region = _image()
    tokens = cc.image_tokens(model, img)
    assert len(tokens["boxes"]) == 2  # 256 x 512 -> two tiles
    rec = cc.region_cc(model, tokens, region, seed=1)
    assert rec["pred_label"] == "edited"
    assert rec["tokens_removed"] == 64  # 128 x 128 px = 8 x 8 tokens in the first tile
    assert rec["tiles_touched"] == 1
    assert rec["drop"] >= cc.DELTA
    assert rec["drop"] > rec["random_drop"]
    assert rec["passed"]
    blank = np.zeros_like(region)
    blank[200:256, 400:512] = True  # dark corner of the second tile
    rec2 = cc.region_cc(model, tokens, blank, seed=1)
    assert rec2["scored"]
    assert not rec2["passed"]


def test_the_image_level_drop_uses_the_tile_aggregation():
    model = TokenModel()
    img, region = _image()
    tokens = cc.image_tokens(model, img)
    p = torch.softmax(model.logits_from_tokens(tokens["s"], tokens["t"]), -1)
    rec = cc.region_cc(model, tokens, region, seed=1)
    assert rec["p_pred"] == pytest.approx(aggregate_tiles(p)["edited"])  # max over tiles


def test_a_region_without_a_token_at_the_share_is_not_scored():
    model = TokenModel()
    img, _ = _image()
    tiny = np.zeros(img.shape[:2], bool)
    tiny[0:2, 0:2] = True
    rec = cc.region_cc(model, cc.image_tokens(model, img), tiny)
    assert rec["scored"] is False
    s = cc.cc_summary([rec])
    assert s["unscored_no_tokens"] == 1
    assert s["cc"] is None


def test_cc_is_seeded_per_region():
    model = TokenModel()
    img, region = _image()
    tokens = cc.image_tokens(model, img)
    a = cc.region_cc(model, tokens, region, seed=cc.seed_for("u", "E1"))
    b = cc.region_cc(model, tokens, region, seed=cc.seed_for("u", "E1"))
    assert a["random_drop"] == b["random_drop"]
    assert cc.seed_for("u", "E1") != cc.seed_for("u", "E2")


def test_cc_summary_counts():
    recs = [
        {"scored": True, "passed": True, "drop": 0.5, "random_drop": 0.1},
        {"scored": True, "passed": False, "drop": 0.1, "random_drop": 0.2},
        {"scored": False},
    ]
    s = cc.cc_summary(recs)
    assert (s["regions"], s["scored"], s["passed"]) == (3, 2, 1)
    assert s["cc"] == 0.5
    assert s["drop_over_delta"] == 1
    assert s["drop_over_random"] == 1


# ---- P5 perturbations --------------------------------------------------------------------------


def _rgb(h=40, w=60):
    return (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)


def test_jpeg_changes_pixels_and_lower_quality_changes_more():
    img = _rgb()
    d95 = np.abs(perturb.jpeg(img, 95).astype(int) - img).mean()
    d50 = np.abs(perturb.jpeg(img, 50).astype(int) - img).mean()
    assert 0 < d95 < d50


def test_resize_keeps_the_mask_aligned():
    img = _rgb()
    mask = np.zeros(img.shape[:2], bool)
    mask[10:20, 30:50] = True
    out, m = perturb.resize(img, 0.5, mask)
    assert out.shape == (20, 30, 3)
    assert m.shape == (20, 30)
    assert m[5:10, 15:25].all()
    assert m.sum() == 50
    up, m2 = perturb.resize(img, 1.5, mask)
    assert up.shape == (60, 90, 3)
    assert m2.shape == (60, 90)


def test_blur_and_noise():
    img = _rgb()
    assert perturb.blur(img, 3).std() < img.std()
    a = perturb.noise(img, 10, seed=1)
    b = perturb.noise(img, 10, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, img)
    assert a.dtype == np.uint8


def test_apply_is_deterministic_per_uid_and_clean_is_unchanged():
    img = _rgb()
    for name, kind, level in perturb.iter_conditions():
        a, _ = perturb.apply(img, kind, level, uid="x")
        b, _ = perturb.apply(img, kind, level, uid="x")
        assert np.array_equal(a, b), name
    clean, _ = perturb.apply(img, None, None)
    assert clean is img
    n1, _ = perturb.apply(img, "noise", 10.0, uid="x")
    n2, _ = perturb.apply(img, "noise", 10.0, uid="y")
    assert not np.array_equal(n1, n2)
    with pytest.raises(ValueError):
        perturb.apply(img, "rotate", 3)


def test_conditions_cover_the_spec_end_points():
    names = [n for n, _, _ in perturb.iter_conditions()]
    assert names[0] == "clean"
    assert len(names) == 1 + 4 * 3
    assert {"jpeg_95", "jpeg_50", "resize_0.5", "resize_1.5", "blur_1", "blur_3"} <= set(names)
