"""Unit tests for the CP1 building blocks: hashing, harmonization, crops, sampler, splits,
shortcut classifier."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from eviground.data import crops, harmonize, hashing, sampler, shortcut, splits
from eviground.data.imageinfo import blockiness, header_info


def _img(seed: int, size=(64, 48), mode="RGB") -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def _bytes(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


# --- hashing -------------------------------------------------------------------------------


def test_pixel_hash_ignores_container_but_file_hash_does_not():
    img = _img(1)
    a, b = _bytes(img, "PNG"), _bytes(img, "PNG", compress_level=1)
    ha, hb = hashing.image_hashes(a), hashing.image_hashes(b)
    assert ha["sha256"] != hb["sha256"]
    assert ha["pixel_sha256"] == hb["pixel_sha256"]
    assert ha["phash"] == hb["phash"]
    assert (ha["width"], ha["height"]) == (64, 48)


def _texture(seed: int, size: int = 256) -> Image.Image:
    """Smooth random texture (pHash bits are unstable on flat or purely linear images)."""
    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    return small.resize((size, size), Image.Resampling.BICUBIC)


def test_phash_near_for_recompressed_far_for_different():
    img = _texture(0)
    p0 = hashing.phash(img)
    p1 = hashing.phash(Image.open(io.BytesIO(_bytes(img, "JPEG", quality=60))))
    p_small = hashing.phash(img.resize((200, 200)))
    p2 = hashing.phash(_texture(100))
    dist = hashing.popcount64(hashing.as_uint64([p0 ^ p1, p0 ^ p_small, p0 ^ p2])).tolist()
    assert dist[0] <= 2 and dist[1] <= 2 and dist[2] > 10


def test_min_hamming_and_counts():
    ref = hashing.as_uint64([0b0, 0b1111, 2**63])
    q = hashing.as_uint64([0b1, 0b1110000, -(2**63)])  # signed storage of 2**63
    dist, idx = hashing.min_hamming(q, ref, chunk=2)
    assert dist.tolist() == [1, 3, 0]
    assert idx.tolist() == [0, 0, 2]  # 0b1110000 is 3 bits from 0, 7 from 0b1111
    assert hashing.near_dup_counts(dist, (0, 1, 2)) == {"<=0": 1, "<=1": 2, "<=2": 2}
    d, i = hashing.min_hamming(q, hashing.as_uint64([]))
    assert d.tolist() == [65, 65, 65] and i.tolist() == [-1, -1, -1]


def test_popcount_fallback_matches():
    x = hashing.as_uint64([0, 1, 2**64 - 1, 0xF0F0])
    table = hashing._POP8[x.view(np.uint8).reshape(-1, 8)].sum(axis=-1)
    assert hashing.popcount64(x).tolist() == table.tolist() == [0, 1, 64, 8]


def test_image_hashes_reports_corrupt():
    out = hashing.image_hashes(b"\x89PNG\r\n\x1a\nnot really")
    assert "error" in out and "sha256" in out


# --- harmonization -------------------------------------------------------------------------


def _dist():
    return harmonize.QualityDistribution({(96, "4:4:4"): 3, (75, "4:2:0"): 1})


def test_quality_distribution_measure_and_deterministic_sample():
    infos = [
        {"format": "jpeg", "jpeg_quality": 96, "subsampling": "4:4:4"},
        {"format": "jpeg", "jpeg_quality": 96, "subsampling": "4:4:4"},
        {"format": "jpeg", "jpeg_quality": 80, "subsampling": "4:2:0"},
        {"format": "jpeg", "jpeg_quality": 90, "subsampling": None},  # greyscale: left out
        {"format": "png"},
    ]
    d = harmonize.QualityDistribution.measure(infos)
    assert d.counts == {(96, "4:4:4"): 2, (80, "4:2:0"): 1}
    again = harmonize.QualityDistribution.measure(infos)  # a new session, same data
    assert again.sample("a/b.png") == d.sample("a/b.png")
    draws = [d.sample(f"img{i}") for i in range(3000)]
    share = sum(x == (96, "4:4:4") for x in draws) / len(draws)
    assert 0.62 < share < 0.71
    assert harmonize.QualityDistribution.from_dict(d.to_dict()).counts == d.counts
    assert d.summary()["top_quality"] == 96
    with pytest.raises(ValueError):
        harmonize.QualityDistribution.measure([{"format": "png"}])


def _info(fmt, w=512, h=512, q=None, ss=None):
    return {"format": fmt, "width": w, "height": h, "jpeg_quality": q, "subsampling": ss}


def test_plan_genimage():
    d = _dist()
    real = harmonize.plan_image("genimage", "real", _info("jpeg", q=96, ss="4:4:4"), "r", d)
    assert real["action"] == "keep"
    gen = harmonize.plan_image("genimage", "generated", _info("png"), "g", d)
    assert gen["action"] == "compress" and (gen["quality"], gen["subsampling"]) in d.counts
    small = harmonize.plan_image("genimage", "real", _info("jpeg", 500, 200, 96), "s", d)
    assert small == {**small, "action": "skip", "reason": "too_small"}
    with pytest.raises(ValueError):
        harmonize.plan_image("genimage", "generated", _info("png"), "g", None)


@pytest.mark.parametrize(
    "info,action,reason",
    [
        (_info("png"), "compress", "lossless"),
        (_info("jpeg", q=99, ss="4:4:4"), "compress", "above_q75"),
        (_info("jpeg", q=75, ss="4:2:0"), "keep", "at_target"),
        (_info("jpeg", q=75, ss=None), "keep", "at_target"),
        (_info("jpeg", q=75, ss="4:4:4"), "compress", "q75_other_subsampling"),
        (_info("jpeg", q=70, ss="4:2:0"), "skip", "below_q75"),
    ],
)
def test_plan_sidset(info, action, reason):
    p = harmonize.plan_image("sidset", "real", info, "x", None)
    assert (p["action"], p["reason"]) == (action, reason)
    if action == "compress":
        assert (p["quality"], p["subsampling"]) == (75, "4:2:0")


def test_plan_diffseg_and_coco_use_genimage_distribution():
    d = _dist()
    for source in ("diffseg30k", "coco"):
        p = harmonize.plan_image(source, "edited", _info("png", 1024, 1024), "u", d)
        assert p["action"] == "compress" and (p["quality"], p["subsampling"]) == d.sample("u")


def test_plan_unreadable():
    p = harmonize.plan_image("genimage", "real", {"error": "decode"}, "x", _dist())
    assert p["action"] == "skip" and p["reason"] == "unreadable"


def test_apply_plan_writes_requested_quality_and_subsampling():
    data = _bytes(_img(3, (300, 280)), "PNG")
    out = harmonize.apply_plan(data, {"action": "compress", "quality": 75, "subsampling": "4:2:0"})
    info = header_info(out)
    assert (info["format"], info["jpeg_quality"], info["subsampling"]) == ("jpeg", 75, "4:2:0")
    assert info["jpeg_quality_exact"] is True
    out2 = harmonize.apply_plan(data, {"action": "compress", "quality": 96, "subsampling": "4:4:4"})
    assert header_info(out2)["subsampling"] == "4:4:4"
    assert harmonize.apply_plan(data, {"action": "keep"}) is data
    assert harmonize.apply_plan(data, {"action": "skip"}) is None


def test_compression_creates_blockiness():
    smooth = Image.fromarray(
        (np.add.outer(np.arange(256), np.arange(256)) // 2 % 256).astype(np.uint8)
    ).convert("RGB")
    raw = np.asarray(smooth)
    jpg = np.asarray(Image.open(io.BytesIO(harmonize.encode_jpeg(smooth, 30, "4:2:0"))))
    assert blockiness(jpg) > blockiness(raw) + 0.1


def test_prepare_to_size_cover_and_crop():
    out = harmonize.prepare_to_size(_img(4, (640, 480)), (320, 288))
    assert out.size == (320, 288) and out.mode == "RGB"
    out = harmonize.prepare_to_size(_img(4, (300, 900), "L"), (1024, 1024))
    assert out.size == (1024, 1024)
    same = _img(5, (640, 480))
    assert np.array_equal(np.asarray(harmonize.prepare_to_size(same, (640, 480))), np.asarray(same))
    assert harmonize.resampling((640, 480), (640, 480)) == "unchanged"
    assert harmonize.resampling((640, 480), (640, 400)) == "cropped"
    assert harmonize.resampling((640, 480), (256, 256)) == "downscaled"
    assert harmonize.resampling((640, 480), (1024, 1024)) == "upscaled"
    assert harmonize.plan_summary([{"action": "keep", "reason": "a"}] * 2) == {"keep:a": 2}


def test_size_distribution_sampling():
    sizes = [(1024, 1024)] * 3 + [(640, 480)] * 1
    d = harmonize.SizeDistribution.measure(sizes)
    assert d.n == 4 and d.summary()["top"] == {"1024x1024": 3, "640x480": 1}
    again = harmonize.SizeDistribution.measure(sizes)  # a new session, same data
    assert again.sample("coco/a.jpg") == d.sample("coco/a.jpg")
    draws = [d.sample(f"u{i}") for i in range(4000)]
    assert 0.70 < draws.count((1024, 1024)) / len(draws) < 0.80
    with pytest.raises(ValueError):
        harmonize.SizeDistribution.measure([])


# --- crops ---------------------------------------------------------------------------------


def test_random_box():
    rng = np.random.default_rng(0)
    assert crops.random_box(255, 400, rng) is None
    for _ in range(50):
        left, top, right, bottom = crops.random_box(300, 260, rng)
        assert right - left == 256 and bottom - top == 256
        assert 0 <= left <= 44 and 0 <= top <= 4


def test_edited_box_share_in_range():
    mask = np.zeros((512, 512), np.uint8)
    mask[:, :200] = 3
    rng = np.random.default_rng(1)
    box, share, k = crops.edited_box(mask, rng)
    assert box is not None and 0.01 <= share <= 0.90 and 1 <= k <= 10
    left, top, right, bottom = box
    assert abs((mask[top:bottom, left:right] > 0).mean() - share) < 1e-9


def test_edited_box_fails_when_impossible():
    rng = np.random.default_rng(1)
    box, share, k = crops.edited_box(np.full((300, 300), 4, np.uint8), rng)
    assert box is None and share == pytest.approx(1.0) and k == 10
    assert crops.edited_box(np.zeros((100, 300)), rng)[0] is None


# --- sampler -------------------------------------------------------------------------------


def test_quotas_sum_and_shares():
    q = sampler.quotas(9000)  # shares decided on 14 Nov 2026
    assert q[("real", "genimage")] == q[("real", "sidset")] == 3000
    assert q[("real", "coco")] == q[("real", "diffseg30k")] == 1500
    assert q[("generated", "genimage")] == 6000 and q[("generated", "sidset")] == 3000
    assert q[("edited", "diffseg30k")] == 9000 and ("edited", "sidset") not in q
    q = sampler.quotas(100)
    for label in sampler.CLASSES:
        assert sum(v for (c, _), v in q.items() if c == label) == 100
    det = sampler.quotas(10, "detection")
    assert det == {("real", "genimage"): 10, ("generated", "genimage"): 10}


def test_draw_respects_pools_and_is_seeded():
    src = ["genimage"] * 4 + ["sidset"] * 6 + ["coco"] * 2 + ["diffseg30k"] * 4
    lab = ["real", "generated"] * 2 + ["real", "generated", "edited"] * 2
    lab += ["real"] * 2 + ["edited"] * 3 + ["real"]
    idx = sampler.draw(src, lab, 300, seed=42)
    assert len(idx) == 900
    picked = [(lab[i], src[i]) for i in idx]
    assert picked.count(("edited", "sidset")) == 0  # SID-Set tampered not in training
    assert picked.count(("edited", "diffseg30k")) == 300
    assert picked.count(("real", "diffseg30k")) == picked.count(("real", "coco")) == 50
    assert picked.count(("generated", "genimage")) == 200
    assert (sampler.draw(src, lab, 300, seed=42) == idx).all()
    with pytest.raises(ValueError):
        sampler.draw(src[:4], lab[:4], 3, seed=1)


# --- splits --------------------------------------------------------------------------------


def test_stratified_val_ten_percent_order_independent():
    uids = [f"u{i:04d}" for i in range(1000)]
    strata = ["a" if i % 4 else "b" for i in range(1000)]
    out = splits.stratified_val(uids, strata)
    assert out.count("val") == 100
    assert sum(1 for o, s in zip(out, strata, strict=True) if o == "val" and s == "b") == 25
    rev = splits.stratified_val(uids[::-1], strata[::-1])
    assert dict(zip(uids, out, strict=True)) == dict(zip(uids[::-1], rev, strict=True))


@pytest.mark.parametrize(
    "split,editors,role",
    [
        ("train", {1, 3}, "train_pool"),
        ("train", {1, 6}, "excluded_holdout"),
        ("train", {2}, "excluded_holdout"),
        ("train", set(), "real_pool"),
        ("train", {9}, "excluded_unknown_value"),
        ("validation", {5, 2}, "test_P2"),
        ("validation", {5}, "unused_validation"),
    ],
)
def test_diffseg_role(split, editors, role):
    assert splits.diffseg_role(split, editors) == role


def test_editor_text_roundtrip():
    assert splits.editors_text([0, 5, 1, 5]) == "1,5"
    assert splits.parse_editors("1,5") == {1, 5}
    assert splits.parse_editors("") == set() and splits.parse_editors(float("nan")) == set()
    assert splits.TRAIN_EDITOR_VALUES == {1, 3, 4, 5, 7, 8}


# --- shortcut classifier --------------------------------------------------------------------


def test_lookup_cv_detects_and_ignores():
    rng = np.random.default_rng(0)
    y = ["real"] * 300 + ["fake"] * 600
    leak = [("jpeg",) if c == "real" else ("png",) for c in y]
    assert shortcut.lookup_cv(leak, y) == pytest.approx(1.0)
    noise = [(int(rng.integers(0, 3)),) for _ in y]
    assert 0.4 < shortcut.lookup_cv(noise, y) < 0.6
    assert shortcut.balanced_accuracy(["a", "a", "b"], ["a", "a", "a"]) == pytest.approx(0.5)


def test_feature_report_finds_leaking_feature():
    rng = np.random.default_rng(1)
    y = ["real", "generated"] * 200
    feats = {
        "format": ["jpeg"] * 400,
        "quality": [96 if c == "real" else int(rng.choice([96, 80])) for c in y],
        "crop_w": [256] * 400,
    }
    rep = shortcut.feature_report(feats, y)
    assert rep["single_feature"]["format"] == pytest.approx(0.5)
    assert rep["single_feature"]["quality"] > 0.7
    assert rep["drop_one"]["quality"] == pytest.approx(0.5)
    assert rep["distinct_values"] == {"format": 1, "quality": 2, "crop_w": 1}


def test_threshold_cv():
    rng = np.random.default_rng(2)
    y = ["a"] * 200 + ["b"] * 200
    sep = np.r_[rng.normal(1.0, 0.02, 200), rng.normal(1.3, 0.02, 200)]
    same = rng.normal(1.0, 0.05, 400)
    assert shortcut.threshold_cv(sep, y) > 0.95
    assert shortcut.threshold_cv(same, y) < 0.62
    d = shortcut.describe(sep)
    assert d["n"] == 400 and d["p10"] < d["median"] < d["p90"]
    assert shortcut.describe([]) == {"n": 0}


def test_weighted_balanced_accuracy_and_threshold():
    weighted_ba = shortcut.balanced_accuracy(["a", "a", "b"], ["a", "b", "b"], [3, 1, 1])
    assert weighted_ba == pytest.approx(0.875)
    rng = np.random.default_rng(3)
    # class b = two sources; source 2 (level 1.3) is over-measured but its sampler weight is
    # small, so with weights the threshold must mostly see source 1 (level 1.0, as class a)
    a = list(rng.normal(1.0, 0.02, 200))
    b = list(rng.normal(1.0, 0.02, 200)) + list(rng.normal(1.3, 0.02, 400))
    y = ["a"] * 200 + ["b"] * 600
    unweighted = shortcut.threshold_cv(a + b, y)
    w = [1 / 200] * 200 + [0.95 / 200] * 200 + [0.05 / 400] * 400
    weighted = shortcut.threshold_cv(a + b, y, weights=w)
    assert unweighted > 0.8 and weighted < 0.6
