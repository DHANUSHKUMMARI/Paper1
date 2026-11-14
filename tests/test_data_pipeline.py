"""Milestone 3, PR B: packed shards, the training / validation datasets and augmentation."""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent))

from test_cp1 import _cfg, _ensure, _world  # noqa: E402

from eviground.data import cp1, crops, diffseg_mask, pack, sampler  # noqa: E402
from eviground.data.datasets import ForensicDataset, ShardReader  # noqa: E402
from eviground.data.transforms import Augment, resize_mask  # noqa: E402

CROP = crops.CROP


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("packed")
    paths, sources = _world(tmp)
    _ensure(paths, sources)
    cp1.run_cp1(paths, _cfg(), out_dir=tmp / "out", cache_dir=tmp / "cache", progress=None)
    manifest = tmp / "out" / "manifest_cp1.csv.gz"
    out = tmp / "shards"
    summary = pack.pack(
        manifest,
        out,
        free_bytes=10**12,
        margin=0,
        shard_bytes=200_000,  # several shards per source in the fake world
        sample_per_source=5,
        progress=None,
    )
    return {"manifest": manifest, "out": out, "summary": summary, "tmp": tmp}


def _rows(packed):
    return cp1._read_csv_gz(packed["manifest"])


# ------------------------------------------------------------------ packing


def test_pack_holds_exactly_the_training_and_validation_images(packed):
    want = {
        r["uid"]
        for r in _rows(packed)
        if r["split"] in ("train", "val") and r["h_action"] in ("keep", "compress")
    }
    index = pack.read_index(packed["out"] / "index.parquet")
    assert {e["uid"] for e in index} == want
    assert len(index) == len(want) == packed["summary"]["images"]
    assert all(e["split"] in ("train", "val") for e in index)
    assert not any(str(e["role"]).startswith(("excluded", "test")) for e in index)
    assert packed["summary"]["shards"] > 4  # the small shard size splits every source
    assert set(index[0]) == set(pack.INDEX_COLUMNS)


def test_no_held_out_generator_or_editor_in_the_shards(packed):
    held = {r["uid"] for r in _rows(packed) if "holdout" in (r.get("role") or "")}
    assert held  # the fake world has held-out images
    index = pack.read_index(packed["out"] / "index.parquet")
    assert not held & {e["uid"] for e in index}
    names = " ".join(str(e.get("generator")) + " " + str(e.get("editors")) for e in index).lower()
    assert "glide" not in names and "kolors" not in names


def test_packed_bytes_equal_a_fresh_harmonization_for_every_source(packed):
    res = pack.verify_sample(packed["out"], _rows(packed), per_source=1000, seed=42)
    sources = {k for k in res if k != "all_equal"}
    assert sources == {"coco", "diffseg30k", "genimage", "sidset"}
    assert res["all_equal"], res
    assert all(res[s]["checked"] > 0 for s in sources)


def test_shard_round_trip_reads_the_stored_bytes(packed):
    index = pack.read_index(packed["out"] / "index.parquet")
    by_uid = {r["uid"]: pack.typed(r) for r in _rows(packed)}
    for e in index[::7]:
        data = pack.read_image_bytes(packed["out"], e)
        assert data == cp1.harmonized_bytes(by_uid[e["uid"]])
        assert pack.image_format(data) == e["img_format"]
        with Image.open(io.BytesIO(data)) as img:
            assert img.size == (e["width"], e["height"])


def test_stored_edit_mask_is_the_cp1_reading_rule(packed):
    index = pack.read_index(packed["out"] / "index.parquet")
    by_uid = {r["uid"]: pack.typed(r) for r in _rows(packed)}
    edited = [e for e in index if e["label"] == "edited"]
    assert edited and all(e["mask_size"] for e in edited)
    assert not any(e["mask_size"] for e in index if e["label"] != "edited")
    reader = ShardReader(packed["out"])
    for e in edited:
        r = by_uid[e["uid"]]
        with Image.open(r["mask_path"]) as m:
            raw = np.asarray(m.convert("L"))
        want = diffseg_mask.edit_mask(raw, r["mask_base"]) > 0
        assert np.array_equal(reader.mask(e), want)
        assert e["mask_rule"] == pack.MASK_RULE
        assert e["mask_base"] == r["mask_base"]
        data = pack.read_image_bytes(packed["out"], e, "mask")
        with Image.open(io.BytesIO(data)) as m:
            assert m.mode == "1"


def test_free_space_stop(packed):
    est = {"total_bytes": 5 * 10**9}
    with pytest.raises(pack.PackSpaceError, match="stop and ask"):
        pack.check_space(est, free_bytes=14 * 10**9)  # < 5 GB + 10 GB margin
    pack.check_space(est, free_bytes=16 * 10**9)
    with pytest.raises(pack.PackSpaceError):
        pack.pack(packed["manifest"], packed["tmp"] / "small", free_bytes=10, progress=None)
    assert not (packed["tmp"] / "small").exists()  # nothing written


def test_pack_does_nothing_when_done(packed, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("source image read although _DONE exists")

    monkeypatch.setattr(cp1, "harmonized_bytes", boom)
    again = pack.pack(packed["manifest"], packed["out"], free_bytes=None, progress=None)
    assert again["images"] == packed["summary"]["images"]


def test_pack_resumes_after_an_interrupted_run(packed, tmp_path):
    out = tmp_path / "shards"
    kw = {"free_bytes": 10**12, "margin": 0, "shard_bytes": 200_000, "sample_per_source": 5}
    first = pack.pack(packed["manifest"], out, progress=None, **kw)
    (out / "_DONE.json").unlink()
    victim = sorted(out.rglob("*.tar"))[0]
    victim.write_bytes(victim.read_bytes()[:100])  # a shard cut off mid-write
    again = pack.pack(packed["manifest"], out, progress=None, **kw)
    assert again["shards_written"] == 1
    assert again["shards_skipped"] == first["shards"] - 1
    assert pack.verify_sample(out, _rows(packed), per_source=1000)["all_equal"]
    assert json.loads((out / "_DONE.json").read_text())["images"] == first["images"]


# ------------------------------------------------------------------ datasets


def test_training_items_are_crops_never_padded(packed):
    ds = ForensicDataset(packed["out"], "train", n_per_class=40, seed=42)
    assert len(ds) == 120
    for k in range(len(ds)):
        i, (left, top, right, bottom, s) = ds.items[k]
        e = ds.rows[i]
        w, h = round(e["width"] * s), round(e["height"] * s)
        assert right - left == CROP and bottom - top == CROP
        assert 0 <= left and right <= w and 0 <= top and bottom <= h
        assert min(w, h) >= CROP
    for k in (0, 1, 2, 50, 119):
        it = ds[k]
        assert it["image"].shape == (3, CROP, CROP) and it["mask"].shape == (1, CROP, CROP)
        assert 0 <= float(it["image"].min()) and float(it["image"].max()) <= 1
        assert it["has_mask"] == (it["label"] == 2)


def test_edited_training_crops_follow_the_crop_rule(packed):
    ds = ForensicDataset(packed["out"], "train", n_per_class=30, seed=123)
    for k, (i, _spec) in enumerate(ds.items):
        if ds.rows[i]["label"] != "edited":
            continue
        share = float(ds[k]["mask"].mean())
        assert crops.EDITED_SHARE[0] <= share <= crops.EDITED_SHARE[1]


def test_class_ratio_and_source_shares_over_3000_draws(packed):
    ds = ForensicDataset(packed["out"], "train", n_per_class=1000, seed=2026)
    labels = Counter(ds.rows[i]["label"] for i, _ in ds.items)
    assert labels == {"real": 1000, "generated": 1000, "edited": 1000}
    pools = Counter((ds.rows[i]["label"], ds.rows[i]["source"]) for i, _ in ds.items)
    for (label, source), share in (
        (key, sampler.SHARES["unified"][key[0]][key[1]]) for key in sampler.quotas(1000)
    ):
        assert abs(pools[(label, source)] / 1000 - share) <= 0.05


def test_epoch_plans_are_deterministic_and_differ_between_epochs(packed):
    a = ForensicDataset(packed["out"], "train", n_per_class=20, seed=42)
    b = ForensicDataset(packed["out"], "train", n_per_class=20, seed=42)
    assert a.items == b.items
    first = list(a.items)
    a.set_epoch(1)
    assert a.items != first
    b.set_epoch(1)
    assert a.items == b.items
    for k in (0, 7, 33):  # the same item after a "restart": resume can start at any k
        x, y = a[k], b[k]
        assert x["uid"] == y["uid"]
        assert bool((x["image"] == y["image"]).all()) and bool((x["mask"] == y["mask"]).all())


def test_never_valid_edited_images_are_left_out_and_listed(packed):
    rows = pack.read_index(packed["out"] / "index.parquet")
    edited = [e for e in rows if e["label"] == "edited"]
    edited[0]["valid_crop"] = False
    ds = ForensicDataset(packed["out"], "train", n_per_class=20, rows=rows)
    assert ds.dropped == [edited[0]["uid"]]
    assert edited[0]["uid"] not in {ds.rows[i]["uid"] for i, _ in ds.items}


def test_validation_has_one_fixed_crop_per_image(packed):
    rows = pack.read_index(packed["out"] / "index.parquet")
    for e in rows:  # the fake world has no edited validation image: use the training ones
        if e["label"] == "edited":
            e["split"] = "val"
    a = ForensicDataset(packed["out"], "val", rows=rows, seed=42)
    b = ForensicDataset(packed["out"], "val", rows=rows, seed=42)
    n_val = sum(1 for e in rows if e["split"] == "val")
    assert len(a) == n_val and a.items == b.items
    assert not a.augment.enabled
    a.set_epoch(3)  # no effect on validation
    assert a.items == b.items
    for k, (i, _spec) in enumerate(a.items):
        it = a[k]
        assert it["image"].shape == (3, CROP, CROP)
        assert bool((it["image"] == b[k]["image"]).all())
        if a.rows[i]["label"] == "edited":
            share = float(it["mask"].mean())
            assert crops.EDITED_SHARE[0] <= share <= crops.EDITED_SHARE[1]


def test_detection_profile_has_no_edited_images(packed):
    ds = ForensicDataset(packed["out"], "train", profile="detection", n_per_class=50)
    assert {ds.rows[i]["source"] for i, _ in ds.items} == {"genimage"}
    assert Counter(ds.rows[i]["label"] for i, _ in ds.items) == {"real": 50, "generated": 50}


# ------------------------------------------------------------------ augmentation


def _striped(w=400, h=320):
    """An image whose red channel is 255 exactly where the mask is set."""
    mask = np.zeros((h, w), dtype=bool)
    mask[40:200, 30:170] = True
    mask[:, 300:330] = True
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[mask, 0] = 255
    img[..., 1] = np.arange(w, dtype=np.uint8)[None, :]  # makes a flip visible
    return img, mask


def test_flip_and_resize_keep_image_and_mask_aligned():
    img, mask = _striped()
    geo = Augment(resize_p=1.0, flip_p=0.5, jpeg_p=0.0, blur_p=0.0, noise_p=0.0)
    flips = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        s = geo.scale(rng, img.shape[1], img.shape[0], CROP)
        assert min(img.shape[:2]) * s >= CROP - 0.5
        pil = Image.fromarray(img)
        from eviground.data.transforms import resize_image

        big = np.asarray(resize_image(pil, s))
        m = resize_mask(mask, s)
        assert big.shape[:2] == m.shape
        box = crops.random_box(m.shape[1], m.shape[0], rng, CROP)
        left, top, right, bottom = box
        crop, flipped = geo.photometric(big[top:bottom, left:right], rng)
        mc = m[top:bottom, left:right][:, ::-1] if flipped else m[top:bottom, left:right]
        flips += flipped
        agree = np.mean((crop[..., 0] > 127) == mc)
        assert agree > 0.98, (seed, s, agree)
    assert 0 < flips < 12


def test_augmentation_off_returns_the_crop_unchanged():
    img, _ = _striped()
    off = Augment(enabled=False)
    rng = np.random.default_rng(0)
    assert off.scale(rng, 400, 320, CROP) == 1.0
    out, flipped = off.photometric(img, rng)
    assert out is img and not flipped


def test_photometric_steps_change_only_pixels_and_are_seeded():
    img, _ = _striped()
    aug = Augment(flip_p=0.0, jpeg_p=1.0, blur_p=1.0, noise_p=1.0)
    a, _ = aug.photometric(img, np.random.default_rng(5))
    b, _ = aug.photometric(img, np.random.default_rng(5))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, img)


def test_augment_from_config():
    assert Augment.from_config(False) == Augment(enabled=False)
    a = Augment.from_config({"jpeg_p": 0.25, "resize_range": [0.75, 1.25]})
    assert a.jpeg_p == 0.25 and a.resize_range == (0.75, 1.25) and a.enabled


def test_shard_paths_cannot_leave_the_shard_folder(tmp_path):
    assert (
        pack.inside(tmp_path, "coco/shard-00000.tar")
        == (tmp_path / "coco/shard-00000.tar").resolve()
    )
    for bad in ("../x.tar", "coco/../../x.tar", "/etc/passwd", "."):
        with pytest.raises(ValueError):
            pack.inside(tmp_path, bad)
    with pytest.raises(ValueError, match="source name"):
        pack.plan_shards([{"source": "../evil", "uid": "a"}], {})
    with pytest.raises(ValueError):
        pack.read_image_bytes(tmp_path, {"shard": "../x.tar", "img_offset": 0, "img_size": 1})
