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


def test_a_shard_from_another_manifest_is_rewritten(packed, tmp_path):  # noqa: F811
    out = tmp_path / "shards"
    kw = {"free_bytes": 10**12, "margin": 0, "shard_bytes": 200_000, "sample_per_source": 5}
    first = pack.pack(packed["manifest"], out, progress=None, **kw)
    (out / "_DONE.json").unlink()
    other = tmp_path / "manifest_other.csv.gz"  # same rows, different bytes: another manifest
    import gzip

    other.write_bytes(gzip.compress(gzip.decompress(packed["manifest"].read_bytes()) + b"\n"))
    again = pack.pack(other, out, progress=None, **kw)
    assert again["shards_skipped"] == 0
    assert again["shards_written"] == first["shards"]


def test_stored_offsets_match_the_tar_members(packed):  # noqa: F811
    import tarfile

    index = pack.read_index(packed["out"] / pack.INDEX_FILE)
    by_shard: dict[str, list[dict]] = {}
    for e in index:
        by_shard.setdefault(e["shard"], []).append(e)
    for shard, entries in by_shard.items():
        with tarfile.open(packed["out"] / shard) as tar:
            members = {m.offset_data: m.size for m in tar.getmembers()}
        for e in entries:
            assert members[e["img_offset"]] == e["img_size"]
            if e["mask_offset"] is not None:
                assert members[e["mask_offset"]] == e["mask_size"]


# ------------------------------------------------------------------ local copy (15 Nov 2026)


def test_staged_shards_are_identical_and_staging_resumes(packed, tmp_path):  # noqa: F811
    local = tmp_path / "local_shards"
    first = pack.stage_shards(packed["out"], local, progress=None)
    tars = sorted(p.relative_to(packed["out"]) for p in packed["out"].rglob("*.tar"))
    assert first["shards"] == first["copied"] == len(tars)
    for rel in tars:
        assert pack.file_sha256(local / rel) == pack.file_sha256(packed["out"] / rel)
    for name in (pack.INDEX_FILE, "_DONE.json"):
        assert (local / name).read_bytes() == (packed["out"] / name).read_bytes()
    again = pack.stage_shards(packed["out"], local, progress=None)
    assert again["copied"] == 0 and again["skipped"] == len(tars)
    victim = local / tars[0]
    victim.write_bytes(victim.read_bytes()[:100])  # a copy cut off
    third = pack.stage_shards(packed["out"], local, progress=None)
    assert third["copied"] == 1
    assert pack.file_sha256(victim) == pack.file_sha256(packed["out"] / tars[0])


def test_staging_stops_without_local_space(packed, tmp_path):  # noqa: F811
    with pytest.raises(pack.PackSpaceError, match="local disk"):
        pack.stage_shards(packed["out"], tmp_path / "x", free_bytes=10, progress=None)


def test_training_items_from_the_local_copy_equal_the_originals(packed, tmp_path):  # noqa: F811
    local = tmp_path / "local_shards"
    pack.stage_shards(packed["out"], local, progress=None)
    a = ForensicDataset(packed["out"], "train", n_per_class=10, seed=42)
    b = ForensicDataset(local, "train", n_per_class=10, seed=42)
    assert a.items == b.items
    for k in range(len(a)):
        x, y = a[k], b[k]
        assert x["uid"] == y["uid"]
        assert bool((x["image"] == y["image"]).all()) and bool((x["mask"] == y["mask"]).all())


def test_reader_opens_each_shard_once_and_drops_descriptors_when_pickled(packed, monkeypatch):  # noqa: F811
    import os
    import pickle

    opened = []
    real_open = os.open

    def counting_open(path, *a, **k):
        opened.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr(os, "open", counting_open)
    reader = ShardReader(packed["out"])
    rows = reader.rows
    for e in rows + rows:  # every image twice
        assert reader.read(e) == pack.read_image_bytes(packed["out"], e)
        assert reader.read(e, "mask") == pack.read_image_bytes(packed["out"], e, "mask")
    assert sorted(opened) == sorted({str(pack.inside(packed["out"], e["shard"])) for e in rows})
    clone = pickle.loads(pickle.dumps(reader))
    assert clone._fds == {} and reader._fds
    assert clone.read(rows[0]) == reader.read(rows[0])
    reader.close()
    assert reader._fds == {}


def test_a_local_copy_of_another_pack_is_never_reused(packed, tmp_path):  # noqa: F811
    """Codex, 15 Nov: a re-packed shard of the same size must not be skipped."""
    import shutil

    local = tmp_path / "local_shards"
    pack.stage_shards(packed["out"], local, progress=None)
    drive2 = tmp_path / "drive2"  # the same folder after a re-pack
    shutil.copytree(packed["out"], drive2)
    tars = sorted(p.relative_to(drive2) for p in drive2.rglob("*.tar"))
    victim = drive2 / tars[0]
    data = bytearray(victim.read_bytes())
    data[700] ^= 0xFF  # different bytes, same size
    victim.write_bytes(bytes(data))
    rows = pack.read_index(drive2 / pack.INDEX_FILE)
    rows[0]["role"] = "repacked"  # a new index
    pack.write_index(rows, drive2 / pack.INDEX_FILE)
    res = pack.stage_shards(drive2, local, progress=None)
    assert res["copied"] == len(tars) and res["skipped"] == 0
    assert (local / tars[0]).read_bytes() == bytes(data)
    assert json.loads((local / pack.STAGED_FILE).read_text()) == {
        "index_sha256": pack.file_sha256(drive2 / pack.INDEX_FILE),
        "complete": True,
    }


def test_an_interrupted_staging_resumes_for_the_same_pack(packed, tmp_path, monkeypatch):  # noqa: F811
    local = tmp_path / "local_shards"
    real = pack._copy_verified
    calls = []

    def flaky(src, dst, size):
        calls.append(dst)
        if len(calls) == 3:
            raise OSError("runtime disconnected")
        return real(src, dst, size)

    monkeypatch.setattr(pack, "_copy_verified", flaky)
    with pytest.raises(OSError, match="disconnected"):
        pack.stage_shards(packed["out"], local, progress=None)
    assert json.loads((local / pack.STAGED_FILE).read_text())["complete"] is False
    monkeypatch.setattr(pack, "_copy_verified", real)
    res = pack.stage_shards(packed["out"], local, progress=None)
    assert res["skipped"] == 2 and res["copied"] == res["shards"] - 2
    assert json.loads((local / pack.STAGED_FILE).read_text())["complete"] is True


# ------------------------------------------------------------------ CP4 fix: context crops


def test_context_crop_keeps_only_edit_free_or_valid_windows():
    mask = np.zeros((600, 600), bool)
    mask[:300, :300] = True  # a quarter edited
    rng = np.random.default_rng(0)
    seen = Counter()
    for _ in range(300):
        box, share, _tries = crops.context_crop(mask, rng)
        if box is None:
            continue
        assert share == 0 or crops.EDITED_SHARE[0] <= share <= crops.EDITED_SHARE[1]
        seen["free" if share == 0 else "edited"] += 1
    assert seen["free"] > 0
    assert seen["edited"] > 0
    assert crops.context_crop(np.zeros((100, 100), bool), rng)[0] is None  # smaller than a crop
    full = np.ones((300, 300), bool)  # every window is all edited: never accepted
    assert crops.context_crop(full, rng, tries=5)[0] is None


def test_context_share_zero_leaves_the_epoch_plan_unchanged(packed):
    """CP3 reproducibility: the CP4 fix is off by default and draws nothing extra."""
    a = ForensicDataset(packed["out"], "train", n_per_class=40, seed=42)
    b = ForensicDataset(packed["out"], "train", n_per_class=40, seed=42, context_share=0.0)
    assert a.items == b.items
    assert all(len(spec) == 5 for _i, spec in a.items)
    assert a.context_counts() == {"context_edited": 0, "context_generated": 0}


def test_context_crops_only_on_ai_base_edits_and_edit_free_ones_are_generated(packed):
    ds = ForensicDataset(packed["out"], "train", n_per_class=300, seed=7, context_share=1.0)
    ai = [
        k
        for k, (i, _s) in enumerate(ds.items)
        if ds.rows[i]["label"] == "edited" and ds.rows[i].get("base_type") == "ai"
    ]
    assert ai, "the fake world has AI-base edited images"
    tagged = Counter()
    for k, (i, spec) in enumerate(ds.items):
        if len(spec) > 5:
            assert ds.rows[i]["label"] == "edited"
            assert ds.rows[i]["base_type"] == "ai"
            tagged[spec[5]] += 1
            it = ds[k]
            share = float(it["mask"].mean())
            if spec[5] == "context_generated":
                assert it["label"] == 1  # generated
                assert not it["has_mask"]
                assert share == 0
            else:
                assert it["label"] == 2
                assert it["has_mask"]
                assert crops.EDITED_SHARE[0] <= share <= crops.EDITED_SHARE[1]
    assert sum(tagged.values()) == sum(ds.context_counts().values()) > 0
    real_base = [
        spec
        for i, spec in ds.items
        if ds.rows[i]["label"] == "edited" and ds.rows[i].get("base_type") != "ai"
    ]
    assert all(len(spec) == 5 for spec in real_base)  # real-base edits keep the CP3 rule
    again = ForensicDataset(packed["out"], "train", n_per_class=300, seed=7, context_share=1.0)
    assert again.items == ds.items  # seeded


def test_context_share_is_about_the_requested_share(packed):
    ds = ForensicDataset(packed["out"], "train", n_per_class=1000, seed=11, context_share=1 / 3)
    ai = sum(
        1
        for i, _s in ds.items
        if ds.rows[i]["label"] == "edited" and ds.rows[i].get("base_type") == "ai"
    )
    n = sum(ds.context_counts().values())
    assert ai > 50
    assert abs(n / ai - 1 / 3) < 0.12  # some context draws fall back to the edited-crop rule


def test_edit_free_context_crop_with_edited_pixels_is_refused(packed):
    """The loader re-checks every crop labelled generated against the edit mask."""
    from eviground.data import datasets

    ds = ForensicDataset(packed["out"], "train", n_per_class=300, seed=7, context_share=1.0)
    k, (i, spec) = next(
        (k, item) for k, item in enumerate(ds.items) if len(item[1]) > 5 and item[1][5] == "context"
    )  # a context crop that holds part of the edit...
    ds.items[k] = (i, (*spec[:5], datasets.CONTEXT_GENERATED))  # ...wrongly tagged edit-free
    with pytest.raises(datasets.ContextCropError):
        ds[k]


def test_context_crops_move_edit_free_draws_from_edited_to_generated(packed):
    base = ForensicDataset(packed["out"], "train", n_per_class=500, seed=3)
    fix = ForensicDataset(packed["out"], "train", n_per_class=500, seed=3, context_share=1 / 3)
    b = Counter(LABEL for LABEL in (ds_label(base, k) for k in range(len(base))))
    f = Counter(LABEL for LABEL in (ds_label(fix, k) for k in range(len(fix))))
    moved = fix.context_counts()["context_generated"]
    assert b == {"real": 500, "generated": 500, "edited": 500}
    assert f == {"real": 500, "generated": 500 + moved, "edited": 500 - moved}


def ds_label(ds, k):
    i, spec = ds.items[k]
    if len(spec) > 5 and spec[5] == "context_generated":
        return "generated"
    return ds.rows[i]["label"]


def test_sampler_check_compares_the_cp3_and_fixed_samplers(packed):
    from eviground.data import sampler_check

    cp3 = ForensicDataset(packed["out"], "train", n_per_class=200, seed=5)
    fix = ForensicDataset(packed["out"], "train", n_per_class=200, seed=5, context_share=1.0)
    rec_cp3 = sampler_check.plan_records(cp3, range(2))
    rec_fix = sampler_check.plan_records(fix, range(2))
    assert len(rec_cp3) == len(rec_fix) == 2 * 600
    a, b = sampler_check.check(rec_cp3), sampler_check.check(rec_fix)
    assert set(a["metadata_real_vs_not_real"]["features"]) == set(sampler_check.FEATURES)
    assert a["context"] == {}
    assert sum(b["context"].values()) > 0
    gen_ds = b["source_mix"]["generated"].get("diffseg30k", 0.0)
    assert gen_ds > 0  # edit-free DiffSeg30k AI-base crops now count as generated
    assert a["source_mix"]["generated"].get("diffseg30k", 0.0) == 0.0
    assert gen_ds == pytest.approx(
        b["context"].get("context_generated", 0) / b["class_counts"]["generated"], abs=1e-3
    )
    same = sampler_check.compare(a, a)
    assert same["material"] == ([] if a["criterion_pass"] else same["material"])
    assert all(v == 0 for v in same["source_share_change"].values())
    diff = sampler_check.compare(a, b)
    assert diff["source_share_change_designed"]["generated.diffseg30k"] == pytest.approx(
        gen_ds, abs=1e-4
    )
    # judged without the edit-free context crops: the designed move is not "material"
    assert "generated.diffseg30k" not in diff["source_share_change"]
    assert not [m for m in diff["material"] if "source share" in m]


def test_sampler_check_flags_a_source_level_quality_75_shift():
    """Codex review: a source's quality-75 share can move while its class total does not."""
    from eviground.data import sampler_check

    def summary(q_src):
        per_class = {"real": {"n": 10, "q75_any": 0.3, "q75_420": 0.3}}
        per_src = {f"real/{s}": {"n": 5, "q75_any": v, "q75_420": v} for s, v in q_src.items()}
        return {
            "criterion_score": 0.55,
            "criterion_pass": True,
            "metadata_three_class": {"balanced_accuracy": 0.5},
            "quality_75": {"per_class": per_class, "per_class_source": per_src},
            "source_mix": {"real": {"coco": 0.5, "genimage": 0.5}},
            "source_mix_without_context_generated": {"real": {"coco": 0.5, "genimage": 0.5}},
        }

    base = summary({"coco": 0.5, "genimage": 0.1})
    shifted = summary({"coco": 0.3, "genimage": 0.3})  # same class total, sources moved
    out = sampler_check.compare(base, shifted)
    assert out["quality_75_change"]["per_class.real.q75_any"] == 0
    assert out["quality_75_change"]["per_class_source.real/coco.q75_any"] == 0.2
    assert any("real/coco" in m for m in out["material"])
    assert sampler_check.compare(base, base)["material"] == []
