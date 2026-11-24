"""P6 image loaders (notebook 14): each part on fake files."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from eviground.eval import p6data


def _write(path, arr, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode).save(path)
    return path


def _world(tmp_path):
    rgb = np.full((40, 60, 3), 120, np.uint8)
    m = np.zeros((40, 60), np.uint8)
    m[:10, :20] = 200  # > 127: edited
    m[30:, 50:] = 100  # <= 127: not edited
    ds_mask = np.zeros((40, 60), np.uint8)
    ds_mask[5:15, 5:15] = 6  # a Glide edit on a real base (base 0)
    rows = {
        "ds/1": {
            "uid": "ds/1",
            "source": "diffseg30k",
            "label": "edited",
            "role": "test_P2",
            "path": str(_write(tmp_path / "ds1.png", rgb)),
            "mask_path": str(_write(tmp_path / "ds1.mask.png", ds_mask, "L")),
            "mask_base": "0",
        },
        "sid/t": {
            "uid": "sid/t",
            "source": "sidset",
            "label": "edited",
            "role": "test_P3b",
            "path": str(_write(tmp_path / "sid.png", rgb)),
            "mask_path": str(_write(tmp_path / "sid.mask.png", m, "L")),
        },
        "gi/g": {
            "uid": "gi/g",
            "source": "genimage",
            "label": "generated",
            "generator": "adm",
            "path": str(_write(tmp_path / "g.png", rgb)),
        },
    }
    br = "BR-Gen/Forged/LaMa/Stuff/COCO/1_stuff.png"
    brm = "BR-Gen/Mask/Stuff/COCO/1_stuff.png"
    _write(tmp_path / "br" / br, rgb)
    _write(tmp_path / "br" / brm, m, "L")
    _write(tmp_path / "cham" / "Chameleon/test/0_real/r1.png", rgb)
    part_of = {
        "ds/1": "diffseg_heldout",
        "sid/t": "sidset_tampered",
        "gi/g": "generated",
        br: "brgen",
        "Chameleon/test/0_real/r1.png": "real",
    }
    return rows, part_of, br, brm


def test_every_part_loads_with_the_right_mask(tmp_path):
    rows, part_of, br, brm = _world(tmp_path)
    items = {
        it.uid: it
        for it in p6data.p6_items(
            list(part_of),
            part_of,
            rows,
            brgen_dir=tmp_path / "br",
            brgen_masks={br: brm},
            chameleon_dir=tmp_path / "cham",
        )
    }
    assert items["sid/t"].mask.sum() == 200  # only values > 127
    assert items[br].mask.sum() == 200
    assert items["ds/1"].mask.sum() == 100  # the DiffSeg30k training rule
    assert items["gi/g"].mask is None
    assert items["gi/g"].meta["label"] == "generated"
    cham = items["Chameleon/test/0_real/r1.png"]
    assert cham.mask is None
    assert cham.meta == {
        "part": "real",
        "label": "real",
        "source": "chameleon",
    }
    assert {items[u].meta["label"] for u in ("ds/1", "sid/t", br)} == {"edited"}
    assert items["ds/1"].image.shape == (40, 60, 3)


def test_brgen_hash_check_and_missing_files(tmp_path):
    rows, part_of, br, brm = _world(tmp_path)
    sha = lambda p: hashlib.sha256((tmp_path / "br" / p).read_bytes()).hexdigest()  # noqa: E731
    files = {br: {"sha256": sha(br), "mask": brm, "mask_sha256": sha(brm)}}
    assert p6data.check_brgen(files, tmp_path / "br") == []
    (tmp_path / "br" / brm).write_bytes(b"changed")
    assert p6data.check_brgen(files, tmp_path / "br") == [brm]
    with pytest.raises(p6data.P6DataError, match="neither"):
        list(p6data.p6_items(["x"], {"x": "generated"}, {}))
    with pytest.raises(p6data.P6DataError, match="Chameleon file missing"):
        list(p6data.p6_items(["c/x.png"], {"c/x.png": "real"}, {}, chameleon_dir=tmp_path))


def test_manifest_rows_keeps_only_p6():
    rows = [{"uid": "a"}, {"uid": "b"}]
    assert list(p6data.manifest_rows({"uids": ["b", "z"]}, rows)) == ["b"]


def test_cache_round_trip_and_refuses_a_changed_member(tmp_path):
    import tarfile

    from eviground.eval.tiled import Item

    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3)).astype(np.uint8)
    mask = np.zeros((20, 30), bool)
    mask[2:8, 3:9] = True
    items = [
        Item("a/1", img, mask, None, {"part": "brgen", "label": "edited"}),
        Item("b/2", img, None, None, {"part": "real", "label": "real"}),
    ]
    man = p6data.write_cache(items, tmp_path / "c.tar", tmp_path / "c.json")
    assert man["images"] == 2
    got = p6data.read_cache(tmp_path / "c.tar", tmp_path / "c.json", tmp_path / "out")
    back = {it.uid: it for it in p6data.cached_items(["a/1", "b/2"], got, tmp_path / "out")}
    assert (back["a/1"].image == img).all()
    assert (back["a/1"].mask == mask).all()
    assert back["b/2"].mask is None
    assert back["a/1"].meta == {"part": "brgen", "label": "edited"}
    # a tar rewritten with a changed member is refused
    with tarfile.open(tmp_path / "c.tar", "w") as tf:
        name = man["entries"]["a/1"]["image"]
        data = p6data.png_bytes(img + 1)
        info = tarfile.TarInfo(name)
        info.size = len(data)
        import io

        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(p6data.P6DataError):
        p6data.read_cache(tmp_path / "c.tar", tmp_path / "c.json", tmp_path / "out2")


# --------------------------------------------------------------------------------------
# Mask size alignment (Dhanush's notebook 14 cell 7 patch, moved into the loaders, 24 Nov)
# --------------------------------------------------------------------------------------


def _dhanush_rule(mask, shape):
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    return (F.interpolate(t, size=shape, mode="nearest")[0, 0] > 0).numpy()


def test_align_mask_same_size_is_unchanged():
    m = np.zeros((40, 60), bool)
    m[:10, :20] = True
    out, orig = p6data.align_mask(m, (40, 60))
    assert orig is None
    assert np.array_equal(out, m)


def test_align_mask_matches_the_notebook_14_rule():
    rng = np.random.default_rng(0)
    m = rng.random((20, 30)) > 0.7
    out, orig = p6data.align_mask(m, (40, 60))
    assert orig == [20, 30]
    assert out.shape == (40, 60)
    assert out.dtype == bool
    assert np.array_equal(out, _dhanush_rule(m, (40, 60)))


def test_align_mask_refuses_another_aspect_ratio():
    with pytest.raises(p6data.P6DataError, match="aspect ratio"):
        p6data.align_mask(np.zeros((40, 40), bool), (40, 60))


def test_align_mask_accepts_a_rounding_difference():
    out, orig = p6data.align_mask(np.zeros((33, 50), bool), (40, 60))  # 1.515 vs 1.5: 1%
    assert orig == [33, 50]
    assert out.shape == (40, 60)


def test_every_edited_part_aligns_a_half_size_mask(tmp_path):
    rows, part_of, br, brm = _world(tmp_path)
    half = np.zeros((20, 30), np.uint8)
    half[:5, :10] = 200
    ds_half = np.zeros((20, 30), np.uint8)
    ds_half[2:8, 2:8] = 6
    _write(tmp_path / "sid.mask.png", half, "L")
    _write(tmp_path / "ds1.mask.png", ds_half, "L")
    _write(tmp_path / "br" / brm, half, "L")
    items = {
        it.uid: it
        for it in p6data.p6_items(
            list(part_of),
            part_of,
            rows,
            brgen_dir=tmp_path / "br",
            brgen_masks={br: brm},
            chameleon_dir=tmp_path / "cham",
        )
    }
    for uid, src in (("sid/t", half > 127), (br, half > 127), ("ds/1", ds_half > 0)):
        it = items[uid]
        assert it.mask.shape == it.image.shape[:2]
        assert it.meta["mask_resized_from"] == [20, 30]
        assert np.array_equal(it.mask, _dhanush_rule(src, (40, 60)))
    ds = items["ds/1"]
    assert ds.editors.shape == ds.mask.shape
    assert set(np.unique(ds.editors[ds.mask])) == {6}
    assert not ds.editors[~ds.mask].any()
    summary = p6data.resize_summary(it.meta for it in items.values())
    assert summary["sidset_tampered"] == {
        "items": 1,
        "resized": 1,
        "aspect_mismatch": 0,
        "from_sizes": {"20x30": 1},
    }
    assert summary["generated"]["resized"] == 0


def test_a_misshaped_mask_stops_the_loader_outside_sidset(tmp_path):
    rows, part_of, br, brm = _world(tmp_path)
    _write(tmp_path / "ds1.mask.png", np.zeros((40, 40), np.uint8), "L")
    with pytest.raises(p6data.P6DataError, match="aspect ratio"):
        list(p6data.p6_items(["ds/1"], part_of, rows))


def _sidset_misshaped(tmp_path):
    rows, part_of, br, brm = _world(tmp_path)
    m = np.zeros((40, 40), np.uint8)
    m[:10, :10] = 200
    _write(tmp_path / "sid.mask.png", m, "L")
    return list(p6data.p6_items(["sid/t", "ds/1"], part_of, rows))


def test_a_misshaped_sidset_mask_is_resized_and_flagged(tmp_path):
    sid, ds = _sidset_misshaped(tmp_path)
    assert sid.mask.shape == sid.image.shape[:2]
    assert sid.meta["mask_aspect_mismatch"] is True
    assert sid.meta["mask_resized_from"] == [40, 40]
    assert "mask_aspect_mismatch" not in ds.meta
    assert p6data.resize_summary([sid.meta])["sidset_tampered"]["aspect_mismatch"] == 1


def test_mask_metrics_leave_out_flagged_masks_unless_asked(tmp_path):
    sid, ds = _sidset_misshaped(tmp_path)
    assert [it.uid for it in p6data.mask_metric_items([sid, ds])] == ["ds/1"]
    kept = p6data.mask_metric_items([sid, ds], include_mismatched=True)
    assert [it.uid for it in kept] == ["sid/t", "ds/1"]


def test_items_cached_before_the_flag_are_recognised_by_their_size():
    meta = {"part": "sidset_tampered", "mask_resized_from": [764, 1024]}
    assert p6data.is_aspect_mismatched(meta, (1024, 1024))
    assert not p6data.is_aspect_mismatched({"mask_resized_from": [512, 512]}, (1024, 1024))
    assert not p6data.is_aspect_mismatched({}, (1024, 1024))


def test_the_cache_keeps_the_aligned_mask(tmp_path):
    rows, part_of, br, brm = _world(tmp_path)
    half = np.zeros((20, 30), np.uint8)
    half[:5, :10] = 200
    _write(tmp_path / "sid.mask.png", half, "L")
    items = list(p6data.p6_items(["sid/t"], part_of, rows))
    man = p6data.write_cache(items, tmp_path / "c.tar", tmp_path / "c.json")
    p6data.read_cache(tmp_path / "c.tar", tmp_path / "c.json", tmp_path / "out")
    (back,) = p6data.cached_items(["sid/t"], man, tmp_path / "out")
    assert np.array_equal(back.mask, items[0].mask)
    assert back.meta["mask_resized_from"] == [20, 30]
