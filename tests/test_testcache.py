"""Drive test cache (Phase A, notebook 16): build, publish with Drive checks, stage with hashes."""

from __future__ import annotations

import json
import tarfile

import numpy as np
import pytest
from PIL import Image

from eviground.data import testcache


def _rows(tmp_path, n=3):
    src = tmp_path / "src"
    src.mkdir()
    rows = []
    for i in range(n):
        img = src / f"im{i}.jpg"
        Image.fromarray(np.full((20 + i, 30, 3), 40 * i, np.uint8)).save(img, quality=80)
        row = {"uid": f"set/im{i}", "label": "edited" if i else "real", "path": str(img)}
        if i:
            mask = src / f"im{i}.mask.png"
            Image.fromarray(np.full((20 + i, 30), 255, np.uint8)).save(mask)
            row["mask_path"] = str(mask)
        rows.append(row)
    return rows


def _published(tmp_path):
    rows = _rows(tmp_path)
    built = testcache.build(rows, "p2", tmp_path / "local")
    man = testcache.publish(built, tmp_path / "local", tmp_path / "drive", seed=1)
    return rows, man


def test_build_keeps_the_original_bytes_and_hashes(tmp_path):
    rows = _rows(tmp_path)
    man = testcache.build(rows, "p2", tmp_path / "local")
    assert man["images"] == 3
    assert man["masks"] == 2
    rec = man["entries"]["set/im1"]
    assert rec["row"] == {"uid": "set/im1", "label": "edited"}
    with tarfile.open(tmp_path / "local" / "p2.tar") as tf:
        data = tf.extractfile(rec["image"]).read()
    with open(rows[1]["path"], "rb") as fh:
        assert data == fh.read()


def test_publish_rehashes_from_drive_and_decodes_one_member(tmp_path):
    _, man = _published(tmp_path)
    assert man["drive_sha256_ok"] is True
    check = man["decode_check"]
    assert check["uid"] in man["entries"]
    assert check["shape"][1] == 30
    on_drive = json.loads((tmp_path / "drive" / "p2_manifest.json").read_text())
    assert on_drive == man
    assert testcache.is_ready(tmp_path / "drive", "p2")


def test_stage_returns_local_rows_ready_for_the_loaders(tmp_path):
    rows, _ = _published(tmp_path)
    staged = testcache.stage(tmp_path / "drive", "p2", tmp_path / "data")
    assert [r["uid"] for r in staged] == [r["uid"] for r in rows]
    for old, new in zip(rows, staged, strict=True):
        with open(old["path"], "rb") as a, open(new["path"], "rb") as b:
            assert a.read() == b.read()
        assert ("mask_path" in new) == ("mask_path" in old)
        assert new["label"] == old["label"]


def test_stage_refuses_a_changed_tar(tmp_path):
    _published(tmp_path)
    tar = tmp_path / "drive" / "p2.tar"
    data = bytearray(tar.read_bytes())
    data[600] ^= 0xFF
    tar.write_bytes(bytes(data))
    with pytest.raises(testcache.CacheError, match="differs"):
        testcache.stage(tmp_path / "drive", "p2", tmp_path / "data")


def test_an_unpublished_set_is_not_ready(tmp_path):
    rows = _rows(tmp_path)
    testcache.build(rows, "p2", tmp_path / "local")
    (tmp_path / "drive").mkdir()
    for f in ("p2.tar", "p2_manifest.json"):
        (tmp_path / "drive" / f).write_bytes((tmp_path / "local" / f).read_bytes())
    assert not testcache.is_ready(tmp_path / "drive", "p2")
    with pytest.raises(testcache.CacheError, match="no verified tar"):
        testcache.stage(tmp_path / "drive", "p2", tmp_path / "data")


def test_publish_refuses_an_undecodable_member(tmp_path):
    src = tmp_path / "bad.png"
    src.write_bytes(b"not an image")
    built = testcache.build([{"uid": "x", "path": str(src)}], "bad", tmp_path / "local")
    with pytest.raises(testcache.CacheError, match="does not decode"):
        testcache.publish(built, tmp_path / "local", tmp_path / "drive")
    assert not (tmp_path / "drive" / "bad_manifest.json").exists()


def test_build_refuses_missing_files_and_duplicates(tmp_path):
    rows = _rows(tmp_path)
    with pytest.raises(testcache.CacheError, match="twice"):
        testcache.build(rows + rows[:1], "p2", tmp_path / "a")
    with pytest.raises(testcache.CacheError, match="missing"):
        testcache.build([{"uid": "y", "path": str(tmp_path / "nope.jpg")}], "p2", tmp_path / "b")
