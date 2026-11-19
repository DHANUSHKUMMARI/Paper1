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
