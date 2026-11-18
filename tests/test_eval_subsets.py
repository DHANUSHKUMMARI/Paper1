from __future__ import annotations

import io
import zipfile

import pytest
from PIL import Image

from eviground.data import chameleon
from eviground.eval import subsets


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _zip(path, n_real, n_fake, extra=()):
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n_real):
            zf.writestr(f"Chameleon/test/0_real/r{i}.png", _png())
        for i in range(n_fake):
            zf.writestr(f"Chameleon/test/1_fake/f{i}.png", _png())
        for name in extra:
            zf.writestr(name, b"x")
    return path


def test_labels_from_the_path():
    assert chameleon.label_of("Chameleon/test/0_real/a.jpg") == "real"
    assert chameleon.label_of("x/1_fake/b.png") == "generated"
    assert chameleon.label_of("x/other/c.png") is None
    assert not chameleon.is_image("__MACOSX/x/0_real/._a.png")


def test_inspect_unpack_and_read(tmp_path):
    z = _zip(tmp_path / "Chameleon.zip", 6, 4, extra=["Chameleon/readme.txt"])
    report = chameleon.inspect_zip(z, expect_images=10)
    assert report["images"] == 10
    assert report["per_class"] == {"generated": 4, "real": 6}
    assert report["other_members"] == 1
    names = chameleon.unpack(z, tmp_path / "out")
    assert len(names) == 10
    assert chameleon.unreadable(tmp_path / "out", names) == []
    (tmp_path / "out" / names[0]).write_bytes(b"not an image")
    assert chameleon.unreadable(tmp_path / "out", names) == [names[0]]


def test_inspect_stops_on_missing_corrupt_or_wrong_count(tmp_path):
    with pytest.raises(chameleon.ChameleonError, match="missing"):
        chameleon.inspect_zip(tmp_path / "Chameleon.zip")
    (tmp_path / "bad.zip").write_bytes(b"not a zip")
    with pytest.raises(chameleon.ChameleonError, match="not a readable zip"):
        chameleon.inspect_zip(tmp_path / "bad.zip")
    z = _zip(tmp_path / "small.zip", 3, 3)
    with pytest.raises(chameleon.ChameleonError, match="expected about"):
        chameleon.inspect_zip(z, expect_images=26000)


def test_p4_subset_balanced_and_stratified_and_deterministic():
    items = [{"uid": f"r{i}", "label": "real"} for i in range(30)]
    items += [{"uid": f"f{i}", "label": "generated"} for i in range(20)]
    a = subsets.p4_subset(items, total=20)
    assert a["per_class"] == {"generated": 10, "real": 10}
    assert a["rule"] == "balanced"
    assert a == subsets.p4_subset(list(reversed(items)), total=20)
    b = subsets.p4_subset(items, total=49)  # 24 + 25 impossible: proportional
    assert b["per_class"] == {"generated": 20, "real": 29}
    c = subsets.p4_subset(items, total=46)  # 23 each impossible: proportional
    assert c["rule"].startswith("stratified")
    assert sum(c["per_class"].values()) == 46
    assert c["per_class"]["generated"] <= 20


def _p1_rows(n):
    rows = []
    for g in ("adm", "sd14"):
        for lab in ("real", "generated"):
            rows += [
                {
                    "uid": f"{g}/{lab}/{i}",
                    "role": "test_P1",
                    "split": "test",
                    "generator": g,
                    "label": lab,
                }
                for i in range(n)
            ]
    return rows


def test_p1_subset_excludes_dev_and_takes_per_label():
    rows = _p1_rows(10)
    dev = [r["uid"] for r in rows if r["uid"].endswith("/0")]
    s = subsets.p1_test_subset(rows, dev, per_label=5)
    assert s["images"] == 20
    assert not set(s["uids"]) & set(dev)
    assert set(s["per_stratum"].values()) == {5}
    assert s["available"]["adm/real"] == 9
    assert s["sha256"] == subsets.p1_test_subset(rows, dev, per_label=5)["sha256"]
    with pytest.raises(ValueError, match="fewer than"):
        subsets.p1_test_subset(rows, dev, per_label=10)


def test_a_stored_subset_with_an_edited_uid_list_is_not_intact():
    items = [{"uid": f"r{i}", "label": "real"} for i in range(6)]
    items += [{"uid": f"f{i}", "label": "generated"} for i in range(6)]
    s = subsets.p4_subset(items, total=6)
    assert subsets.intact(s)
    assert not subsets.intact({**s, "uids": s["uids"][:-1]})
