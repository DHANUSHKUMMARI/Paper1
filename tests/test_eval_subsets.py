from __future__ import annotations

import io
import zipfile
from collections import Counter

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


def test_p1_subset_refuses_a_missing_stratum_or_generator_count():
    rows = _p1_rows(10)
    dev = [r["uid"] for r in rows if r["uid"].startswith("adm/real/")]  # the whole stratum
    with pytest.raises(ValueError, match="adm/real"):
        subsets.p1_test_subset(rows, dev, per_label=5)
    with pytest.raises(ValueError, match="generators found"):
        subsets.p1_test_subset(rows, [], per_label=5, generators=8)
    assert subsets.p1_test_subset(rows, [], per_label=5, generators=2)["images"] == 20


def _p6_world(n=500):
    rows = []
    for i in range(n):
        rows.append(
            {"uid": f"ds/p2/{i}", "source": "diffseg30k", "role": "test_P2", "label": "edited"}
        )
        rows.append(
            {"uid": f"sid/t/{i}", "source": "sidset", "role": "test_P3b", "label": "edited"}
        )
        rows.append(
            {
                "uid": f"sid/r/{i}",
                "source": "sidset",
                "role": "test_P3b",
                "label": "real",
                "sha256": f"sr{i}",
                "phash": 1000 + i,
            }
        )
    for g in ("adm", "biggan", "glide", "midjourney", "sd14", "sd15", "vqdm", "wukong"):
        for i in range(40):
            rows.append(
                {"uid": f"{g}/gen/{i}", "role": "test_P1", "label": "generated", "generator": g}
            )
            rows.append(
                {
                    "uid": f"{g}/real/{i}",
                    "role": "test_P1",
                    "label": "real",
                    "generator": g,
                    "sha256": f"{g}r{i}",
                    "phash": 0,
                }
            )
    for i in range(20):  # DiffSeg30k unedited validation: too few, so SID-Set real tops up
        rows.append(
            {
                "uid": f"ds/u/{i}",
                "source": "diffseg30k",
                "role": "unused_validation",
                "editors": "",
                "sha256": f"du{i}",
            }
        )
    rows.append(
        {"uid": "ds/v/edited", "source": "diffseg30k", "role": "unused_validation", "editors": "1"}
    )
    brgen = [
        f"BR-Gen/Forged/{m}/Background/COCO/{i}_background.png"
        for m in ("BrushNet", "LaMa", "MAT", "PowerPaint", "SDXL")
        for i in range(150)
    ]
    cham = [
        {"uid": f"cham/real/{i}", "label": "real", "sha256": f"c{i}", "phash": 0} for i in range(80)
    ]
    return rows, brgen, cham


def test_p6_mix_nested_subsets_and_training_check():
    rows, brgen, cham = _p6_world()
    # one exact match (adm/real/0); the pHash is 64 bits from every test row's
    train = [{"uid": "t", "sha256": "admr0", "phash": (1 << 64) - 1}]
    index = subsets.TrainingIndex(train)
    pools = subsets.p6_pools(rows, dev_uids=["ds/p2/0"], p1_uids=["adm/real/1"])
    assert "ds/v/edited" not in {r["uid"] for r in pools["diffseg_unedited"]}
    out = subsets.p6_subset(pools, brgen, cham, index)
    p6 = out["P6"]
    assert p6["per_part"] == subsets.P6_PARTS
    assert p6["images"] == 1500
    assert p6["real_sources"] == {"genimage": 50, "diffseg_unedited": 50, "chameleon": 50}
    assert p6["sidset_top_up"] == 30
    uids = set(p6["uids"])
    assert "ds/p2/0" not in uids and "adm/real/1" not in uids and "adm/real/0" not in uids
    assert not any("/SDXL/" in u for u in uids)
    gens = Counter(u.split("/")[0] for u in uids if "/gen/" in u)
    assert sorted(gens.values()) == [18, 18, 19, 19, 19, 19, 19, 19]
    assert set(out["P6_mllm"]["uids"]) <= uids and out["P6_mllm"]["images"] == 500
    dele = out["P6_deletion"]
    assert dele["images"] == 300 and set(dele["uids"]) <= uids
    assert set(dele["per_part"]) == set(subsets.P6_EDITED)
    assert all(p6["part_of"][u] in subsets.P6_EDITED for u in dele["uids"])
    assert subsets.p6_subset(pools, brgen, cham, index)["P6"]["sha256"] == p6["sha256"]


def test_training_index_refuses_exact_and_near_duplicates():
    index = subsets.TrainingIndex([{"sha256": "a", "phash": 0b1111}])
    assert not index.clean({"sha256": "a"})
    assert not index.clean({"sha256": "b", "phash": 0b1110})  # 1 bit away
    assert index.clean({"sha256": "b", "phash": (1 << 40) - 1})  # 36 bits away


def test_p6_stops_when_a_part_is_short():
    rows, brgen, cham = _p6_world(n=100)
    with pytest.raises(ValueError, match="short"):
        subsets.p6_subset(subsets.p6_pools(rows, [], []), brgen, cham, subsets.TrainingIndex([]))


def test_hashing_only_the_chameleon_head_gives_the_same_pick():
    rows, brgen, cham = _p6_world()
    # every 3rd Chameleon real image is a training duplicate
    cham = [{**c, "sha256": "dup" if i % 3 == 0 else c["sha256"]} for i, c in enumerate(cham)]
    index = subsets.TrainingIndex([{"sha256": "dup"}])
    pools = subsets.p6_pools(rows, [], [])
    full = subsets.p6_subset(pools, brgen, cham, index)["P6"]
    head = subsets.chameleon_candidates(cham, 76)
    assert subsets.p6_subset(pools, brgen, head, index)["P6"]["sha256"] == full["sha256"]
    assert full["skipped_near_training"]["chameleon"] > 0
