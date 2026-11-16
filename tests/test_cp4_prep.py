"""CP4 preparation: the development hold-out and the BR-Gen listing, subset and extraction."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from eviground.data import brgen
from eviground.eval import devsplit


def _rows():
    rows = []
    for gen in ("ADM", "BigGAN", "Midjourney"):
        for label, n in (("real", 50), ("generated", 50)):
            for i in range(n):
                uid = f"genimage_val/{gen}/{label}/{i:03d}.png"
                rows.append(
                    {
                        "uid": uid,
                        "role": "test_P1",
                        "split": "test",
                        "generator": gen,
                        "label": label,
                        "sha256": f"s-{uid}",
                        "pixel_sha256": f"p-{uid}",
                    }
                )
    # one ImageNet real image repeated in three generator folders: one group
    for gen in ("ADM", "BigGAN", "Midjourney"):
        rows.append(
            {
                "uid": f"genimage_val/{gen}/real/shared.jpg",
                "role": "test_P1",
                "split": "test",
                "generator": gen,
                "label": "real",
                "sha256": "s-shared",
                "pixel_sha256": "p-shared",
            }
        )
    for base, n in (("real", 30), ("ai", 20)):
        for i in range(n):
            uid = f"diffseg30k/validation/{base}/{i:03d}"
            rows.append(
                {
                    "uid": uid,
                    "role": "test_P2",
                    "split": "test",
                    "base_type": base,
                    "label": "edited",
                    "sha256": f"s-{uid}",
                    "pixel_sha256": f"p-{uid}",
                }
            )
    # rows outside P1 and P2 are ignored
    rows.append({"uid": "x/train", "role": "train_pool", "split": "train", "label": "real"})
    rows.append({"uid": "x/p3b", "role": "test_P3b", "split": "test", "label": "real"})
    return rows


def test_dev_split_takes_20_percent_per_stratum_and_is_deterministic():
    rows = _rows()
    a = devsplit.build_dev_split(rows)
    b = devsplit.build_dev_split(list(reversed(rows)))
    assert a["dev_sha256"] == b["dev_sha256"]
    assert a["strata"]["P1/ADM/generated"] == {"images": 50, "dev": 10}
    assert a["strata"]["P2/real"] == {"images": 30, "dev": 6}
    assert a["strata"]["P2/ai"] == {"images": 20, "dev": 4}
    for s, c in a["strata"].items():
        assert abs(c["dev"] - round(0.2 * c["images"])) <= 1, s
    assert not any(u.startswith("x/") for u in a["dev_uids"])
    assert a["totals"]["P2"] == {"images": 50, "dev": 10}
    assert a["reported_uids_count"] + len(a["dev_uids"]) == 3 * 101 + 50
    other = devsplit.build_dev_split(rows, salt="another")
    assert other["dev_sha256"] != a["dev_sha256"]


def test_duplicates_stay_on_one_side():
    split = devsplit.build_dev_split(_rows())
    shared = [u for u in split["dev_uids"] if u.endswith("shared.jpg")]
    assert len(shared) in (0, 3)
    assert split["duplicate_groups"] == 1
    assert split["groups_across_strata"] == 1


def test_dev_split_save_load_verify(tmp_path):
    rows = _rows()
    split = devsplit.build_dev_split(rows)
    path = devsplit.save(split, tmp_path / "cp4_dev_split.json")
    again = devsplit.load(path)
    assert again["dev_uids"] == split["dev_uids"]
    assert devsplit.verify(again, rows)["same"] is True
    assert devsplit.verify(again, rows[:-60])["same"] is False
    assert "dev_uids" not in devsplit.summary(split)
    broken = dict(split, dev_uids=split["dev_uids"][1:])
    devsplit.save(broken, path)
    with pytest.raises(ValueError, match="SHA-256"):
        devsplit.load(path)


def test_file_list_hash_ignores_order():
    assert devsplit.file_list_sha256(["b", "a"]) == devsplit.file_list_sha256(["a", "b"])


# ------------------------------------------------------------------ BR-Gen


def test_parse_summarize_pair_and_subset_a_listing():
    entries = brgen.parse_slt(brgen.fake_listing(4))
    assert len(entries) == 2 * 2 * 4 * 2
    paths = [e["path"] for e in entries]
    summary = brgen.summarize(entries)
    assert summary["kinds"] == {"image": 16, "mask": 16}
    assert summary["directories"]["BR-Gen/methodA"]["files"] == 16
    pairs = brgen.pair_masks(paths)
    assert len(pairs["pairs"]) == 16
    assert pairs["images_without_mask"] == 0
    img = "BR-Gen/methodB/object/image/0002.png"
    assert pairs["pairs"][img] == "BR-Gen/methodB/object/mask/0002.png"

    def stratum(p):
        return "/".join(p.split("/")[1:3])

    sub = brgen.subset(sorted(pairs["pairs"]), 2, stratum)
    assert len(sub["items"]) == 8
    assert all(c == {"available": 4, "chosen": 2} for c in sub["strata"].values())
    assert brgen.subset(list(reversed(sorted(pairs["pairs"]))), 2, stratum) == sub
    assert brgen.subset(sorted(pairs["pairs"]), 2, stratum, salt="x")["sha256"] != sub["sha256"]


def test_mask_suffix_pairs_in_the_same_folder():
    p = brgen.pair_masks(["d/a.png", "d/a_mask.png", "d/b.jpg", "d/b_gt.png", "d/c.png"])
    assert p["pairs"] == {"d/a.png": "d/a_mask.png", "d/b.jpg": "d/b_gt.png"}
    assert p["images_without_mask"] == 1


@pytest.mark.skipif(
    not (brgen.seven_zip() and shutil.which("zip")), reason="needs 7z and zip installed"
)
def test_list_and_extract_members_of_a_split_zip(tmp_path):
    src = tmp_path / "src"
    for i in range(3):
        for kind, size in (("image", 300_000), ("mask", 1_000)):
            f = src / "BR-Gen" / "m1" / kind / f"{i}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(os.urandom(size))  # incompressible: the zip really splits
    subprocess.run(
        ["zip", "-q", "-r", "-s", "300k", str(tmp_path / "BR-Gen_split.zip"), "BR-Gen"],
        cwd=src,
        check=True,
    )
    assert (tmp_path / "BR-Gen_split.z01").is_file()  # really split
    entries = brgen.list_archive(tmp_path / "BR-Gen_split.zip")
    files = sorted(e["path"] for e in entries if not e["is_dir"])
    assert len(files) == 6
    out = brgen.extract(
        tmp_path / "BR-Gen_split.zip",
        ["BR-Gen/m1/image/1.png", "BR-Gen/m1/mask/1.png"],
        tmp_path / "out",
    )
    assert [p.stat().st_size for p in out] == [300_000, 1_000]
    assert sorted(p.name for p in Path(tmp_path / "out").rglob("*.png")) == ["1.png", "1.png"]
