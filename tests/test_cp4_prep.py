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
    """BR-Gen's README layout: Au/ (authentic), Tp/ (tampered), Gt/ (``<stem>_mask.png``)."""
    entries = brgen.parse_slt(brgen.fake_listing(4))
    assert len(entries) == 2 * (4 + 4 * 2 * 2)
    paths = [e["path"] for e in entries]
    summary = brgen.summarize(entries)
    assert summary["kinds"] == {"authentic": 8, "image": 16, "mask": 16}
    assert summary["directories"]["BR-Gen/BRGen-Stuff-Val"]["files"] == 20
    pairs = brgen.pair_masks(paths)
    assert len(pairs["pairs"]) == 16
    assert pairs["images_without_mask"] == 0
    img = "BR-Gen/BRGen-Stuff-Val/Tp/SDXL_Stuff_COCO_00000000002.png"
    assert pairs["pairs"][img] == "BR-Gen/BRGen-Stuff-Val/Gt/SDXL_Stuff_COCO_00000000002_mask.png"
    assert brgen.method_of(img) == "SDXL"
    assert brgen.stratum_key(img, 2) == "BR-Gen/BRGen-Stuff-Val/SDXL"

    def stratum(p):
        return brgen.stratum_key(p, 2)

    sub = brgen.subset(sorted(pairs["pairs"]), 2, stratum)
    assert len(sub["items"]) == 8
    assert all(c == {"available": 4, "chosen": 2} for c in sub["strata"].values())
    assert brgen.subset(list(reversed(sorted(pairs["pairs"]))), 2, stratum) == sub
    assert brgen.subset(sorted(pairs["pairs"]), 2, stratum, salt="x")["sha256"] != sub["sha256"]
    assert len(brgen.sample(sub["items"], 3)) == 3


def test_kinds_do_not_match_mask_words_inside_other_names():
    assert brgen.kind("BR-Gen/BRGen-Stuff-Val/Tp/x.png") == "image"
    assert brgen.kind("data/target_images/x.png") == "image"  # 'gt' only as a folder or suffix
    assert brgen.kind("data/Gt/x_mask.png") == "mask"
    assert brgen.kind("data/Au/x.png") == "authentic"
    assert brgen.kind("notes.txt") == "other"


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


# -- completing the split archive (17 Nov 2026) -------------------------------------------


def _remote(tmp_path, sizes=(300, 300, 300, 120)):
    names = ["BR-Gen_split.z01", "BR-Gen_split.z02", "BR-Gen_split.z03", "BR-Gen_split.zip"]
    remote = tmp_path / "remote"
    remote.mkdir()
    files = {}
    for i, (name, n) in enumerate(zip(names, sizes, strict=True)):
        (remote / name).write_bytes(bytes([i + 1]) * n)
        files[name] = remote / name
    (remote / "README.md").write_text("card")
    files["README.md"] = remote / "README.md"
    return files


def _ctx(tmp_path, files):
    from eviground.data import download
    from eviground.data.fake_sources import FakeSources
    from eviground.env import Paths

    paths = Paths(root=tmp_path / "drive", local_data=tmp_path / "local")
    fake = FakeSources(hf={brgen.REPO: files})
    ctx = download.make_context(paths, fake, reserve_gb=0, drive_free=None, cache_wait_s=0)
    folder = paths.datasets / "04_BR-Gen"
    folder.mkdir(parents=True)
    return ctx, fake, folder


def test_complete_split_downloads_only_the_missing_volumes(tmp_path):
    files = _remote(tmp_path)
    ctx, fake, folder = _ctx(tmp_path, files)
    (folder / "BR-Gen_split.z01").write_bytes(files["BR-Gen_split.z01"].read_bytes())
    (folder / "BR-Gen_split.z02").write_bytes(b"\x00" * 100)  # truncated copy
    (folder / "BR-Gen_split.z99").write_bytes(b"x")  # not published: reported, kept
    out = brgen.complete_split(ctx, folder)
    assert out["kept"] == ["BR-Gen_split.z01"]
    assert out["size_only"] == ["BR-Gen_split.z01"]
    assert out["downloaded"] == ["BR-Gen_split.z02", "BR-Gen_split.z03", "BR-Gen_split.zip"]
    assert out["replaced"] == [{"name": "BR-Gen_split.z02", "bytes": 100}]
    assert out["extra"] == ["BR-Gen_split.z99"]
    assert "README.md" not in out["expected"]
    assert out["bytes_downloaded"] == 720
    assert not any(c.endswith("z01") for c in fake.calls if c.startswith("hf_download"))
    for name, src in files.items():
        if name.startswith(brgen.PART_PREFIX):
            assert (folder / name).read_bytes() == src.read_bytes(), name
    assert (folder / "BR-Gen_split.z99").exists()
    # a second run downloads nothing; the downloaded volumes are now stamped
    again = brgen.complete_split(ctx, folder)
    assert again["downloaded"] == []
    assert again["size_only"] == ["BR-Gen_split.z01"]


def test_complete_split_removes_a_volume_whose_hash_does_not_match(tmp_path, monkeypatch):
    from eviground.data.storage import VerificationError

    files = _remote(tmp_path)
    ctx, fake, folder = _ctx(tmp_path, files)
    real_info = fake.hf_file_info

    def bad_info(repo, names):
        info = real_info(repo, names)
        info["BR-Gen_split.z01"]["sha256"] = "0" * 64
        return info

    monkeypatch.setattr(fake, "hf_file_info", bad_info)
    with pytest.raises(VerificationError, match="z01"):
        brgen.complete_split(ctx, folder)
    assert not (folder / "BR-Gen_split.z01").exists()


def test_complete_split_refuses_before_writing_without_space(tmp_path):
    from eviground.data import download
    from eviground.data.storage import InsufficientSpaceError

    files = _remote(tmp_path)
    ctx, fake, folder = _ctx(tmp_path, files)
    ctx.drive_free = lambda: 10  # bytes free in the Drive quota
    ctx.reserve_gb = 0
    assert isinstance(ctx, download.Context)
    with pytest.raises(InsufficientSpaceError):
        brgen.complete_split(ctx, folder)
    assert not any(c.startswith("hf_download") for c in fake.calls)
    assert not list(folder.glob("BR-Gen_split.*"))


def test_complete_split_needs_published_volumes(tmp_path):
    from eviground.data.storage import VerificationError

    files = _remote(tmp_path)
    ctx, _fake, folder = _ctx(tmp_path, {"README.md": files["README.md"]})
    with pytest.raises(VerificationError, match="publishes no file"):
        brgen.complete_split(ctx, folder)


def test_complete_split_checks_the_total_space_before_the_first_download(tmp_path):
    from eviground.data.storage import InsufficientSpaceError

    files = _remote(tmp_path)  # missing: 300 + 300 + 300 + 120 bytes
    ctx, fake, folder = _ctx(tmp_path, files)
    ctx.drive_free = lambda: 500  # room for one volume, not for all four
    with pytest.raises(InsufficientSpaceError, match="4 missing volumes"):
        brgen.complete_split(ctx, folder)
    assert not any(c.startswith("hf_download") for c in fake.calls)
    assert not list(folder.glob("BR-Gen_split.*"))


def test_released_layout_parse_pair_strata_and_subset():
    """The archive as released (17 Nov 2026): Forged/<Method>/<Region>/<Source>/, one mask per
    source image and region under Mask/, and only image lists under RealImage/."""
    entries = brgen.parse_slt(brgen.fake_listing(2, layout="released"))
    paths = [e["path"] for e in entries]
    forged = [p for p in paths if "/Forged/" in p]
    assert len(forged) == 5 * 2 * 3 * 2
    f = "BR-Gen/Forged/SDXL/Stuff/ImageNet/imagenet_000001_stuff.png"
    assert brgen.parse_member(f) == {
        "kind": "forged",
        "method": "SDXL",
        "region": "Stuff",
        "source": "ImageNet",
        "name": "imagenet_000001_stuff.png",
    }
    assert brgen.kind(f) == "image"
    assert brgen.kind("BR-Gen/Mask/Stuff/ImageNet/imagenet_000001_stuff.png") == "mask"
    assert brgen.kind("BR-Gen/RealImage/COCO/COCO_image_list.txt") == "other"
    assert brgen.method_of(f) == "SDXL"
    assert brgen.stratum_key(f, 0) == "SDXL/Stuff/ImageNet"

    pairs = brgen.pair_masks(paths)
    assert len(pairs["pairs"]) == len(forged)  # every forged file pairs; masks are shared
    assert pairs["images_without_mask"] == 0
    assert pairs["masks_without_image"] == 0
    assert pairs["ambiguous"] == 0
    assert pairs["pairs"][f] == "BR-Gen/Mask/Stuff/ImageNet/imagenet_000001_stuff.png"

    summary = brgen.summarize(entries)
    assert summary["layout"].startswith("released")
    assert summary["kinds"] == {"image": 60, "mask": 12, "other": 3}
    assert len(summary["strata"]) == 30
    assert set(summary["strata"].values()) == {2}
    assert summary["mask_strata"]["Background/COCO"] == 2
    assert summary["source_images"] == {"COCO": 2, "ImageNet": 2, "Places": 2}
    assert len(summary["real_lists"]) == 3

    sub = brgen.subset(sorted(pairs["pairs"]), 1, lambda p: brgen.stratum_key(p, 0))
    assert len(sub["items"]) == 30
    assert all(c == {"available": 2, "chosen": 1} for c in sub["strata"].values())
