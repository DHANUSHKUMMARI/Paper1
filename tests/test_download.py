"""ensure_data() on small synthetic archives: subset selection, resume, space checks,
verification before _DONE, no network once done, and per-session unpacking."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eviground.data import storage
from eviground.data.download import (
    DataStepError,
    ensure_data,
    plan_sid_split,
    read_done,
    select_sd14_subset,
    stage,
)
from eviground.data.fake_sources import (
    DIFFSEG_REPO,
    GENIMAGE_MIRROR_DIRS,
    FakeSources,
    dry_run_overrides,
    make_fake_world,
    make_genimage_validation,
    make_sid_legacy,
)
from eviground.data.registry import REGISTRY
from eviground.data.sources import kaggle_auth_header
from eviground.data.storage import Member
from eviground.env import MissingSecretError, Paths

KEYS = ["genimage_val", "genimage_sd14", "diffseg30k", "sidset"]


def _paths(tmp: Path) -> Paths:
    return Paths(root=tmp / "drive" / "Paper 1", local_data=tmp / "local")


def _world(tmp: Path, **kw) -> tuple[Paths, FakeSources, dict]:
    paths = _paths(tmp)
    n_val = make_genimage_validation(paths.datasets / REGISTRY["genimage_val"].folder)
    ov = dry_run_overrides(n_val)
    make_sid_legacy(paths.datasets / REGISTRY["sidset"].folder, ov["sidset"]["caps"])
    sources = make_fake_world(tmp / "remote", **kw)
    return paths, sources, ov


def _no_network():
    raise AssertionError("ensure_data created a network client although _DONE exists")


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture()
def done_world(tmp_path: Path):
    paths, sources, ov = _world(tmp_path)
    report = ensure_data(paths=paths, sources=sources, overrides=ov)
    return paths, sources, ov, report


def test_full_run_verifies_everything(done_world) -> None:
    paths, _sources, _ov, report = done_world
    assert {k: v["status"] for k, v in report.items()} == dict.fromkeys(KEYS, "verified")
    for key in KEYS:
        assert read_done(paths.datasets / REGISTRY[key].folder)["verified"] is True


def test_genimage_subset_on_drive(done_world) -> None:
    paths, _s, _ov, report = done_world
    folder = paths.datasets / REGISTRY["genimage_sd14"].folder
    rows = _rows(folder / "manifest.csv")
    assert len(rows) == 10 and sorted({r["label"] for r in rows}) == ["0", "1"]
    assert sum(r["label"] == "0" for r in rows) == 5
    assert all("/train/" in r["name"] for r in rows)  # never the val split
    assert all(len(r["sha256"]) == 64 for r in rows)
    assert sorted(p.name for p in folder.glob("*.tar")) == [
        "sd14_subset-00000.tar", "sd14_subset-00001.tar", "sd14_subset-00002.tar"]  # fmt: skip
    summary = report["genimage_sd14"]["summary"]
    assert summary["per_label"] == {"0": 5, "1": 5} and summary["local_archive_deleted"]
    assert not any((paths.local_data / "downloads").glob("genimage_sd14.zip*"))


def test_diffseg_zips_kept_as_published(done_world, tmp_path: Path) -> None:
    paths, _s, _ov, report = done_world
    folder = paths.datasets / "02_DiffSeg30k"
    for name in ("train.zip", "validation.zip"):
        assert (folder / name).read_bytes() == (tmp_path / "remote/diffseg" / name).read_bytes()
    assert report["diffseg30k"]["summary"]["train.zip"]["members"] == 12
    assert not list(folder.rglob("*.png"))  # no loose files on Drive


def test_sidset_verified_in_place(done_world) -> None:
    paths, sources, _ov, report = done_world
    folder = paths.datasets / "03_SID-Set"
    summary = report["sidset"]["summary"]
    assert summary["train"]["per_label"] == {"0": 6, "1": 6, "2": 6}
    assert summary["validation"]["per_label"] == {"0": 3, "1": 3, "2": 3}
    assert summary["train"]["masks_per_label"] == {"2": 6}
    assert summary["train"]["files_on_disk_not_in_manifest"] == 0
    assert not any("SID_Set" in c for c in sources.calls)  # nothing downloaded
    assert not list(folder.glob("*.tar"))  # nothing repacked


def _sid_failure(tmp_path: Path, damage) -> str:
    paths, _sources, ov = _world(tmp_path)
    folder = paths.datasets / "03_SID-Set"
    damage(folder)
    with pytest.raises(DataStepError) as err:
        ensure_data(datasets=["sidset"], paths=paths, sources=_no_network, overrides=ov)
    assert read_done(folder) is None
    return err.value.report["sidset"]["error"]


def _edit_manifest(folder: Path, split: str, edit) -> None:
    path = folder / f"{split}_manifest.csv"
    rows = _rows(path)
    rows = edit(rows)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mask", "label", "label_name"])
        writer.writeheader()
        writer.writerows(rows)


def test_sidset_missing_file_fails(tmp_path: Path) -> None:
    msg = _sid_failure(tmp_path, lambda f: (f / "train/real/images/train_real_00002.jpg").unlink())
    assert "1 listed files missing" in msg and "train_real_00002.jpg" in msg


def test_sidset_tampered_without_mask_fails(tmp_path: Path) -> None:
    def drop_mask(rows):
        rows[-1]["mask"] = ""
        return rows

    msg = _sid_failure(tmp_path, lambda f: _edit_manifest(f, "validation", drop_mask))
    assert "validation: 1 tampered images have no mask" in msg


def test_sidset_mask_on_real_image_fails(tmp_path: Path) -> None:
    def add_mask(rows):
        rows[0]["mask"] = rows[-1]["mask"]
        return rows

    msg = _sid_failure(tmp_path, lambda f: _edit_manifest(f, "train", add_mask))
    assert "real or fully synthetic images have a mask" in msg


def test_sidset_count_below_cap_fails(tmp_path: Path) -> None:
    msg = _sid_failure(tmp_path, lambda f: _edit_manifest(f, "train", lambda r: r[1:]))
    assert "counts per class {'0': 5} differ from 6" in msg


def test_sidset_label_map_mismatch_fails(tmp_path: Path) -> None:
    def swap_label(rows):
        rows[0]["label"] = "1"  # a real image labelled fully synthetic
        rows[6]["label"] = "0"
        return rows

    msg = _sid_failure(tmp_path, lambda f: _edit_manifest(f, "train", swap_label))
    assert "disagrees with the label map" in msg


def test_sidset_row_from_other_split_fails(tmp_path: Path) -> None:
    def borrow(rows):  # a train row pointing at a validation image of the same class
        rows[0]["image"] = "validation/real/images/validation_real_00000.jpg"
        return rows

    msg = _sid_failure(tmp_path, lambda f: _edit_manifest(f, "train", borrow))
    assert "outside the train/ folder" in msg and "validation_real_00000.jpg" in msg


def test_sidset_parquet_fallback(tmp_path: Path) -> None:
    """The kept fallback: download 10k/2k per class from parquet into tar files."""
    paths = _paths(tmp_path)
    sources = make_fake_world(tmp_path / "remote")
    ov = {"sidset": {"method": "hf_parquet_subset", "caps": {"train": 6, "validation": 3}}}
    report = ensure_data(datasets=["sidset"], paths=paths, sources=sources, overrides=ov)
    summary = report["sidset"]["summary"]
    assert summary["train"]["per_label"] == {"0": 6, "1": 6, "2": 6}
    assert summary["train"]["masks_per_label"] == {"2": 6}
    folder = paths.datasets / "03_SID-Set"
    assert len(_rows(folder / "train_manifest.csv")) == 18 and list(folder.glob("*.tar"))
    assert not list(folder.rglob("*.parquet"))  # parquet files never stored on Drive
    assert read_done(folder)["method"] == "hf_parquet_subset"
    staged = stage("sidset", paths=paths)  # unpacks the fallback's tar files
    assert staged["files"] == 18 + 6 + 9 + 3 and staged["archives"]


def test_second_run_makes_no_network_call(done_world) -> None:
    paths, _s, ov, _r = done_world
    report = ensure_data(paths=paths, sources=_no_network, overrides=ov)
    assert {v["status"] for v in report.values()} == {"done"}


def test_subset_is_seeded_and_reproducible(tmp_path: Path) -> None:
    outs = []
    for run in ("a", "b"):
        paths, sources, ov = _world(tmp_path / run)
        ov["sidset"]["method"] = "hf_parquet_subset"
        sid = paths.datasets / "03_SID-Set"
        for leftover in sid.glob("*_manifest.csv"):
            leftover.unlink()  # start the fallback from an empty folder
        ensure_data(datasets=["genimage_sd14", "sidset"], paths=paths, sources=sources,
                    overrides=ov)  # fmt: skip
        gi = paths.datasets / REGISTRY["genimage_sd14"].folder
        outs.append([(gi / "manifest.csv").read_text(), (sid / "train_manifest.csv").read_text(),
                     [p.read_bytes() for p in sorted(gi.glob("*.tar"))],
                     [p.read_bytes() for p in sorted(sid.glob("*.tar"))]])  # fmt: skip
    assert outs[0] == outs[1]


def test_interrupted_downloads_resume(tmp_path: Path) -> None:
    paths, sources, ov = _world(tmp_path, fail_after=300)
    with pytest.raises(DataStepError) as err:
        ensure_data(paths=paths, sources=sources, overrides=ov)
    first = err.value.report
    assert first["genimage_val"]["status"] == "verified"
    assert first["sidset"]["status"] == "verified"  # nothing to download
    for key in ("genimage_sd14", "diffseg30k"):
        assert first[key]["status"] == "error" and "simulated interruption" in first[key]["error"]
        assert read_done(paths.datasets / REGISTRY[key].folder) is None
    assert (paths.local_data / "downloads/genimage_sd14.zip.part").stat().st_size == 300
    # each file is interrupted once, so every rerun gets further; nothing is redone
    for _round in range(10):
        report = ensure_data(paths=paths, sources=sources, overrides=ov, raise_errors=False)
        errors = [k for k, v in report.items() if v["status"] == "error"]
        for key in errors:
            assert read_done(paths.datasets / REGISTRY[key].folder) is None
        if not errors:
            break
    assert {v["status"] for v in report.values()} <= {"verified", "done"}
    assert sources.calls.count("kaggle_download vtphatt2/GenImage-stable-diffusion-v1-4") == 2
    downloads = [c for c in sources.calls if c.startswith("hf_download")]
    assert all(downloads.count(c) == 2 for c in downloads)  # one interrupted + one resumed


def test_resume_at_shard_level(done_world) -> None:
    paths, sources, ov, _r = done_world
    folder = paths.datasets / REGISTRY["genimage_sd14"].folder
    (folder / "_DONE").unlink()
    kept = {p.name: p.stat().st_mtime_ns for p in folder.glob("*.tar")}
    before = len(sources.calls)
    ensure_data(datasets=["genimage_sd14"], paths=paths, sources=sources, overrides=ov)
    assert not any(c.startswith("kaggle") for c in sources.calls[before:])  # all shards fine
    (folder / "_DONE").unlink()
    (folder / "sd14_subset-00001.tar").unlink()
    ensure_data(datasets=["genimage_sd14"], paths=paths, sources=sources, overrides=ov)
    now = {p.name: p.stat().st_mtime_ns for p in folder.glob("*.tar")}
    assert now["sd14_subset-00000.tar"] == kept["sd14_subset-00000.tar"]
    assert now["sd14_subset-00002.tar"] == kept["sd14_subset-00002.tar"]
    assert "sd14_subset-00001.tar" in now and read_done(folder)


@pytest.mark.parametrize("disk", ["drive", "local"])
def test_space_check_stops_before_writing(tmp_path: Path, monkeypatch, disk: str) -> None:
    paths, sources, ov = _world(tmp_path)
    real = storage.free_bytes
    small = paths.root if disk == "drive" else paths.local_data

    def fake_free(path):
        return 1 if Path(path).is_relative_to(small) else real(path)

    monkeypatch.setattr(storage, "free_bytes", fake_free)
    with pytest.raises(DataStepError) as err:
        ensure_data(datasets=["genimage_sd14", "diffseg30k"], paths=paths, sources=sources,
                    overrides=ov)  # fmt: skip
    where = "Drive" if disk == "drive" else "the local disk"
    for key in ("genimage_sd14", "diffseg30k"):
        msg = err.value.report[key]["error"]
        assert "InsufficientSpaceError" in msg and f"on {where}" in msg
    assert not list((paths.datasets / "01_GenImage/SD_v1.4_subset").glob("*"))
    assert not list((paths.datasets / "02_DiffSeg30k").glob("*.zip"))
    assert not list((paths.local_data / "downloads").glob("*")) if disk == "local" else True


def test_checksum_mismatch_removes_file_and_writes_no_done(tmp_path: Path) -> None:
    paths, sources, ov = _world(tmp_path)
    real_info = sources.hf_file_info

    def wrong(repo, names):
        info = real_info(repo, names)
        info["train.zip"]["sha256"] = "0" * 64
        return info

    sources.hf_file_info = wrong
    with pytest.raises(DataStepError) as err:
        ensure_data(datasets=["diffseg30k"], paths=paths, sources=sources, overrides=ov)
    assert "VerificationError" in err.value.report["diffseg30k"]["error"]
    folder = paths.datasets / "02_DiffSeg30k"
    assert not (folder / "train.zip").exists() and read_done(folder) is None


def test_genimage_validation_must_be_complete(tmp_path: Path) -> None:
    paths, sources, ov = _world(tmp_path)
    folder = paths.datasets / REGISTRY["genimage_val"].folder
    for p in (folder / "imagenet_ai_0424_wukong/val/ai").iterdir():
        p.unlink()
    with pytest.raises(DataStepError) as err:
        ensure_data(datasets=["genimage_val"], paths=paths, sources=_no_network, overrides=ov)
    msg = err.value.report["genimage_val"]["error"]
    assert "wukong/ai" in msg and "expected" in msg
    assert (folder / "_DONE").read_text() == "vtphatt2/GenImage-Validation"  # untouched


def test_legacy_marker_is_kept(done_world) -> None:
    paths, *_ = done_world
    done = read_done(paths.datasets / REGISTRY["genimage_val"].folder)
    assert done["legacy_marker"] == "vtphatt2/GenImage-Validation"
    assert done["summary"]["per_generator"]["glide"] == {"nature": 2, "ai": 2}


def test_deferred_datasets_are_not_downloaded(tmp_path: Path) -> None:
    report = ensure_data(datasets=["brgen", "chameleon"], paths=_paths(tmp_path),
                         sources=_no_network)  # fmt: skip
    assert {v["status"] for v in report.values()} == {"deferred"}


def test_stage_unpacks_locally_and_resumes(done_world) -> None:
    paths, *_ = done_world
    res = stage("diffseg30k", paths=paths)
    assert res["files"] == 12 + 6 and not res["skipped"]
    assert (paths.local_data / "diffseg30k/train/masks/train_000001.png").is_file()
    assert stage("diffseg30k", paths=paths)["skipped"]
    assert stage("genimage_sd14", paths=paths)["files"] == 10
    sid = stage("sidset", paths=paths)  # loose files on Drive: copied, checked by size
    assert sid["files"] == 18 + 6 + 9 + 3 and sid["copied"] == sid["files"]  # + tampered masks
    assert (
        paths.local_data / "sidset/train/tampered/masks/train_tampered_00000_mask.png"
    ).is_file()
    assert stage("sidset", paths=paths)["copied"] == 0
    assert stage("genimage_val", paths=paths)["staged"] is False


def test_stage_requires_done(tmp_path: Path) -> None:
    with pytest.raises(storage.VerificationError):
        stage("diffseg30k", paths=_paths(tmp_path))


def test_select_subset_rejects_small_archives() -> None:
    members = [Member(f"x/train/ai/{i}.png", 1) for i in range(3)]
    members += [Member(f"x/train/nature/{i}.JPEG", 1) for i in range(9)]
    params = {**REGISTRY["genimage_sd14"].params, "per_class": 5}
    with pytest.raises(storage.VerificationError, match="too few"):
        select_sd14_subset(members, params)


def test_plan_sid_split_uses_few_shards() -> None:
    shards = {f"s{i}": [i % 3] * 10 for i in range(9)}  # one label per shard, 30 per label
    plan = plan_sid_split(shards, cap=15, labels=[0, 1, 2], seed=42)
    assert plan["counts"] == {"0": 15, "1": 15, "2": 15} and plan["shortfall"] == {}
    assert len(plan["rows"]) == 6  # 2 shards per label, the second one subsampled
    short = plan_sid_split(shards, cap=40, labels=[0, 1, 2], seed=42)
    assert short["shortfall"] == {"0": 10, "1": 10, "2": 10}
    assert plan == plan_sid_split(shards, cap=15, labels=[0, 1, 2], seed=42)


def test_kaggle_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(MissingSecretError, match="KAGGLE_API_TOKEN.*KAGGLE_USERNAME"):
        kaggle_auth_header()
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert kaggle_auth_header().startswith("Basic ")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_x")
    assert kaggle_auth_header() == "Bearer KGAT_x"


def test_missing_kaggle_secret_stops_cleanly(tmp_path: Path, monkeypatch, caplog) -> None:
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    paths = _paths(tmp_path)
    report = ensure_data(datasets=["genimage_sd14"], paths=paths, raise_errors=False,
                         overrides={"genimage_sd14": {"subset_gb_estimate": 0.001}})  # fmt: skip
    assert "MissingSecretError" in report["genimage_sd14"]["error"]
    assert read_done(paths.datasets / REGISTRY["genimage_sd14"].folder) is None


def test_report_is_json(done_world) -> None:
    json.dumps(done_world[3])
    assert DIFFSEG_REPO in REGISTRY["diffseg30k"].source


# The Kaggle mirror's real layout on Drive (12 Nov 2026): Validation/GenImage/<Generator>/val/...


def test_genimage_validation_mirror_names(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    folder = paths.datasets / REGISTRY["genimage_val"].folder
    n = make_genimage_validation(folder, generator_dirs=GENIMAGE_MIRROR_DIRS, prefix="GenImage")
    stray = folder / "GenImage/BigGAN/val/ai/999_sdv4_00000.png"  # a name that mentions SD v1.4
    stray.write_bytes((folder / "GenImage/BigGAN/val/ai/000_BigGAN.PNG").read_bytes())
    report = ensure_data(datasets=["genimage_val"], paths=paths, sources=_no_network,
                         overrides={"genimage_val": {"expected_images": n + 1}})  # fmt: skip
    summary = report["genimage_val"]["summary"]
    assert summary["unassigned_images"] == 0
    assert summary["per_generator"]["sd14"] == {"nature": 2, "ai": 2}
    assert summary["per_generator"]["sd15"] == {"nature": 2, "ai": 2}
    assert summary["per_generator"]["biggan"] == {"nature": 2, "ai": 3}  # by folder, not name


def test_genimage_validation_reports_missing_mirror_generators(tmp_path: Path) -> None:
    """What Drive holds on 12 Nov 2026: 5 of 8 generators."""
    paths = _paths(tmp_path)
    folder = paths.datasets / REGISTRY["genimage_val"].folder
    present = ("ADM", "BigGAN", "Midjourney", "stable_diffusion_v_1_5", "wukong")
    make_genimage_validation(folder, generator_dirs=present, prefix="GenImage")
    with pytest.raises(DataStepError) as err:
        ensure_data(datasets=["genimage_val"], paths=paths, sources=_no_network)
    msg = err.value.report["genimage_val"]["error"]
    assert "no images for glide/nature, glide/ai, sd14/nature, sd14/ai, vqdm/nature, vqdm/ai" in msg
    assert "not under exactly one generator" not in msg
