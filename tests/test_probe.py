"""Data probe on a miniature, partially downloaded copy of the Drive datasets folder."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from eviground.data import probe as probe_mod
from eviground.data.probe import fetch_hf_cards, format_summary, run_probe
from eviground.data.synthetic import FAKE_MASK_VALUES, make_fake_drive

PER_FOLDER = 4


def _no_network(*_args, **_kwargs):
    raise AssertionError("the probe made a network call")


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_fake_drive(tmp_path_factory.mktemp("drive") / "datasets", per_folder=PER_FOLDER)


@pytest.fixture(scope="module")
def report(fake_root: Path) -> dict:
    mp = pytest.MonkeyPatch()
    mp.setattr(probe_mod.urllib.request, "urlopen", _no_network)
    try:
        yield run_probe(fake_root, sample_headers=50, sample_blockiness=5, sample_masks=50)
    finally:
        mp.undo()


def _group(report: dict, suffix: str) -> dict:
    groups = report["groups"] + report.get("archives", {}).get("groups", [])
    (g,) = [g for g in groups if g["dir"].endswith(suffix)]
    return g


def _status(report: dict, item: str) -> dict:
    (row,) = [r for r in report["medium_profile"] if r["item"] == item]
    return row


def test_report_is_json_and_has_no_errors(report: dict) -> None:
    json.dumps(report)
    assert report["errors"] == []
    assert report["disk"]["free_gb"] > 0


def test_folder_inventory(report: dict) -> None:
    ds = report["datasets"]
    assert ds["01_GenImage"]["exists"] and ds["01_GenImage"]["files"] == 3 * 2 * PER_FOLDER + 1
    assert ds["07_FaceForensics++_optional"]["exists"] is False
    assert ds["01_GenImage"]["done_marker"] is None  # the marker sits in Validation/
    names = {t["name"] for t in report["top_level"]}
    assert "Chameleon.zip" in names


def test_genimage_present_and_missing(report: dict) -> None:
    val = report["genimage"]["subsets"]["Validation"]
    assert val["generators_found"] == ["adm", "biggan", "glide"]
    assert set(val["generators_missing"]) == {"midjourney", "sd14", "sd15", "vqdm", "wukong"}
    assert val["done_marker"] == "vtphatt2/GenImage-Validation"
    assert _status(report, "GenImage Validation")["status"] == "partial"
    assert _status(report, "GenImage SD v1.4 subset")["status"] == "missing"


def test_group_stats_per_source_and_class(report: dict) -> None:
    ai = _group(report, "imagenet_glide/val/ai")
    assert ai["format_counts"] == {"png": PER_FOLDER}
    assert ai["jpeg_quality_hist"] == {} and ai["top_sizes"] == {"320x320": PER_FOLDER}
    nature = _group(report, "imagenet_glide/val/nature")
    assert nature["format_counts"] == {"jpeg": PER_FOLDER}
    assert nature["jpeg_quality_hist"] == {"96": PER_FOLDER}
    assert nature["jpeg_quality_exact"] == PER_FOLDER
    assert nature["subsampling_counts"] == {"4:2:0": PER_FOLDER}
    assert nature["min_side_lt_256"] == 1
    assert nature["ext_counts"] == {".jpeg": PER_FOLDER}
    assert nature["blockiness"]["n"] == PER_FOLDER and ai["blockiness"]["n"] == PER_FOLDER


def test_diffseg_zip_contents(report: dict) -> None:
    arcs = {a["path"]: a for a in report["archives"]["archives"]}
    for split, values in FAKE_MASK_VALUES.items():
        a = arcs[f"02_DiffSeg30k/{split}.zip"]
        assert a["kind"] == "zip" and a["members"] == 2 * PER_FOLDER
        assert a["per_folder"] == {f"{split}/images": PER_FOLDER, f"{split}/masks": PER_FOLDER}
        (m,) = [
            m for m in report["archives"]["masks"] if m["dir"] == f"02_DiffSeg30k:{split}/masks"
        ]
        assert m["masks"] == m["images"] == PER_FOLDER
        assert m["distinct_values"] == ["0", *map(str, values)]
        assert m["mask_format_counts"] == {"png": PER_FOLDER}
        assert m["mask_mode_counts"] == {"L": PER_FOLDER}
        assert m["all_zero_masks"] == 1
        assert m["mask_matches_image_size"] == {"True": PER_FOLDER}
        assert m["masks_without_image"] == m["images_without_mask"] == 0
        g = _group(report, f"02_DiffSeg30k:{split}/images")
        assert g["format_counts"] == {"png": PER_FOLDER} and g["blockiness"]["n"] == PER_FOLDER
        row = _status(report, f"DiffSeg30k {split}.zip")
        assert row["status"] == "not verified"  # ensure_data has not checked the checksum
    legacy = report["diffseg30k"]["legacy_folders"]["train"]
    assert legacy["images"] == legacy["masks"] == 0 and legacy["manifest_rows"] == 0


def test_sidset_counts_against_caps(report: dict) -> None:
    train = report["sidset"]["splits"]["train"]
    assert train["manifest_rows"] == 3 * PER_FOLDER
    assert train["labels"]["tampered"]["legacy_masks"] == PER_FOLDER
    assert train["labels"]["real"]["legacy_masks"] == 0
    assert train["labels"]["real"]["manifest_rows"] == PER_FOLDER
    assert _status(report, "SID-Set train/real")["status"] == "not verified"  # legacy files


def test_chameleon_counts(report: dict) -> None:
    ch = report["chameleon"]
    assert ch["folders"] == {"0_real": PER_FOLDER, "1_fake": PER_FOLDER - 1}
    assert ch["zip"]["per_class"] == ch["folders"]
    assert ch["folder_matches_zip"] is True
    assert _status(report, "Chameleon")["status"] == "present"


def test_brgen_archives(report: dict) -> None:
    assert [a["path"] for a in report["brgen"]["archives"]] == [
        "04_BR-Gen/BR-Gen.zip.001",
        "04_BR-Gen/BR-Gen.zip.002",
    ]
    assert _status(report, "BR-Gen")["status"] == "archive"


def test_probe_is_read_only_and_deterministic(fake_root: Path, report: dict) -> None:
    before = {p: p.stat().st_mtime_ns for p in fake_root.rglob("*")}
    again = run_probe(fake_root, sample_headers=50, sample_blockiness=5, sample_masks=50)
    after = {p: p.stat().st_mtime_ns for p in fake_root.rglob("*")}
    assert before == after
    strip = ("elapsed_s", "disk")
    assert {k: v for k, v in again.items() if k not in strip} == {
        k: v for k, v in report.items() if k not in strip
    }


def test_missing_root(tmp_path: Path) -> None:
    rep = run_probe(tmp_path / "nope")
    assert rep["exists"] is False


def test_summary_mentions_every_profile_item(report: dict) -> None:
    text = format_summary(report)
    for row in report["medium_profile"]:
        assert row["item"] in text


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _info_json(repo: str) -> bytes:
    return json.dumps(
        {"dataset_info": {"default": {"splits": {"train": {"num_examples": 7}}, "features": {}}}}
    ).encode()


def test_fetch_cards_with_token_only_after_401() -> None:
    calls: list[tuple[str, str | None]] = []
    secret = "hf_SECRET_value_123"

    def opener(req, timeout):
        auth = req.get_header("Authorization")
        calls.append((req.full_url, auth))
        if "README" in req.full_url and auth is None:
            raise urllib.error.HTTPError(req.full_url, 401, "gated", {}, None)
        if "README" in req.full_url:
            return _Resp(b"# Card\nmask value 2 = Kolors\nother line\n")
        return _Resp(_info_json(req.full_url))

    cards = fetch_hf_cards({"x": "org/ds"}, token_getter=lambda: secret, opener=opener)
    entry = cards["org/ds"]
    assert entry["readme"].startswith("# Card")
    assert entry["readme_lines_of_interest"] == ["mask value 2 = Kolors"]
    assert entry["split_sizes"] == {"train": 7}
    assert [a for _, a in calls] == [None, f"Bearer {secret}", None]
    assert secret not in json.dumps(cards)


def test_fetch_cards_failure_is_recorded_not_raised() -> None:
    def opener(req, timeout):
        raise urllib.error.URLError("offline")

    entry = fetch_hf_cards({"x": "org/ds"}, opener=opener)["org/ds"]
    assert entry["readme"] is None and entry["readme_status"].startswith("error")
    assert entry["split_sizes"] is None


def test_card_splits_feed_the_assessment(fake_root: Path) -> None:
    def opener(req, timeout):
        if "README" in req.full_url:
            return _Resp(b"card")
        body = {"train": {"num_examples": 100}, "validation": {"num_examples": 4}}
        return _Resp(json.dumps({"dataset_info": {"d": {"splits": body}}}).encode())

    rep = run_probe(fake_root, sample_headers=5, sample_blockiness=1, sample_masks=5,
                    fetch_cards=True, opener=opener)  # fmt: skip
    assert "100 examples on the card" in _status(rep, "DiffSeg30k train.zip")["expected"]


# Regression tests for review findings on the present/missing assessment (PR #1).


def _card_opener(split_sizes: dict[str, int]):
    def opener(req, timeout):
        if "README" in req.full_url:
            return _Resp(b"card")
        body = {k: {"num_examples": v} for k, v in split_sizes.items()}
        return _Resp(json.dumps({"dataset_info": {"d": {"splits": body}}}).encode())

    return opener


def _small_probe(root: Path, **kw) -> dict:
    return run_probe(root, sample_headers=5, sample_blockiness=1, sample_masks=5, **kw)


def test_card_split_missing_on_drive_is_reported(tmp_path: Path) -> None:
    root = make_fake_drive(tmp_path / "datasets", per_folder=2)
    opener = _card_opener({"train": 2, "validation": 2, "test": 9})
    rep = _small_probe(root, fetch_cards=True, opener=opener)
    row = _status(rep, "DiffSeg30k test.zip")
    assert row["status"] == "missing" and row["present"] == "no file"


def test_generator_folder_without_images_is_not_found(tmp_path: Path) -> None:
    root = make_fake_drive(tmp_path / "datasets", per_folder=2)
    (root / "01_GenImage/Validation/imagenet_ai_0419_vqdm/val/ai").mkdir(parents=True)
    val = _small_probe(root)["genimage"]["subsets"]["Validation"]
    assert "vqdm" not in val["generators_found"]
    assert "vqdm" in val["generators_missing"]
    assert val["generators_without_images"] == ["vqdm"]
    assert val["images_per_generator"] == {"adm": 4, "biggan": 4, "glide": 4}


def test_unpaired_diffseg_files_are_not_complete(tmp_path: Path) -> None:
    import zipfile

    root = make_fake_drive(tmp_path / "datasets", per_folder=2)
    ds = root / "02_DiffSeg30k"
    for name in ("train.zip", "validation.zip"):  # as if ensure_data had checked them
        (ds / f"{name}.verified.json").write_text('{"bytes": 1, "sha256": "x"}')
    assert _status(_small_probe(root), "DiffSeg30k validation.zip")["status"] == "present"
    src = ds / "validation.zip"
    with zipfile.ZipFile(src) as zin:
        items = [(i.filename, zin.read(i)) for i in zin.infolist()]
    with zipfile.ZipFile(src, "w") as zout:
        for name, data in items:
            name = name.replace("validation_000001.png", "validation_999999.png") \
                if "/masks/" in name else name  # fmt: skip
            zout.writestr(name, data)
    row = _status(_small_probe(root), "DiffSeg30k validation.zip")
    assert row["status"] == "partial"  # counts still match: 2 images, 2 masks
    assert "2 unpaired" in row["present"]


def test_all_generators_without_done_marker_is_partial(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    root = make_fake_drive(tmp_path / "datasets", per_folder=2)
    val_dir = root / "01_GenImage/Validation"
    for gen in ("imagenet_midjourney", "imagenet_ai_0419_sdv4", "imagenet_ai_0424_sdv5",
                "imagenet_ai_0419_vqdm", "imagenet_ai_0424_wukong"):  # fmt: skip
        (val_dir / gen / "val/ai").mkdir(parents=True)
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(val_dir / gen / "val/ai/0.png")
    assert _status(_small_probe(root), "GenImage Validation")["status"] == "present"
    (val_dir / "_DONE").unlink()  # interrupted copy: every generator has images, no marker
    row = _status(_small_probe(root), "GenImage Validation")
    assert row["status"] == "partial" and "_DONE marker absent" in row["present"]


def test_probe_after_ensure_data(tmp_path: Path) -> None:
    """End to end on fake sources: what ensure_data stores is what the probe reports."""
    from eviground.data.download import ensure_data
    from eviground.data.fake_sources import (
        dry_run_overrides,
        make_fake_world,
        make_genimage_validation,
        make_sid_legacy,
    )
    from eviground.env import Paths

    paths = Paths(root=tmp_path / "drive", local_data=tmp_path / "local")
    n_val = make_genimage_validation(paths.datasets / "01_GenImage/Validation")
    make_sid_legacy(paths.datasets / "03_SID-Set", dry_run_overrides(n_val)["sidset"]["caps"])
    ensure_data(paths=paths, sources=make_fake_world(tmp_path / "remote"),
                overrides=dry_run_overrides(n_val))  # fmt: skip
    rep = _small_probe(paths.datasets)
    assert rep["errors"] == []
    assert _status(rep, "GenImage Validation")["status"] == "present"
    assert "verified by ensure_data: yes" in _status(rep, "GenImage Validation")["present"]
    sub = _status(rep, "GenImage SD v1.4 subset")
    assert sub["status"] == "present" and "{'0': 5, '1': 5}" in sub["present"]
    for name in ("train.zip", "validation.zip"):
        assert _status(rep, f"DiffSeg30k {name}")["status"] == "present"
    assert _status(rep, "SID-Set train/real")["status"] == "partial"  # 6 of the real 10,000
    nature = _group(rep, "01_GenImage/SD_v1.4_subset:stable_diffusion_v_1_4/"
                         "imagenet_ai_0419_sdv4/train/nature")  # fmt: skip
    assert nature["n_images"] == 5 and nature["format_counts"] == {"jpeg": 5}
    (sid_masks,) = [m for m in rep["mask_folders"] if m["dir"] == "03_SID-Set/train/tampered/masks"]
    assert sid_masks["masks"] == sid_masks["images"] == 6
    assert sid_masks["images_without_mask"] == sid_masks["masks_without_image"] == 0
    assert sid_masks["distinct_values"] == ["0", "255"]
    assert _status(rep, "SID-Set train/tampered")["status"] == "partial"  # 6 of 10,000


def test_probe_counts_filled_generators(tmp_path: Path) -> None:
    """Validation with 5 loose generators + 3 fill tars reads as complete once verified."""
    from eviground.data.download import ensure_data
    from eviground.data.fake_sources import make_fake_world, make_genimage_validation
    from eviground.env import Paths

    paths = Paths(root=tmp_path / "drive", local_data=tmp_path / "local")
    folder = paths.datasets / "01_GenImage/Validation"
    present = ("ADM", "BigGAN", "Midjourney", "stable_diffusion_v_1_5", "wukong")
    make_genimage_validation(folder, generator_dirs=present, prefix="GenImage")
    before = _status(_small_probe(paths.datasets), "GenImage Validation")
    assert before["status"] == "partial"
    ensure_data(datasets=["genimage_val"], paths=paths, sources=make_fake_world(tmp_path / "r"),
                overrides={"genimage_val": {"expected_images": 32}})  # fmt: skip
    rep = _small_probe(paths.datasets)
    val = rep["genimage"]["subsets"]["Validation"]
    assert val["generators_missing"] == [] and len(val["fill_tars"]) == 3
    assert _status(rep, "GenImage Validation")["status"] == "present"
