"""CP1 end to end on a fake Drive with the real layouts, plus the COCO download step."""

from __future__ import annotations

import http.server
import io
import json
import shutil
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eviground.data import cp1
from eviground.data.download import DataStepError, ensure_data, read_done
from eviground.data.fake_cp1 import COCO_URLS, cp1_overrides, make_cp1_world
from eviground.data.sources import NetworkSources
from eviground.env import Paths

COCO_N = 40
CARD = {"train": 18, "validation": 12}


def _world(tmp: Path, **kw):
    paths = Paths(root=tmp / "drive" / "Paper 1", local_data=tmp / "local")
    sources = make_cp1_world(paths.datasets, tmp / "remote", coco_images=COCO_N, **kw)
    return paths, sources


def _ensure(paths, sources):
    return ensure_data(
        "cp1",
        paths=paths,
        sources=sources,
        overrides=cp1_overrides(COCO_N),
        stage_local=True,
        reserve_gb=0,
        drive_free=None,
    )


def _accept_local_edits(paths) -> None:
    """Record the local files as staged after a test edited them on purpose (the edit stands
    for a dataset that itself contains the problem being tested)."""
    from eviground.data.storage import STAGED_FILES, write_file_list

    for key in cp1.STAGED_KEYS:
        dest = paths.local_data / key
        files = {
            p.relative_to(dest).as_posix(): p.stat().st_size
            for p in dest.rglob("*")
            if p.is_file() and p.name not in ("_STAGED.json", STAGED_FILES)
        }
        write_file_list(dest, files)


def _cfg(**kw) -> cp1.CP1Config:
    base = {
        "workers": 1,
        "crops_per_class": 60,
        "blockiness_per_group": 20,
        "diffseg_card_splits": CARD,
    }
    return cp1.CP1Config(**{**base, **kw})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cp1")
    paths, sources = _world(tmp)
    _ensure(paths, sources)
    res = cp1.run_cp1(paths, _cfg(), out_dir=tmp / "out", cache_dir=tmp / "cache", progress=None)
    return tmp, paths, res


# --- COCO download ------------------------------------------------------------------------


def test_coco_downloaded_verified_and_staged(run):
    tmp, paths, _ = run
    done = read_done(paths.datasets / "06_COCO")
    assert done["method"] == "url_files"
    s = done["summary"]["val2017.zip"]
    assert s["matching_members"] == COCO_N and len(s["sha256"]) == 64
    assert "annotations/instances_val2017.json" not in str(list((tmp / "local").rglob("*.json")))
    assert len(list((tmp / "local/coco_val2017/val2017").glob("*.jpg"))) == COCO_N


def test_coco_wrong_member_count_is_rejected_and_removed(tmp_path):
    paths, sources = _world(tmp_path)
    with pytest.raises(DataStepError) as err:
        ensure_data(
            datasets=["coco_val2017"],
            paths=paths,
            sources=sources,
            overrides=cp1_overrides(COCO_N + 1),
            reserve_gb=0,
            drive_free=None,
        )
    assert "expected 41" in str(err.value)
    assert not (paths.datasets / "06_COCO/val2017.zip").exists()
    assert read_done(paths.datasets / "06_COCO") is None


def test_coco_corrupt_zip_is_rejected(tmp_path):
    paths, sources = _world(tmp_path)
    bad = tmp_path / "bad.zip"
    data = bytearray(sources.urls[COCO_URLS["val2017.zip"]].read_bytes())
    data[200:260] = b"\0" * 60  # damage a member's compressed bytes
    bad.write_bytes(bytes(data))
    sources.urls[COCO_URLS["val2017.zip"]] = bad
    with pytest.raises(DataStepError) as err:
        ensure_data(
            datasets=["coco_val2017"],
            paths=paths,
            sources=sources,
            overrides=cp1_overrides(COCO_N),
            reserve_gb=0,
            drive_free=None,
        )
    assert "val2017.zip" in str(err.value)
    assert not (paths.datasets / "06_COCO/val2017.zip").exists()


def test_coco_interrupted_download_resumes_and_skips_when_done(tmp_path):
    paths, sources = _world(tmp_path)
    sources.fail_after = 1000
    kw = {
        "paths": paths,
        "sources": sources,
        "overrides": cp1_overrides(COCO_N),
        "reserve_gb": 0,
        "drive_free": None,
        "raise_errors": False,
    }
    rep = ensure_data(datasets=["coco_val2017"], **kw)
    assert rep["coco_val2017"]["status"] == "error"
    assert (paths.datasets / "06_COCO/val2017.zip.part").stat().st_size == 1000
    for _ in range(3):  # each of the two files is interrupted once, then resumes
        rep = ensure_data(datasets=["coco_val2017"], **kw)
        if rep["coco_val2017"]["status"] == "verified":
            break
    assert rep["coco_val2017"]["status"] == "verified"
    assert sources.calls.count(f"url_download {COCO_URLS['val2017.zip']}") == 2
    n_calls = len(sources.calls)
    rep = ensure_data(datasets=["coco_val2017"], **kw)
    assert rep["coco_val2017"]["status"] == "done" and len(sources.calls) == n_calls


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    cut_first = 0  # close the first response after this many bytes
    served: list[str] = []

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = self.payload[start:]
        self.served.append(rng or "full")
        self.send_response(206 if rng else 200)
        if rng:
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).cut_first:
            n, type(self).cut_first = type(self).cut_first, 0
            self.wfile.write(body[:n])
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_network_url_download_resumes_with_range(tmp_path, monkeypatch):
    """The real downloader (no fake) against a local HTTP server that drops the connection."""
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    payload = bytes(range(256)) * 4000
    _RangeHandler.payload, _RangeHandler.cut_first, _RangeHandler.served = payload, 300_000, []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/val2017.zip"
        checks = []
        src = NetworkSources(retries=3)
        monkeypatch.setattr("eviground.data.sources.time.sleep", lambda s: None)
        total = src.url_download(
            url, tmp_path / "f.zip", lambda need, tot: checks.append((need, tot))
        )
    finally:
        server.shutdown()
    assert total == len(payload)
    assert (tmp_path / "f.zip").read_bytes() == payload
    assert not (tmp_path / "f.zip.part").exists()
    assert _RangeHandler.served[0] == "full" and _RangeHandler.served[1].startswith("bytes=")
    assert checks == [(len(payload), len(payload))]  # the space check ran once, before writing


# --- Claims-Edit reservation ----------------------------------------------------------------


def test_reserve_claims_edit_rules():
    inst = {
        "images": [{"id": i, "width": 100, "height": 100} for i in range(1, 21)],
        "categories": [
            {"id": 3, "name": "car"},
            {"id": 62, "name": "chair"},
            {"id": 1, "name": "person"},
        ],
        "annotations": [
            *[{"image_id": i, "category_id": 3, "area": 900, "iscrowd": 0} for i in range(1, 8)],
            *[{"image_id": i, "category_id": 62, "area": 600, "iscrowd": 0} for i in range(5, 12)],
            {"image_id": 12, "category_id": 3, "area": 100, "iscrowd": 0},  # too small
            {"image_id": 13, "category_id": 3, "area": 5000, "iscrowd": 1},  # crowd
            {"image_id": 14, "category_id": 1, "area": 5000, "iscrowd": 0},  # other category
        ],
    }
    params = {"n": 8, "categories": ["car", "chair"], "min_area_share": 0.05, "seed": 42}
    got = cp1.reserve_claims_edit(inst, params)
    assert len(got) == len(set(got)) == 8
    assert set(got) <= set(range(1, 12))
    assert got == cp1.reserve_claims_edit(inst, params)
    with pytest.raises(ValueError):
        cp1.reserve_claims_edit(inst, {**params, "n": 12})


# --- the CP1 run ------------------------------------------------------------------------------


def test_run_cp1_result_record(run):
    tmp, _, res = run
    assert set(res["criteria"]) == {
        "no_cross_split_duplicates",
        "class_counts_equal_manifests",
        "metadata_classifier_le_60",
    }
    assert res["criteria"]["no_cross_split_duplicates"] is True
    assert res["criteria"]["class_counts_equal_manifests"] is True
    assert res["verdict"] in ("PASS", "FAIL", "STOP_AND_ASK")
    json.dumps(res)  # JSON-ready
    for name in (
        "cp1_grid_real.png",
        "cp1_grid_generated.png",
        "cp1_grid_edited.png",
        "cp1_grid_edited_masks.png",
        "manifest_cp1.csv.gz",
    ):
        assert (tmp / "out" / name).is_file()
    with Image.open(tmp / "out/cp1_grid_real.png") as g:
        assert g.size == (1024, 1024)


def test_roles_splits_and_holdout(run):
    _, _, res = run
    roles = res["diffseg30k_masks"]["roles"]
    # fake editor sets: train cycles 9 sets (2 hold-out, 1 empty), validation 6 (4 hold-out);
    # the 2 empty-mask images are real negatives in the training pool (14 Nov 2026)
    assert roles == {
        "train/excluded_holdout": 4,
        "train/train_pool": 14,
        "validation/test_P2": 8,
        "validation/unused_validation": 4,
    }
    by = res["counts"]["by_source_split_label"]
    assert by["coco/test/real"] == 6 and len(res["coco_negatives"]["reserved_claims_edit_ids"]) == 6
    assert by["genimage_val/test/real"] == by["genimage_val/test/generated"] == 16
    assert by["sidset/test/edited"] == 5
    assert by.get("diffseg30k/train/real", 0) + by.get("diffseg30k/val/real", 0) == 2
    assert res["counts"]["by_role"]["sidset/excluded_sid_tampered"] == 10
    assert "sidset/train/edited" not in by and "sidset/val/edited" not in by
    assert res["diffseg30k_masks"]["distinct_values"] == list(range(9))
    assert res["diffseg30k_masks"]["values_outside_0_8"] == []
    assert res["sidset_masks"]["train"]["with_values_other_than_0_255"] == 10


def test_harmonization_follows_spec_table(run):
    _, _, res = run
    plan = res["harmonization_plan"]
    assert plan["genimage/real"].get("keep:real_jpeg", 0) > 0
    assert set(plan["genimage/generated"]) == {"compress:lossless"}
    assert plan["sidset/real"]["skip:below_q75"] > 0 and plan["sidset/real"]["keep:at_target"] > 0
    assert plan["sidset/real"]["compress:above_q75"] > 0
    assert "sidset/edited" not in plan  # SID-Set tampered left out of training
    assert set(plan["diffseg30k/real"]) <= {"compress:lossless", "skip:too_small"}
    # COCO negatives take DiffSeg30k sizes, including its few under 256 (skipped like them)
    assert plan["coco/real"]["compress:lossless"] > 0
    assert set(plan["coco/real"]) <= {"compress:lossless", "skip:too_small"}
    dist = res["quality_distribution_genimage_real"]
    assert dist["top_quality"] == 96
    # COCO negatives take the measured sizes of DiffSeg30k training images (14 Nov 2026)
    neg = res["coco_negatives"]
    fake_diffseg_sizes = {"320x288", "288x288", "256x256", "240x300"}
    assert set(neg["target_sizes_top"]) <= fake_diffseg_sizes
    assert set(neg["diffseg30k_sizes"]["top"]) <= fake_diffseg_sizes
    assert neg["resampling"] == {"downscaled": 34}  # the fake COCO photos are 427-640 px


def test_metadata_classifier_on_harmonized_crops(run):
    _, _, res = run
    meta = res["metadata_classifier"]
    rep = meta["criterion"]["real_vs_not_real"]
    assert rep["features"] == ["format", "jpeg_quality", "crop_w", "crop_h"]
    assert rep["distinct_values"]["format"] == 1  # every training file is JPEG after harmonization
    assert rep["distinct_values"]["crop_w"] == rep["distinct_values"]["crop_h"] == 1
    assert meta["criterion_score"] <= 0.60
    # the positive control: raw file facts and original size do give the label away
    # (DiffSeg30k's unedited images are PNG like its edited ones, so raw format alone no
    # longer separates every source; the control stays far above chance)
    assert meta["diagnostics"]["raw_files_positive_control"] > 0.8
    assert res["crops"]["errors"] == {}
    pcs = res["crops"]["per_class_source"]
    assert pcs["real/coco"] == pcs["real/diffseg30k"] == 10 and pcs["real/sidset"] == 20
    assert "edited/sidset" not in pcs


def test_scan_is_resumed_from_cache(tmp_path, monkeypatch):
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    first = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o1", cache_dir=tmp_path / "c", progress=None
    )

    def boom(row):
        raise AssertionError("scanned again")

    real_scan_row = cp1.scan_row
    monkeypatch.setattr(cp1, "scan_row", boom)
    second = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o2", cache_dir=tmp_path / "c", progress=None
    )
    assert first["exact_duplicates"] == second["exact_duplicates"]
    assert first["metadata_classifier"] == second["metadata_classifier"]
    # a file replaced in place (here: a SID-Set validation image copied over a train image)
    # must be scanned again, not served from the cache
    monkeypatch.setattr(cp1, "scan_row", real_scan_row)
    sid = paths.local_data / "sidset"
    src = sorted((sid / "validation/real/images").iterdir())[0]
    dst = sorted((sid / "train/real/images").iterdir())[0]
    shutil.copyfile(src, dst)
    _accept_local_edits(paths)
    third = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o3", cache_dir=tmp_path / "c", progress=None
    )
    assert third["training_leak_filter"]["exact_duplicates_of_test"] == 1


def test_training_copy_of_a_test_image_is_left_out(tmp_path):
    """Decision of 14 Nov 2026: duplicates of test images leave the training pool; test sets
    stay as published."""
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    # a SID-Set validation (test) image copied over a SID-Set train image
    sid = paths.local_data / "sidset"
    src = sorted((sid / "validation/real/images").iterdir())[0]
    dst = sorted((sid / "train/real/images").iterdir())[0]
    shutil.copyfile(src, dst)
    _accept_local_edits(paths)
    res = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", progress=None
    )
    leaks = res["training_leak_filter"]
    assert leaks["exact_duplicates_of_test"] == 1
    assert leaks["exact_by_source_and_test_set"] == {"sidset -> test_P3b": 1}
    assert res["counts"]["by_role"]["sidset/excluded_dup_test"] == 1
    assert res["counts"]["by_source_split_label"]["sidset/test/real"] == 5  # test set intact
    assert res["exact_duplicates"]["cross_split_total"] == 0
    assert res["criteria"]["no_cross_split_duplicates"] is True


def test_exact_duplicates_report_cross_split_groups():
    rows = [
        {"uid": "a", "split": "train", "sha256": "h1", "pixel_sha256": "p1"},
        {"uid": "b", "split": "test", "sha256": "h1", "pixel_sha256": "p1"},
        {"uid": "c", "split": "val", "sha256": "h2", "pixel_sha256": "p2"},
        {"uid": "d", "split": "val", "sha256": "h2", "pixel_sha256": "p2"},
    ]
    rep = cp1.exact_duplicates(rows)
    assert rep["sha256"]["cross_split_groups"] == 1 and rep["sha256"]["within_split_groups"] == 1
    assert rep["cross_split_total"] == 2


def test_duplicates_inside_the_training_pool_share_a_split():
    rows = [
        {
            "uid": f"s/{i:03d}",
            "source": "sidset",
            "label": "real",
            "role": "train_pool",
            "sha256": f"h{i}",
            "pixel_sha256": f"p{i}",
        }
        for i in range(60)
    ]
    for r in rows[:30]:  # 30 pairs of identical images
        r["sha256"] = r["pixel_sha256"] = "same" + r["uid"][-2:]
    for r, twin in zip(rows[:30], rows[30:], strict=True):
        twin["sha256"], twin["pixel_sha256"] = r["sha256"], r["pixel_sha256"]
    cp1.assign_splits(rows, seed=42)
    for r, twin in zip(rows[:30], rows[30:], strict=True):
        assert r["split"] == twin["split"]
    assert cp1.exact_duplicates(rows)["cross_split_total"] == 0


def test_coco_negative_near_duplicate_of_test_is_dropped(tmp_path):
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    test_img = sorted(
        (paths.datasets / "01_GenImage/Validation/GenImage/ADM/val/nature").iterdir()
    )[0]
    coco_dir = paths.local_data / "coco_val2017/val2017"
    reserved = set(
        cp1.reserve_claims_edit(
            cp1.coco_instances(paths.datasets), cp1_overrides(COCO_N)["coco_val2017"]["claims_edit"]
        )
    )
    victim = next(p for p in sorted(coco_dir.glob("*.jpg")) if int(p.stem) not in reserved)
    with Image.open(test_img) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=85)  # recompressed copy: near, not exact
    victim.write_bytes(buf.getvalue())
    _accept_local_edits(paths)
    res = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", progress=None
    )
    assert res["coco_negatives"]["near_dup_filter"] == {"coco -> test_P1": 1}
    assert res["counts"]["by_role"]["coco/excluded_near_dup_test"] == 1
    assert res["exact_duplicates"]["cross_split_total"] == 0


def test_near_duplicates_of_test_images_leave_training(tmp_path):
    """Decision of 14 Nov 2026: training images near a test image (pHash <= 6) are left out,
    so the >1% stop no longer triggers; the test set is unchanged."""
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    # copies of SID-Set train images placed in the SID-Set test split
    sid = paths.local_data / "sidset"
    trains = sorted((sid / "train/full_synthetic/images").iterdir())
    tests = sorted((sid / "validation/full_synthetic/images").iterdir())
    for a, b in zip(trains[:2], tests[:2], strict=True):
        with Image.open(a) as im:
            im.save(b, "PNG", compress_level=9)  # same name/format, other bytes, same pixels
    _accept_local_edits(paths)
    res = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", progress=None
    )
    leaks = res["training_leak_filter"]
    assert leaks["exact_duplicates_of_test"] == 2  # same pixels: caught by the pixel hash
    assert res["near_duplicates"]["test_P3b"]["counts"]["<=6"] == 0
    assert res["stop_and_ask"]["near_duplicates_over_1pct"] is False
    assert res["exact_duplicates"]["cross_split_total"] == 0
    assert res["counts"]["by_source_split_label"]["sidset/test/generated"] == 5


def test_near_duplicate_report_flags_more_than_one_percent():
    rows = [{"uid": f"t{i}", "role": "test_P3b", "phash": i * 977 + 3} for i in range(50)]
    rows += [{"uid": "x", "role": "train_pool", "split": "train", "phash": 3, "source": "sidset"}]
    rep = cp1.near_duplicates(rows, cp1.CP1Config())
    assert rep["test_P3b"]["counts"]["<=0"] == 1 and rep["test_P3b"]["stop_and_ask"] is True


def test_class_count_mismatch_is_reported(tmp_path):
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    # the dataset card promises one image more than the data holds
    res = cp1.run_cp1(
        paths,
        _cfg(diffseg_card_splits={"train": CARD["train"] + 1, "validation": CARD["validation"]}),
        out_dir=tmp_path / "o",
        cache_dir=tmp_path / "c",
        progress=None,
    )
    bad = [t for t in res["class_counts"]["table"] if not t["equal"]]
    assert bad == [
        {
            "source": "diffseg30k",
            "dataset_split": "train",
            "label": "edited",
            "expected": 19,
            "found": 18,
            "equal": False,
        }
    ]
    assert res["criteria"]["class_counts_equal_manifests"] is False


def test_missing_local_copies_stop_before_any_scan(tmp_path, monkeypatch):
    """A new Colab runtime has no local copies (14 Nov 2026): stop at once, say what to do."""
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)

    def boom(row):
        raise AssertionError("scanned without local copies")

    monkeypatch.setattr(cp1, "scan_row", boom)
    shutil.rmtree(paths.local_data / "genimage_sd14")
    victim = sorted((paths.local_data / "diffseg30k/train_copy").glob("*.image.png"))[0]
    victim.unlink()
    with pytest.raises(RuntimeError) as err:
        cp1.run_cp1(paths, _cfg(), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", progress=None)
    msg = str(err.value)
    assert "genimage_sd14: not staged" in msg
    assert f"diffseg30k: 1 missing (e.g. ['train_copy/{victim.name}'])" in msg
    assert "run Cell 3" in msg


def test_scan_cache_survives_restaging(tmp_path, monkeypatch):
    """Unpacked and copied files keep their source times, so a new runtime that stages again
    reuses the scan saved on Drive instead of scanning everything again."""
    from eviground.data.download import stage

    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    first = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o1", cache_dir=tmp_path / "c", progress=None
    )
    shutil.rmtree(paths.local_data)  # a new runtime: the local disk is empty
    for key in cp1.STAGED_KEYS:
        stage(key, paths=paths, reserve_gb=0)

    def boom(row):
        raise AssertionError("scanned again after re-staging")

    monkeypatch.setattr(cp1, "scan_row", boom)
    second = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o2", cache_dir=tmp_path / "c", progress=None
    )
    assert first["exact_duplicates"] == second["exact_duplicates"]
    assert first["metadata_classifier"] == second["metadata_classifier"]


def test_process_pool_gives_same_result(tmp_path):
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    one = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o1", cache_dir=tmp_path / "c1", progress=None
    )
    two = cp1.run_cp1(
        paths, _cfg(workers=2), out_dir=tmp_path / "o2", cache_dir=tmp_path / "c2", progress=None
    )
    for key in (
        "exact_duplicates",
        "near_duplicates",
        "harmonization_plan",
        "metadata_classifier",
        "blockiness",
    ):
        assert one[key] == two[key], key


def test_corrupt_image_is_excluded_not_fatal(tmp_path):
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    victim = sorted((paths.local_data / "sidset/train/real/images").iterdir())[0]
    victim.write_bytes(b"\xff\xd8\xff\xe0 broken")
    _accept_local_edits(paths)
    res = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", progress=None
    )
    assert res["counts"]["by_role"]["sidset/excluded_unreadable"] == 1
    assert sum(res["counts"]["scan_errors"].values()) == 1
    assert res["stop_and_ask"]["images_not_scanned"] is True
    assert res["verdict"] in ("STOP_AND_ASK", "FAIL")


def test_zip_member_listing_of_fake_matches_probe_layout(tmp_path):
    paths, _ = _world(tmp_path)
    with zipfile.ZipFile(paths.datasets / "02_DiffSeg30k/train.zip") as zf:
        names = zf.namelist()
    assert {n.split("/")[0] for n in names} == {"train_copy"}
    assert sum(n.endswith(".image.png") for n in names) == sum(
        n.endswith(".mask.png") for n in names
    )
    assert sum(n.endswith(".jsonl") for n in names) == 1


def test_mask_value_outside_card_map_stops(tmp_path):
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    mask = sorted((paths.local_data / "diffseg30k/train_copy").glob("*.mask.png"))[0]
    with Image.open(mask) as m:
        arr = np.asarray(m).copy()
    arr[:4, :4] = 9
    Image.fromarray(arr).save(mask)
    _accept_local_edits(paths)
    res = cp1.run_cp1(
        paths, _cfg(), out_dir=tmp_path / "o", cache_dir=tmp_path / "c", progress=None
    )
    assert res["diffseg30k_masks"]["values_outside_0_8"] == [9]
    assert res["stop_and_ask"]["diffseg_mask_values_outside_0_8"] is True
    assert res["counts"]["by_role"]["diffseg30k/excluded_unknown_value"] == 1
    assert res["verdict"] == "STOP_AND_ASK"


def test_pooled_blockiness_uses_sampler_mix_not_measured_counts():
    """Sources differ in level, classes within a source do not: the pooled comparison must not
    flag a class difference just because one source was measured more often in one class."""
    rng = np.random.default_rng(0)

    def g(level, n):
        return list(rng.normal(level, 0.01, n))

    groups = {  # SID-Set level 1.2, every other source 1.0; SID-Set real measured 10x more
        "genimage/real": g(1.0, 20),
        "sidset/real": g(1.2, 200),
        "coco/real": g(1.0, 20),
        "diffseg30k/real": g(1.0, 20),
        "genimage/generated": g(1.0, 20),
        "sidset/generated": g(1.2, 20),
        "diffseg30k/edited": g(1.0, 20),
        "sidset/edited": g(1.2, 20),
    }
    samples = [
        {"source": k.split("/")[0], "label": k.split("/")[1], "blockiness": v}
        for k, vs in groups.items()
        for v in vs
    ]
    res = cp1.blockiness_check(samples, seed=42)
    # real and generated both draw SID-Set one third: not separated, whatever was measured
    assert res["separation"]["all sources (sampler mix): real vs generated"] < 0.6
    # since 14 Nov 2026 the edited class has no SID-Set part, so SID-Set's level now separates
    # real and generated from edited in the sampler mix: the check reports this effect
    assert res["separation"]["all sources (sampler mix): real vs edited"] > 0.6
    assert res["stop_and_ask"] is True
    groups["sidset/edited"] = g(1.3, 20)  # a real class difference inside SID-Set
    samples = [
        {"source": k.split("/")[0], "label": k.split("/")[1], "blockiness": v}
        for k, vs in groups.items()
        for v in vs
    ]
    res = cp1.blockiness_check(samples, seed=42)
    assert res["separation"]["sidset: edited vs real"] > 0.95
    assert res["stop_and_ask"] is True


def test_stuck_drive_read_is_recorded_and_run_finishes(tmp_path, monkeypatch):
    """A read that never returns (stuck Drive mount) must not stop CP1 (13 Nov 2026)."""
    import time as _time

    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    real = cp1.scan_row
    victim = "/GenImage/glide/val/ai/"

    def stuck(row):
        if victim in row["path"] and row["path"].endswith("_0.png"):
            _time.sleep(3600)
        return real(row)

    monkeypatch.setattr(cp1, "scan_row", stuck)
    t = _time.monotonic()
    res = cp1.run_cp1(
        paths,
        _cfg(workers=2, stall_s=1.0, scan_retries=2),
        out_dir=tmp_path / "o",
        cache_dir=tmp_path / "c",
        progress=None,
    )
    assert _time.monotonic() - t < 120
    assert res["counts"]["scan_errors"] == {"genimage_val/read timeout": 1}
    assert res["counts"]["by_role"]["genimage_val/excluded_unreadable"] == 1
    bad = [t for t in res["class_counts"]["table"] if not t["equal"]]
    assert bad == []  # the file exists and was indexed; the timeout is reported as an error
    assert res["stop_and_ask"]["images_not_scanned"] is True


def test_check_staged_catches_wrong_size_extra_file_and_missing_list(tmp_path):
    """Codex review of PR #11: compare the exact staged files, not only their number."""
    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    assert cp1.check_staged(paths)["sidset"]["files"] == 45 + 15  # images + tampered masks
    sid = paths.local_data / "sidset"
    victim = sorted((sid / "train/real/images").iterdir())[0]
    victim.write_bytes(victim.read_bytes() + b"x")
    (sid / "train/real/images/stale_extra.jpg").write_bytes(b"stale")
    (paths.local_data / "coco_val2017/_STAGED_FILES.csv.gz").unlink()
    with pytest.raises(RuntimeError) as err:
        cp1.check_staged(paths)
    msg = str(err.value)
    assert f"sidset: 1 with another size (e.g. ['train/real/images/{victim.name}'])" in msg
    assert "sidset: 1 extra (e.g. ['train/real/images/stale_extra.jpg'])" in msg
    assert "coco_val2017: no list of staged files" in msg


def test_stage_records_the_list_for_copies_from_older_versions(tmp_path):
    from eviground.data.download import stage
    from eviground.data.storage import read_file_list

    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    dest = paths.local_data / "diffseg30k"
    before = read_file_list(dest)
    assert len(before) == 62
    (dest / "_STAGED_FILES.csv.gz").unlink()  # a copy staged before the list existed
    res = stage("diffseg30k", paths=paths, reserve_gb=0)
    assert res["skipped"] is True and read_file_list(dest) == before


def test_blockiness_diagnostics_raw_and_mid_block(run):
    _, _, res = run
    block = res["blockiness"]
    assert "raw_separation" in block and "grid_mid" in block
    assert set(block["grid_mid"]["raw"]) == set(block["raw"])
    for v in block["grid_mid"]["harmonized"].values():
        assert v["n"] > 0 and v["median"] > 0


def test_stage_repairs_a_damaged_unpacked_copy(tmp_path):
    """Codex review of PR #12: re-running Cell 3 must repair, not skip, a damaged copy."""
    from eviground.data.download import stage

    paths, sources = _world(tmp_path)
    _ensure(paths, sources)
    dest = paths.local_data / "diffseg30k"
    files = sorted((dest / "train_copy").glob("*.image.png"))
    good = files[1].read_bytes()
    files[0].unlink()
    files[1].write_bytes(good[:100])  # cut short
    with pytest.raises(RuntimeError):
        cp1.check_staged(paths)
    res = stage("diffseg30k", paths=paths, reserve_gb=0)
    assert res["skipped"] is False
    assert files[0].is_file() and files[1].read_bytes() == good
    cp1.check_staged(paths)  # passes again
    assert stage("diffseg30k", paths=paths, reserve_gb=0)["skipped"] is True


def test_accepted_blockiness_pair_is_reported_not_a_stop():
    rng = np.random.default_rng(5)
    groups = {
        "genimage/real": list(rng.normal(1.0, 0.01, 40)),
        "genimage/generated": list(rng.normal(1.1, 0.01, 40)),
    }
    samples = [
        {"source": k.split("/")[0], "label": k.split("/")[1], "blockiness": v}
        for k, vs in groups.items()
        for v in vs
    ]
    plain = cp1.blockiness_check(samples, seed=42)
    assert plain["stop_and_ask"] is True
    acc = cp1.blockiness_check(samples, seed=42, accepted={"genimage: generated vs real": "why"})
    assert acc["stop_and_ask"] is False and acc["clear_difference"] == {}
    assert acc["accepted"]["genimage: generated vs real"]["why"] == "why"
    assert acc["separation"]["genimage: generated vs real"] > 0.95  # still reported


def test_plan_has_q75_third_for_diffseg_edited(run):
    _, _, res = run
    assert res["harmonization_plan"]["diffseg30k/edited"].get("compress:q75_third", 0) > 0
    assert "compress:q75_third" not in res["harmonization_plan"]["diffseg30k/real"]
    assert "genimage: generated vs real" in res["config"]["blockiness_accepted"]
