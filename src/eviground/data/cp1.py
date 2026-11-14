"""Checkpoint CP1: data integrity and leakage (Research Specification section 9.1, rev 49).

Pass criteria: 0 duplicate image hashes across train / validation / test; class counts equal
the manifests; a classifier using only file format, JPEG quality and crop size scores <= 60%
real-vs-generated on the harmonized training data (original image size reported separately);
pixel-level 8x8 blockiness reported per class and source, a clear class difference means stop
and ask. Decisions of 12-13 Nov 2026 add: near-duplicates of a test image in train are counted
and more than 1% of a test set means stop and ask; COCO negatives with a near-duplicate in any
test set are dropped.

Pipeline (:func:`run_cp1`): index the staged files -> scan every image once (header facts,
file / pixel / perceptual hashes, DiffSeg30k mask values) -> roles (train pool, test sets,
exclusions) -> COCO near-duplicate filter -> 10% stratified validation split -> exact and
near duplicates -> class counts -> GenImage real quality distribution -> harmonization plan ->
crops drawn by the training sampler -> metadata classifier, blockiness, sample grids.
The scan is cached in parts, so a disconnected session resumes where it stopped.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.data import crops, harmonize, hashing, sampler, shortcut, splits
from eviground.data.download import _assign, _is_image, read_done
from eviground.data.imageinfo import blockiness, header_info
from eviground.data.registry import REGISTRY
from eviground.data.watchdog import run_watched
from eviground.env import Paths

log = logging.getLogger(__name__)

SID_CLASS = {"real": "real", "full_synthetic": "generated", "tampered": "edited"}
GENIMAGE_CLASS = {"nature": "real", "ai": "generated"}
TEST_ROLES = {
    "test_P1": "GenImage Validation (P1)",
    "test_P2": "DiffSeg30k validation, Glide/Kolors (P2)",
    "test_P3b": "SID-Set validation (P3b)",
    "test_P7": "COCO photos reserved for Claims-Edit (P7)",
}
# Split sizes published on the DiffSeg30k card (read 13 Nov 2026)
DIFFSEG_CARD_SPLITS = {"train": 28_998, "validation": 6_001}


@dataclass
class CP1Config:
    seed: int = 42
    split_seed: int = splits.SPLIT_SEED
    workers: int = max(1, (os.cpu_count() or 2))
    crops_per_class: int = 3000  # sampler draws per class for the metadata classifier
    blockiness_per_group: int = 200  # images per (source, class) for blockiness
    grid: int = 4
    near_dup_primary: int = hashing.NEAR_DUP_PRIMARY
    near_dup_test_share: float = 0.01  # more than this share of a test set -> stop and ask
    scan_part_rows: int = 2000
    stall_s: float = 180.0  # a single image read taking longer is treated as stuck
    scan_retries: int = 2
    diffseg_card_splits: dict[str, int] | None = None
    coco: dict[str, Any] | None = None  # COCO params; default: the verified _DONE record


# --------------------------------------------------------------------------------------
# Index of the staged files
# --------------------------------------------------------------------------------------


def _row(**kw: Any) -> dict[str, Any]:
    base = {
        "uid": "",
        "source": "",
        "dataset_split": "",
        "label": "",
        "path": "",
        "mask_path": "",
        "generator": "",
        "role": "",
    }
    base.update(kw)
    return base


def index_genimage_sd14(local: Path) -> list[dict[str, Any]]:
    rows = []
    root = local / "genimage_sd14"
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if not (path.is_file() and _is_image(path.name)) or len(parts) < 3:
            continue
        if parts[-3] != "train" or parts[-2] not in GENIMAGE_CLASS:
            continue
        rows.append(
            _row(
                uid=f"genimage_sd14/{rel}",
                source="genimage",
                dataset_split="train",
                label=GENIMAGE_CLASS[parts[-2]],
                path=str(path),
                generator="sd14",
                role="train_pool",
            )
        )
    return rows


def index_genimage_val(folder: Path) -> list[dict[str, Any]]:
    rows = []
    classes = REGISTRY["genimage_val"].params["class_dirs"]
    for path in sorted(folder.rglob("*")):
        if not (path.is_file() and _is_image(path.name)):
            continue
        rel = path.relative_to(folder).as_posix()
        gen, cls = _assign(rel, classes)
        if not (gen and cls):
            continue
        rows.append(
            _row(
                uid=f"genimage_val/{rel}",
                source="genimage_val",
                dataset_split="validation",
                label=GENIMAGE_CLASS[cls],
                path=str(path),
                generator=gen,
                role="test_P1",
            )
        )
    return rows


def index_diffseg(local: Path) -> list[dict[str, Any]]:
    rows = []
    root = local / "diffseg30k"
    for path in sorted(root.rglob("*.image.png")):
        rel = path.relative_to(root).as_posix()
        folder = rel.split("/")[0]
        split = "validation" if folder.startswith("validation") else "train"
        mask = path.with_name(path.name[: -len(".image.png")] + ".mask.png")
        rows.append(
            _row(
                uid=f"diffseg30k/{rel}",
                source="diffseg30k",
                dataset_split=split,
                label="edited",
                path=str(path),
                mask_path=str(mask) if mask.is_file() else "",
            )
        )
    return rows


def index_sidset(local: Path, drive_folder: Path) -> list[dict[str, Any]]:
    """Manifests from Drive; the files from the local copy made by ``stage('sidset')``."""
    rows = []
    root = local / "sidset"
    for split, role in (("train", "train_pool"), ("validation", "test_P3b")):
        manifest = drive_folder / f"{split}_manifest.csv"
        with open(manifest, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(
                    _row(
                        uid=f"sidset/{r['image']}",
                        source="sidset",
                        dataset_split=split,
                        label=SID_CLASS[r["label_name"]],
                        path=str(root / r["image"]),
                        mask_path=str(root / r["mask"]) if r.get("mask") else "",
                        role=role,
                    )
                )
    return rows


def reserve_claims_edit(instances: dict[str, Any], params: dict[str, Any]) -> list[int]:
    """COCO val2017 image ids reserved for the Claims-Edit case study (excluded from training).

    An image qualifies for a category if it has a non-crowd instance of it covering at least
    ``min_area_share`` of the image. Categories take turns (in the order given); each turn draws
    the next unused qualifying image from a seeded permutation of that category's candidates,
    until ``n`` images are reserved.
    """
    names = {c["id"]: c["name"] for c in instances["categories"]}
    sizes = {im["id"]: im["width"] * im["height"] for im in instances["images"]}
    cands: dict[str, set[int]] = {c: set() for c in params["categories"]}
    for a in instances["annotations"]:
        name = names.get(a["category_id"])
        if name in cands and not a.get("iscrowd"):
            if a["area"] >= params["min_area_share"] * sizes[a["image_id"]]:
                cands[name].add(a["image_id"])
    rng = np.random.default_rng(params["seed"])
    queues = {c: [int(i) for i in rng.permutation(sorted(ids))] for c, ids in cands.items()}
    chosen: list[int] = []
    taken: set[int] = set()
    while len(chosen) < params["n"] and any(queues.values()):
        for c in params["categories"]:
            q = queues[c]
            while q and q[0] in taken:
                q.pop(0)
            if q and len(chosen) < params["n"]:
                taken.add(q[0])
                chosen.append(q.pop(0))
    if len(chosen) < params["n"]:
        raise ValueError(f"only {len(chosen)} COCO images qualify for Claims-Edit")
    return sorted(chosen)


def coco_instances(datasets: Path) -> dict[str, Any]:
    folder = datasets / REGISTRY["coco_val2017"].folder
    with zipfile.ZipFile(folder / "annotations_trainval2017.zip") as zf:
        return json.loads(zf.read("annotations/instances_val2017.json"))


def index_coco(local: Path, reserved: set[int]) -> list[dict[str, Any]]:
    rows = []
    root = local / "coco_val2017"
    for path in sorted((root / "val2017").glob("*.jpg")):
        image_id = int(path.stem)
        rows.append(
            _row(
                uid=f"coco/val2017/{path.name}",
                source="coco",
                dataset_split="val2017",
                label="real",
                path=str(path),
                role="test_P7" if image_id in reserved else "train_pool",
            )
        )
    return rows


STAGED_KEYS = ("genimage_sd14", "diffseg30k", "sidset", "coco_val2017")


def check_staged(paths: Paths) -> dict[str, Any]:
    """Every dataset CP1 reads from the local disk must be there in full.

    The local copies live on the Colab machine's disk and are gone in a new runtime; without
    this check CP1 would index only what is left and fail much later with a confusing error.
    Requires ``local_data/<key>/_STAGED.json`` (written by ``stage()`` after verification) and at
    least as many files on disk as it records.
    """
    problems, found = [], {}
    for key in STAGED_KEYS:
        dest = paths.local_data / key
        marker = dest / "_STAGED.json"
        if not marker.is_file():
            problems.append(f"{key}: not staged")
            continue
        want = int(json.loads(marker.read_text()).get("files", 0))
        have = sum(
            1
            for p in dest.rglob("*")
            if p.is_file() and p.name != "_STAGED.json" and not p.name.endswith(".part")
        )
        found[key] = {"expected": want, "on_disk": have}
        if have < want:
            problems.append(f"{key}: {have} of {want} files")
    if problems:
        raise RuntimeError(
            "local copies are missing or incomplete (" + "; ".join(problems) + "). This happens "
            "in a new Colab runtime: run Cell 3 (ensure_data with stage_local=True) first."
        )
    return found


# --------------------------------------------------------------------------------------
# Scan: one pass over every image
# --------------------------------------------------------------------------------------


def _mask_facts(path: str) -> dict[str, Any]:
    if not path:
        return {"mask_error": "missing"}
    try:
        with Image.open(path) as m:
            fmt, mode, size = (m.format or "").lower(), m.mode, m.size
            arr = np.asarray(m.convert("L"))
    except Exception as exc:
        return {"mask_error": f"decode: {type(exc).__name__}"}
    values, counts = np.unique(arr, return_counts=True)
    return {
        "mask_format": fmt,
        "mask_mode": mode,
        "mask_w": size[0],
        "mask_h": size[1],
        "mask_values": ",".join(str(int(v)) for v in values),
        "mask_value_pixels": ",".join(str(int(c)) for c in counts),
    }


def fingerprint(row: dict[str, Any]) -> str:
    """Size and modification time of the image (and its mask): a cached scan is reused only
    if these are unchanged, so a file replaced in place is scanned again."""
    parts = []
    for key in ("path", "mask_path"):
        if row.get(key):
            try:
                st = os.stat(row[key])
                parts.append(f"{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                parts.append("missing")
    return "|".join(parts)


def scan_row(row: dict[str, Any]) -> dict[str, Any]:
    """Header facts, hashes and mask facts of one image (runs in a worker process)."""
    out: dict[str, Any] = {"uid": row["uid"], "fingerprint": fingerprint(row)}
    try:
        data = Path(row["path"]).read_bytes()
    except OSError as exc:
        out["error"] = f"read: {type(exc).__name__}"
        return out
    info = header_info(data)
    for k in ("format", "width", "height", "mode", "jpeg_quality", "jpeg_quality_exact"):
        out[k] = info.get(k)
    out["subsampling"] = info.get("subsampling")
    out["bytes"] = len(data)
    h = hashing.image_hashes(data)
    out.update(sha256=h["sha256"], pixel_sha256=h.get("pixel_sha256"), phash=h.get("phash"))
    if "error" in h or "error" in info:
        out["error"] = h.get("error") or info.get("error")
    if row["source"] == "coco" and "error" not in out:
        with Image.open(io.BytesIO(data)) as img:
            prepared = harmonize.prepare_like_diffseg(img, row.get("_prep_size", 1024))
        out["phash_prepared"] = hashing.phash(prepared)
    if row["source"] == "diffseg30k" or (row["source"] == "sidset" and row["mask_path"]):
        out.update(_mask_facts(row["mask_path"]))
    return out


SCAN_COLUMNS = [
    "uid",
    "fingerprint",
    "format",
    "width",
    "height",
    "mode",
    "jpeg_quality",
    "jpeg_quality_exact",
    "subsampling",
    "bytes",
    "sha256",
    "pixel_sha256",
    "phash",
    "phash_prepared",
    "mask_format",
    "mask_mode",
    "mask_w",
    "mask_h",
    "mask_values",
    "mask_value_pixels",
    "mask_error",
    "error",
]


def _write_csv_gz(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _read_csv_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


_INT = {"width", "height", "jpeg_quality", "bytes", "phash", "phash_prepared", "mask_w", "mask_h"}


def _typed(r: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in r.items():
        if v in ("", None):
            out[k] = None
        elif k in _INT:
            out[k] = int(v)
        elif k == "jpeg_quality_exact":
            out[k] = v in (True, "True")
        else:
            out[k] = v
    return out


def scan(
    rows: list[dict[str, Any]],
    cache: Path,
    workers: int,
    part_rows: int = 2000,
    progress: Callable[[str], None] | None = None,
    *,
    stall_s: float = 180.0,
    retries: int = 2,
    progress_every: int = 500,
) -> dict[str, dict[str, Any]]:
    """uid -> scan facts. Parts of ``part_rows`` images are cached as csv.gz in ``cache``;
    a part is read back instead of scanned again only if its images and their fingerprints
    (size and modification time of each image and mask) are unchanged.

    With ``workers > 1`` the images are read by :func:`run_watched`: a read that has not
    returned after ``stall_s`` seconds (a stuck Drive mount) is retried in a fresh process,
    and after ``retries`` stalls the image is recorded with ``error="read timeout"``."""
    cache.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    results: dict[str, dict[str, Any]] = {}
    for source in sorted(by_source):
        src_rows = sorted(by_source[source], key=lambda r: r["uid"])
        for k in range(0, len(src_rows), part_rows):
            part = src_rows[k : k + part_rows]
            path = cache / f"scan_{source}_{k // part_rows:05d}.csv.gz"
            keys = [(r["uid"], fingerprint(r)) for r in part]
            if path.is_file():
                got = _read_csv_gz(path)
                if [(g["uid"], g.get("fingerprint")) for g in got] == keys:
                    results.update({g["uid"]: _typed(g) for g in got})
                    continue
            t0 = time.time()

            def tick(
                done: int,
                total: int,
                _k: int = k,
                _n: int = len(src_rows),
                _src: str = source,
                _t0: float = t0,
            ) -> None:
                if progress and done % progress_every == 0 and done < total:
                    progress(
                        f"scan {_src}: {_k + done}/{_n} "
                        f"({done / max(time.time() - _t0, 1e-6):.0f} images/s)"
                    )

            if workers > 1:
                got = run_watched(
                    scan_row,
                    part,
                    workers,
                    stall_s=stall_s,
                    retries=retries,
                    on_timeout=_timeout_row,
                    progress=tick,
                    describe=lambda r: r["uid"],
                )
                got = [
                    g if isinstance(g, dict) else {"uid": r["uid"], "error": f"worker: {g!r}"[:200]}
                    for g, r in zip(got, part, strict=True)
                ]
            else:
                got = [scan_row(r) for r in part]
            _write_csv_gz(path, got, SCAN_COLUMNS)
            results.update({g["uid"]: _typed(g) for g in got})
            if progress:
                timeouts = sum(1 for g in got if g.get("error") == "read timeout")
                progress(
                    f"scan {source}: {k + len(part)}/{len(src_rows)} "
                    f"({len(part) / max(time.time() - t0, 1e-6):.0f} images/s"
                    + (f", {timeouts} read timeouts" if timeouts else "")
                    + ")"
                )
    return results


def _timeout_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"uid": row["uid"], "fingerprint": fingerprint(row), "error": "read timeout"}


# --------------------------------------------------------------------------------------
# Roles, splits, duplicates
# --------------------------------------------------------------------------------------


def assign_diffseg_roles(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        if r["source"] != "diffseg30k":
            continue
        if r.get("error") or r.get("mask_error"):
            r["role"] = "excluded_unreadable"
            continue
        values = splits.parse_editors(r.get("mask_values"))
        r["editors"] = splits.editors_text(values)
        r["role"] = splits.diffseg_role(r["dataset_split"], values - {0})


def near_dup_filter_coco(rows: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    """Drop COCO negatives with a near-duplicate in any test set (raw or prepared pHash)."""
    test = [r for r in rows if r["role"].startswith("test_") and r.get("phash") is not None]
    ref = hashing.as_uint64([r["phash"] for r in test])
    coco = [r for r in rows if r["source"] == "coco" and r["role"] == "train_pool"]
    if not coco:
        return {"checked": 0, "dropped": 0}
    d_raw, i_raw = hashing.min_hamming(hashing.as_uint64([r["phash"] for r in coco]), ref)
    d_prep, i_prep = hashing.min_hamming(
        hashing.as_uint64([r.get("phash_prepared") or r["phash"] for r in coco]), ref
    )
    dropped = Counter()
    examples = []
    for r, dr, ir, dp, ip in zip(coco, d_raw, i_raw, d_prep, i_prep, strict=True):
        d, i = (dr, ir) if dr <= dp else (dp, ip)
        if d <= threshold:
            r["role"] = "excluded_near_dup_test"
            dropped[test[i]["role"]] += 1
            if len(examples) < 20:
                examples.append({"coco": r["uid"], "test": test[i]["uid"], "hamming": int(d)})
    return {
        "checked": len(coco),
        "dropped": int(sum(dropped.values())),
        "dropped_by_test_set": dict(dropped),
        "threshold": threshold,
        "examples": examples,
    }


def assign_splits(rows: list[dict[str, Any]], seed: int) -> None:
    """split = train / val (10% stratified per source) / test / excluded."""
    pool: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("error") and not r["role"].startswith("excluded"):
            r["role"] = "excluded_unreadable"
        if r["role"] == "train_pool":
            pool.setdefault(r["source"], []).append(r)
        elif r["role"].startswith("test_"):
            r["split"] = "test"
        else:
            r["split"] = "excluded"
    for source, rs in pool.items():
        if source == "diffseg30k":
            strata = [r.get("editors", "") for r in rs]
        else:
            strata = [r["label"] for r in rs]
        out = splits.stratified_val([r["uid"] for r in rs], strata, seed=seed)
        for r, s in zip(rs, out, strict=True):
            r["split"] = s


def exact_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Hashes shared across train / val / test (the criterion) and within a split (reported)."""
    report: dict[str, Any] = {}
    used = [r for r in rows if r.get("split") in ("train", "val", "test")]
    for key in ("sha256", "pixel_sha256"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in used:
            if r.get(key):
                groups.setdefault(r[key], []).append(r)
        cross, within, examples = 0, 0, []
        cross_pairs: Counter[str] = Counter()
        for members in groups.values():
            if len(members) < 2:
                continue
            split_set = {m["split"] for m in members}
            if len(split_set) > 1:
                cross += 1
                cross_pairs["+".join(sorted(split_set))] += 1
                if len(examples) < 20:
                    examples.append([m["uid"] for m in members[:4]])
            else:
                within += 1
        report[key] = {
            "cross_split_groups": cross,
            "cross_split_by_splits": dict(cross_pairs),
            "within_split_groups": within,
            "examples": examples,
        }
    report["cross_split_total"] = sum(
        report[k]["cross_split_groups"] for k in ("sha256", "pixel_sha256")
    )
    return report


def near_duplicates(rows: list[dict[str, Any]], cfg: CP1Config) -> dict[str, Any]:
    """For every test set: how many images have a near-duplicate in train + val."""
    train = [r for r in rows if r.get("split") in ("train", "val") and r.get("phash") is not None]
    ref = hashing.as_uint64([r["phash"] for r in train])
    out: dict[str, Any] = {
        "reference_images": len(train),
        "primary_threshold": cfg.near_dup_primary,
    }
    for role, name in TEST_ROLES.items():
        test = [r for r in rows if r["role"] == role and r.get("phash") is not None]
        if not test:
            out[role] = {"name": name, "images": 0}
            continue
        dist, idx = hashing.min_hamming(hashing.as_uint64([r["phash"] for r in test]), ref)
        n_primary = int((dist <= cfg.near_dup_primary).sum())
        share = n_primary / len(test)
        order = np.argsort(dist)[:10]
        out[role] = {
            "name": name,
            "images": len(test),
            "counts": hashing.near_dup_counts(dist),
            "share_primary": round(share, 5),
            "stop_and_ask": share > cfg.near_dup_test_share,
            "closest": [
                {
                    "test": test[i]["uid"],
                    "train": train[idx[i]]["uid"] if idx[i] >= 0 else None,
                    "hamming": int(dist[i]),
                }
                for i in order
            ],
            "by_train_source": dict(
                Counter(
                    train[idx[i]]["source"] for i in np.flatnonzero(dist <= cfg.near_dup_primary)
                )
            ),
        }
    return out


# --------------------------------------------------------------------------------------
# Class counts against the source manifests
# --------------------------------------------------------------------------------------


def class_counts(rows: list[dict[str, Any]], paths: Paths, cfg: CP1Config) -> dict[str, Any]:
    found = Counter((r["source"], r["dataset_split"], r["label"]) for r in rows)
    expected: dict[tuple[str, str, str], int] = {}
    ds = paths.datasets
    # GenImage SD v1.4 subset: manifest.csv on Drive
    with open(ds / REGISTRY["genimage_sd14"].folder / "manifest.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            key = ("genimage", "train", GENIMAGE_CLASS[r["class_dir"]])
            expected[key] = expected.get(key, 0) + 1
    # GenImage Validation: verified counts per generator and class (_DONE)
    done = read_done(ds / REGISTRY["genimage_val"].folder) or {}
    per_gen = (done.get("summary") or {}).get("per_generator") or {}
    for by in per_gen.values():
        for cls, n in by.items():
            key = ("genimage_val", "validation", GENIMAGE_CLASS[cls])
            expected[key] = expected.get(key, 0) + int(n)
    # DiffSeg30k: split sizes on the dataset card
    for split, n in (cfg.diffseg_card_splits or DIFFSEG_CARD_SPLITS).items():
        expected[("diffseg30k", split, "edited")] = n
    # SID-Set: the manifests on Drive
    sid = ds / REGISTRY["sidset"].folder
    for split in ("train", "validation"):
        with open(sid / f"{split}_manifest.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                key = ("sidset", split, SID_CLASS[r["label_name"]])
                expected[key] = expected.get(key, 0) + 1
    # COCO: images listed in the verified zip
    coco_done = read_done(ds / REGISTRY["coco_val2017"].folder) or {}
    n_coco = coco_done.get("summary", {}).get("val2017.zip", {}).get("matching_members")
    if n_coco is not None:
        expected[("coco", "val2017", "real")] = int(n_coco)
    table = []
    for key in sorted(set(expected) | set(found)):
        table.append(
            {
                "source": key[0],
                "dataset_split": key[1],
                "label": key[2],
                "expected": expected.get(key),
                "found": found.get(key, 0),
                "equal": expected.get(key) == found.get(key, 0),
            }
        )
    return {"table": table, "all_equal": all(t["equal"] for t in table)}


# --------------------------------------------------------------------------------------
# Harmonization plan, crops, metadata classifier, blockiness
# --------------------------------------------------------------------------------------


def _plan_info(r: dict[str, Any], prep_size: int) -> dict[str, Any]:
    if r["source"] == "coco":  # prepared like DiffSeg30k: a lossless 1024x1024 image
        return {"format": "png", "width": prep_size, "height": prep_size}
    return {
        k: r.get(k) for k in ("format", "width", "height", "jpeg_quality", "subsampling", "error")
    }


def plan_rows(
    rows: list[dict[str, Any]], dist: harmonize.QualityDistribution, seed: int, prep_size: int
) -> dict[str, Any]:
    summary: dict[str, Counter] = {}
    for r in rows:
        if r.get("split") not in ("train", "val"):
            continue
        p = harmonize.plan_image(
            r["source"], r["label"], _plan_info(r, prep_size), r["uid"], dist, seed=seed
        )
        r["h_action"], r["h_quality"], r["h_subsampling"], r["h_reason"] = (
            p["action"],
            p["quality"],
            p["subsampling"],
            p["reason"],
        )
        summary.setdefault(f"{r['source']}/{r['label']}", Counter())[
            f"{p['action']}:{p['reason']}"
        ] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(summary.items())}


def _load_mask(r: dict[str, Any]) -> np.ndarray | None:
    if not r.get("mask_path"):
        return None
    with Image.open(r["mask_path"]) as m:
        arr = np.asarray(m.convert("L"))
    if r["source"] == "sidset":
        return (arr > 127).astype(np.uint8)  # lossy masks: binarized (decision of 13 Nov 2026)
    return arr


def harmonized_bytes(r: dict[str, Any], prep_size: int) -> bytes | None:
    """The training file of one row after preparation and harmonization."""
    data = Path(r["path"]).read_bytes()
    plan = {"action": r["h_action"], "quality": r["h_quality"], "subsampling": r["h_subsampling"]}
    if r["source"] == "coco":
        with Image.open(io.BytesIO(data)) as img:
            prepared = harmonize.prepare_like_diffseg(img, prep_size)
        if plan["action"] != "compress":
            return harmonize.png_bytes(prepared)
        return harmonize.encode_jpeg(prepared, plan["quality"], plan["subsampling"])
    return harmonize.apply_plan(data, plan)


def sample_row(task: tuple[dict[str, Any], int, int, int, bool]) -> dict[str, Any]:
    """Harmonize one drawn image, take its training crop, measure it (worker process)."""
    r, draw_seed, prep_size, crop_size, want_block = task
    out: dict[str, Any] = {"uid": r["uid"], "source": r["source"], "label": r["label"]}
    try:
        hb = harmonized_bytes(r, prep_size)
        if hb is None:
            out["error"] = "skipped"
            return out
        info = header_info(hb)
        out.update(
            h_format=info.get("format"),
            h_quality=info.get("jpeg_quality"),
            h_subsampling=info.get("subsampling"),
            h_bytes=len(hb),
            orig_w=r.get("width"),
            orig_h=r.get("height"),
            raw_format=r.get("format"),
            raw_quality=r.get("jpeg_quality"),
        )
        with Image.open(io.BytesIO(hb)) as img:
            img.load()
            rgb = np.asarray(img.convert("RGB"))
        rng = np.random.default_rng(draw_seed)
        mask = _load_mask(r) if r["label"] == "edited" else None
        if mask is not None:
            box, share, tries = crops.edited_box(mask, rng, crop_size)
            out.update(edited_share=share, crop_tries=tries)
        else:
            box = crops.random_box(rgb.shape[1], rgb.shape[0], rng, crop_size)
        if box is None:
            out["error"] = "crop_failed"
            return out
        left, top, right, bottom = box
        out.update(crop_w=right - left, crop_h=bottom - top, box=[left, top, right, bottom])
        if want_block:
            out["blockiness"] = blockiness(rgb)
            raw = np.asarray(Image.open(r["path"]).convert("RGB"))
            out["blockiness_raw"] = blockiness(raw) if min(raw.shape[:2]) >= 16 else None
    except Exception as exc:  # report, never crash the check
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def _bin_bytes(n: int | None) -> int | None:
    return None if not n else int(round(np.log2(n) * 4))


def metadata_check(samples: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ok = [s for s in samples if "error" not in s]
    y3 = [s["label"] for s in ok]
    yb = ["real" if s["label"] == "real" else "not_real" for s in ok]
    crit = {
        "format": [s["h_format"] for s in ok],
        "jpeg_quality": [s["h_quality"] for s in ok],
        "crop_w": [s["crop_w"] for s in ok],
        "crop_h": [s["crop_h"] for s in ok],
    }
    out: dict[str, Any] = {
        "n_crops": len(ok),
        "metric": "5-fold cross-validated balanced accuracy of a lookup table on the exact "
        "feature values (best possible classifier on these features); chance = 0.5 (binary)",
        "criterion": {
            "real_vs_not_real": shortcut.feature_report(crit, yb, seed=seed),
            "three_class": shortcut.feature_report(crit, y3, seed=seed),
        },
    }
    out["criterion_score"] = out["criterion"]["real_vs_not_real"]["balanced_accuracy"]
    out["criterion_pass"] = out["criterion_score"] <= shortcut.CRITERION
    per_source = {}
    for source in sorted({s["source"] for s in ok}):
        idx = [i for i, s in enumerate(ok) if s["source"] == source]
        labels = {y3[i] for i in idx}
        if len(labels) < 2:
            continue
        feats = {k: [v[i] for i in idx] for k, v in crit.items()}
        per_source[source] = shortcut.feature_report(feats, [y3[i] for i in idx], seed=seed)
    out["per_source_harmonized"] = per_source
    diag = {
        "with_subsampling": {**crit, "subsampling": [s["h_subsampling"] for s in ok]},
        "original_size": {
            "orig_w": [s["orig_w"] for s in ok],
            "orig_h": [s["orig_h"] for s in ok],
        },
        "harmonized_file_bytes": {"bytes_bin": [_bin_bytes(s["h_bytes"]) for s in ok]},
        "raw_files_positive_control": {
            "raw_format": [s["raw_format"] for s in ok],
            "raw_quality": [s["raw_quality"] for s in ok],
            "orig_w": [s["orig_w"] for s in ok],
            "orig_h": [s["orig_h"] for s in ok],
        },
    }
    out["diagnostics"] = {
        k: shortcut.feature_report(v, yb, seed=seed)["balanced_accuracy"] for k, v in diag.items()
    }
    return out


def _sampler_weighted(
    groups: dict[str, list[float]], min_group: int = 10
) -> tuple[list[float], list[str], list[float]] | None:
    """Every measured value once, weighted so that each (class, source) group carries its
    share in the unified sampler (weights within a class sum to 1)."""
    values: list[float] = []
    labels: list[str] = []
    weights: list[float] = []
    for label, shares in sampler.SHARES["unified"].items():
        for source, share in sorted(shares.items()):
            v = groups.get(f"{source}/{label}", [])
            if len(v) < min_group:
                return None
            values += [float(x) for x in v]
            labels += [label] * len(v)
            weights += [share / len(v)] * len(v)
    return values, labels, weights


def blockiness_check(samples: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ok = [s for s in samples if "error" not in s and s.get("blockiness") is not None]
    groups: dict[str, list[float]] = {}
    raw: dict[str, list[float]] = {}
    for s in ok:
        groups.setdefault(f"{s['source']}/{s['label']}", []).append(s["blockiness"])
        if s.get("blockiness_raw") is not None:
            raw.setdefault(f"{s['source']}/{s['label']}", []).append(s["blockiness_raw"])
    out: dict[str, Any] = {
        "metric": "pixel-level 8x8 blockiness of the harmonized whole image (about 1.0 = no "
        "block artefacts); separation = cross-validated balanced accuracy of the best threshold",
        "harmonized": {k: shortcut.describe(v) for k, v in sorted(groups.items())},
        "raw": {k: shortcut.describe(v) for k, v in sorted(raw.items())},
        "separation": {},
    }
    by_source: dict[str, dict[str, list[float]]] = {}
    for key, v in groups.items():
        src, lab = key.split("/")
        by_source.setdefault(src, {})[lab] = v
    for src, labs in sorted(by_source.items()):
        names = sorted(labs)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if len(labs[a]) >= 10 and len(labs[b]) >= 10:
                    vals = labs[a] + labs[b]
                    y = [a] * len(labs[a]) + [b] * len(labs[b])
                    out["separation"][f"{src}: {a} vs {b}"] = round(
                        shortcut.threshold_cv(vals, y, seed=seed), 4
                    )
    # Pooled over sources, each (class, source) group weighted by its sampler share, so the
    # comparison sees the same mix of sources as training (SID-Set one third of every class).
    pooled = _sampler_weighted(groups)
    if pooled is not None:
        values, labels, weights = pooled
        y = ["real" if lab == "real" else "not_real" for lab in labels]
        out["separation"]["all sources (sampler mix): real vs not real"] = round(
            shortcut.threshold_cv(values, y, seed=seed, weights=weights), 4
        )
        pairs = {}
        for pair in (("real", "generated"), ("real", "edited"), ("generated", "edited")):
            keep = [i for i, lab in enumerate(labels) if lab in pair]
            pairs[" vs ".join(pair)] = round(
                shortcut.threshold_cv(
                    [values[i] for i in keep],
                    [labels[i] for i in keep],
                    seed=seed,
                    weights=[weights[i] for i in keep],
                ),
                4,
            )
        for name, score in pairs.items():
            out["separation"][f"all sources (sampler mix): {name}"] = score
    out["clear_difference"] = {k: v for k, v in out["separation"].items() if v > shortcut.CRITERION}
    out["stop_and_ask"] = bool(out["clear_difference"])
    return out


def make_grids(
    samples: list[dict[str, Any]],
    rows_by_uid: dict[str, dict[str, Any]],
    out_dir: Path,
    grid: int,
    prep_size: int,
) -> dict[str, Any]:
    """A grid x grid sheet of harmonized training crops per class (edited: with a mask sheet)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for label in sampler.CLASSES:
        picks = [s for s in samples if s["label"] == label and "error" not in s][: grid * grid]
        if not picks:
            continue
        size = picks[0]["crop_w"]
        sheet = Image.new("RGB", (grid * size, grid * size), (255, 255, 255))
        mask_sheet = Image.new("L", sheet.size, 0) if label == "edited" else None
        tiles = []
        for k, s in enumerate(picks):
            r = rows_by_uid[s["uid"]]
            hb = harmonized_bytes(r, prep_size)
            with Image.open(io.BytesIO(hb)) as img:
                tile = img.convert("RGB").crop(tuple(s["box"]))
            sheet.paste(tile, ((k % grid) * size, (k // grid) * size))
            if mask_sheet is not None:
                m = _load_mask(r)
                if m is not None:
                    mt = Image.fromarray(((m > 0) * 255).astype(np.uint8)).crop(tuple(s["box"]))
                    mask_sheet.paste(mt, ((k % grid) * size, (k // grid) * size))
            tiles.append({"uid": s["uid"], "source": s["source"]})
        path = out_dir / f"cp1_grid_{label}.png"
        sheet.save(path)
        entry = {"file": path.name, "tiles": tiles}
        if mask_sheet is not None:
            mpath = out_dir / f"cp1_grid_{label}_masks.png"
            mask_sheet.save(mpath)
            entry["mask_file"] = mpath.name
        result[label] = entry
    return result


# --------------------------------------------------------------------------------------
# Mask reports
# --------------------------------------------------------------------------------------


def diffseg_mask_report(rows: list[dict[str, Any]], local: Path) -> dict[str, Any]:
    ds = [r for r in rows if r["source"] == "diffseg30k"]
    values: Counter[int] = Counter()
    pixels: Counter[int] = Counter()
    fmt = Counter(r.get("mask_format") for r in ds)
    mode = Counter(r.get("mask_mode") for r in ds)
    mismatch = sum(
        1
        for r in ds
        if r.get("mask_w") and (r["mask_w"], r["mask_h"]) != (r.get("width"), r.get("height"))
    )
    for r in ds:
        if r.get("mask_values"):
            vs = [int(v) for v in r["mask_values"].split(",")]
            cs = [int(c) for c in r["mask_value_pixels"].split(",")]
            for v, c in zip(vs, cs, strict=True):
                values[v] += 1
                pixels[v] += c
    jsonl = []
    for p in sorted((local / "diffseg30k").rglob("*.jsonl")):
        with open(p, errors="replace") as fh:
            jsonl.append(
                {
                    "file": p.relative_to(local).as_posix(),
                    "first_lines": [fh.readline()[:300] for _ in range(3)],
                }
            )
    roles = Counter((r["dataset_split"], r["role"]) for r in ds)
    return {
        "masks": len(ds),
        "missing_masks": sum(1 for r in ds if not r["mask_path"]),
        "mask_errors": sum(1 for r in ds if r.get("mask_error")),
        "mask_format": {str(k): v for k, v in fmt.items()},
        "mask_mode": {str(k): v for k, v in mode.items()},
        "size_mismatch": mismatch,
        "distinct_values": sorted(values),
        "images_per_value": {str(k): v for k, v in sorted(values.items())},
        "pixels_per_value": {str(k): v for k, v in sorted(pixels.items())},
        "values_outside_0_8": sorted(v for v in values if v > 8),
        "editor_map": {str(k): v for k, v in splits.DIFFSEG_EDITORS.items()},
        "roles": {f"{s}/{r}": n for (s, r), n in sorted(roles.items())},
        "jsonl": jsonl,
    }


def sid_mask_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for split in ("train", "validation"):
        ms = [
            r
            for r in rows
            if r["source"] == "sidset" and r["dataset_split"] == split and r["label"] == "edited"
        ]
        lossy = 0
        for r in ms:
            vals = [int(v) for v in (r.get("mask_values") or "").split(",") if v != ""]
            if any(v not in (0, 255) for v in vals):
                lossy += 1
        out[split] = {
            "masks": len(ms),
            "missing": sum(1 for r in ms if not r["mask_path"]),
            "errors": sum(1 for r in ms if r.get("mask_error")),
            "format": dict(Counter(str(r.get("mask_format")) for r in ms)),
            "mode": dict(Counter(str(r.get("mask_mode")) for r in ms)),
            "with_values_other_than_0_255": lossy,
            "binarization": "mask > 127",
        }
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "uid",
    "source",
    "dataset_split",
    "label",
    "generator",
    "editors",
    "role",
    "split",
    "path",
    "mask_path",
    "format",
    "width",
    "height",
    "mode",
    "jpeg_quality",
    "jpeg_quality_exact",
    "subsampling",
    "bytes",
    "sha256",
    "pixel_sha256",
    "phash",
    "phash_prepared",
    "mask_values",
    "h_action",
    "h_quality",
    "h_subsampling",
    "h_reason",
    "error",
]


def _counts(rows: Iterable[dict[str, Any]], *keys: str) -> dict[str, int]:
    c = Counter("/".join(str(r.get(k)) for k in keys) for r in rows)
    return dict(sorted(c.items()))


def run_cp1(
    paths: Paths,
    cfg: CP1Config | None = None,
    *,
    out_dir: Path,
    cache_dir: Path,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Run every CP1 check on the staged data; returns the result record (JSON-ready).

    Writes the full manifest (``manifest_cp1.csv.gz``), the sample grids and the result
    JSON to ``out_dir``. ``cache_dir`` keeps the scan parts between sessions.
    """
    cfg = cfg or CP1Config()
    say = progress or (lambda _m: None)
    t0 = time.time()
    local = paths.local_data
    staged = check_staged(paths)
    say(f"local copies complete: {staged}")
    coco = cfg.coco or (read_done(paths.datasets / REGISTRY["coco_val2017"].folder) or {}).get(
        "params"
    )
    if not coco:
        raise RuntimeError("coco_val2017 is not verified; run ensure_data first")
    prep = int(coco["negatives_size"])
    reserved = reserve_claims_edit(coco_instances(paths.datasets), coco["claims_edit"])
    say(f"Claims-Edit: {len(reserved)} COCO photos reserved")

    rows = (
        index_genimage_sd14(local)
        + index_genimage_val(paths.datasets / REGISTRY["genimage_val"].folder)
        + index_diffseg(local)
        + index_sidset(local, paths.datasets / REGISTRY["sidset"].folder)
        + index_coco(local, set(reserved))
    )
    for r in rows:
        if r["source"] == "coco":
            r["_prep_size"] = prep
    say(f"indexed {len(rows)} images: {_counts(rows, 'source')}")
    facts = scan(
        rows,
        cache_dir,
        cfg.workers,
        cfg.scan_part_rows,
        say,
        stall_s=cfg.stall_s,
        retries=cfg.scan_retries,
    )
    for r in rows:
        r.update({k: v for k, v in facts[r["uid"]].items() if k != "uid"})
    assign_diffseg_roles(rows)
    coco_filter = near_dup_filter_coco(rows, cfg.near_dup_primary)
    assign_splits(rows, cfg.split_seed)
    say("roles and splits assigned")

    dups = exact_duplicates(rows)
    near = near_duplicates(rows, cfg)
    counts = class_counts(rows, paths, cfg)

    real_train = [
        r
        for r in rows
        if r["source"] == "genimage" and r["label"] == "real" and r["split"] == "train"
    ]
    dist = harmonize.QualityDistribution.measure(real_train)
    plan = plan_rows(rows, dist, cfg.seed, prep)
    say("harmonization planned")

    eligible = [
        r for r in rows if r.get("split") == "train" and r.get("h_action") in ("keep", "compress")
    ]
    picks = sampler.draw(
        [r["source"] for r in eligible],
        [r["label"] for r in eligible],
        cfg.crops_per_class,
        cfg.seed,
    )
    block_quota: Counter[str] = Counter()
    measured: set[str] = set()  # blockiness once per image, even if drawn several times
    tasks = []
    for k, i in enumerate(picks):
        r = eligible[int(i)]
        key = f"{r['source']}/{r['label']}"
        want = block_quota[key] < cfg.blockiness_per_group and r["uid"] not in measured
        if want:
            block_quota[key] += 1
            measured.add(r["uid"])
        clean = {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
        tasks.append((clean, cfg.seed * 1_000_003 + k, prep, crops.CROP, want))
    say(f"harmonizing and cropping {len(tasks)} sampled images")
    if cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            samples = list(pool.map(sample_row, tasks, chunksize=16))
    else:
        samples = [sample_row(t) for t in tasks]
    errors = Counter(s["error"] for s in samples if "error" in s)
    meta = metadata_check(samples, cfg.seed)
    block = blockiness_check(samples, cfg.seed)
    grids = make_grids(samples, {r["uid"]: r for r in rows}, out_dir, cfg.grid, prep)
    edited_shares = [
        s["edited_share"] for s in samples if s.get("edited_share") is not None and "error" not in s
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_gz(out_dir / "manifest_cp1.csv.gz", rows, MANIFEST_COLUMNS)
    masks = diffseg_mask_report(rows, local)
    criteria = {
        "no_cross_split_duplicates": dups["cross_split_total"] == 0,
        "class_counts_equal_manifests": counts["all_equal"],
        "metadata_classifier_le_60": meta["criterion_pass"],
    }
    stops = {
        "blockiness_clear_class_difference": block["stop_and_ask"],
        "near_duplicates_over_1pct": any(
            v.get("stop_and_ask") for k, v in near.items() if isinstance(v, dict)
        ),
        # mask values outside the card's map: the editor map cannot be trusted (12 Nov 2026)
        "diffseg_mask_values_outside_0_8": bool(masks["values_outside_0_8"]),
        # an image that could not be read (corrupt, or a read that never returned) was left out
        # of the duplicate checks, so CP1 cannot pass until it is looked at
        "images_not_scanned": any(r.get("error") for r in rows),
    }
    verdict = (
        "PASS"
        if all(criteria.values()) and not any(stops.values())
        else ("FAIL" if not all(criteria.values()) else "STOP_AND_ASK")
    )
    return {
        "verdict": verdict,
        "criteria": criteria,
        "stop_and_ask": stops,
        "config": {k: v for k, v in asdict(cfg).items() if k != "coco"},
        "coco_params": coco,
        "elapsed_s": round(time.time() - t0, 1),
        "counts": {
            "by_source_split_label": _counts(rows, "source", "split", "label"),
            "by_role": _counts(rows, "source", "role"),
            "scan_errors": _counts([r for r in rows if r.get("error")], "source", "error"),
        },
        "class_counts": counts,
        "exact_duplicates": dups,
        "near_duplicates": near,
        "coco_negatives": {
            "reserved_claims_edit_ids": reserved,
            "near_dup_filter": coco_filter,
            "prepared_size": prep,
        },
        "diffseg30k_masks": masks,
        "sidset_masks": sid_mask_report(rows),
        "quality_distribution_genimage_real": {**dist.summary(), "table": dist.to_dict()},
        "harmonization_plan": plan,
        "skipped_too_small": _counts(
            [r for r in rows if r.get("h_reason") == "too_small"], "source", "label"
        ),
        "crops": {
            "drawn": len(samples),
            "errors": dict(errors),
            "per_class_source": _counts(
                [s for s in samples if "error" not in s], "label", "source"
            ),
            "edited_share": shortcut.describe(edited_shares),
        },
        "metadata_classifier": meta,
        "blockiness": block,
        "grids": grids,
    }
