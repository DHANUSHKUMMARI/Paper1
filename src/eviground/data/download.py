"""``ensure_data()``: fetch, verify and stage the datasets of a profile.

Storage rules (Research Specification, section 13, revision 45):

- Source archives stay on Drive; they are unpacked on Colab's local disk each session
  (:func:`stage`). No loose-file copies and no duplicate shards on Drive.
- Free space on Drive *and* on the local disk is checked before every download and before
  every shard; a step that would not fit stops with a clear message and writes nothing.
- Every step is resumable, verifies file counts and sizes afterwards, and writes ``_DONE``
  (JSON) only after verification. A dataset with a valid ``_DONE`` is skipped with no
  network call and no secret read.

Per dataset (see :mod:`eviground.data.registry`):

- ``genimage_val``: already on Drive; verified in place, never downloaded.
- ``genimage_sd14``: the 89.8 GB Kaggle archive is downloaded to the local disk without
  extraction, its members are listed, a seeded subset of 40,000 real + 40,000 generated
  training images is written to Drive as a few tar files plus a manifest, then the local
  archive is deleted.
- ``diffseg30k``: ``train.zip`` and ``validation.zip`` copied to Drive as they are, checked
  against the size and SHA-256 published by Hugging Face.
- ``sidset``: already on Drive as loose files (10,000 per class from train, 2,000 per class
  from validation, written by the retired downloader); verified in place and copied to the
  local disk by :func:`stage`. The parquet download into tar files
  (``hf_parquet_subset``) is kept as a tested fallback.
- ``brgen``, ``chameleon``: not downloaded (out of CP1).
"""

from __future__ import annotations

import csv
import datetime
import io
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eviground.data.imageinfo import sniff_format
from eviground.data.registry import PROFILES, REGISTRY, DatasetSpec
from eviground.data.sources import NetworkSources, Sources
from eviground.data.storage import (
    DEFAULT_RESERVE_GB,
    ArchiveReader,
    VerificationError,
    check_space,
    copy_files,
    extract_archive,
    read_json,
    sha256_file,
    verify_tar,
    write_json_atomic,
    write_tar_shard,
    write_text_atomic,
)
from eviground.env import Paths, get_paths, git_commit

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TAR_OVERHEAD = 1536  # bytes per member (header + padding), for space estimates
EXT = {"png": ".png", "jpeg": ".jpg", "webp": ".webp", "tiff": ".tif", "bmp": ".bmp", "gif": ".gif"}


class DataStepError(RuntimeError):
    """One or more datasets could not be ensured; ``report`` says which and why."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class Context:
    paths: Paths
    sources_factory: Callable[[], Sources]
    reserve_gb: float = DEFAULT_RESERVE_GB
    _sources: Sources | None = field(default=None, init=False)

    @property
    def sources(self) -> Sources:
        if self._sources is None:  # created only when a download is really needed
            self._sources = self.sources_factory()
        return self._sources

    def drive_space(self, need: float, what: str) -> dict[str, float]:
        return check_space(
            self.paths.datasets, need, what=what, where="Drive", reserve_gb=self.reserve_gb
        )

    def local_space(self, need: float, what: str) -> dict[str, float]:
        return check_space(
            self.paths.local_data,
            need,
            what=what,
            where="the local disk",
            reserve_gb=self.reserve_gb,
        )


# --------------------------------------------------------------------------------------
# _DONE markers
# --------------------------------------------------------------------------------------


def read_done(folder: Path) -> dict[str, Any] | None:
    """The verified ``_DONE`` record, or None (missing, or a legacy marker without JSON)."""
    done = read_json(folder / "_DONE")
    return done if isinstance(done, dict) and done.get("verified") is True else None


def write_done(folder: Path, spec: DatasetSpec, summary: dict[str, Any]) -> dict[str, Any]:
    record = {
        "dataset": spec.key,
        "source": spec.source,
        "verified": True,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "params": spec.params,
        "summary": summary,
    }
    legacy = folder / "_DONE"
    if legacy.is_file() and read_json(legacy) is None:  # keep what the old downloader wrote
        record["legacy_marker"] = legacy.read_text(errors="replace")[:400]
    write_json_atomic(legacy, record)
    return record


def _csv_text(header: list[str], rows: list[list[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def _read_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _is_image(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES


# --------------------------------------------------------------------------------------
# GenImage Validation: verify what is already on Drive
# --------------------------------------------------------------------------------------


def step_verify_folder(ctx: Context, spec: DatasetSpec, folder: Path) -> dict[str, Any]:
    """Count images per generator and class; require all generators and the stated total."""
    from eviground.data.probe import GENIMAGE_GENERATORS

    if not folder.is_dir():
        raise VerificationError(f"{folder} does not exist")
    classes = spec.params["class_dirs"]
    counts: Counter[tuple[str, str]] = Counter()
    unassigned, total_bytes, n_images = 0, 0, 0
    for path in folder.rglob("*"):
        if not (path.is_file() and _is_image(path.name)):
            continue
        n_images += 1
        total_bytes += path.stat().st_size
        rel = path.relative_to(folder).as_posix().lower()
        gens = [g for g, pat in GENIMAGE_GENERATORS.items() if re.search(pat, rel)]
        cls = [c for c in classes if f"/{c}/" in f"/{rel}"]
        if len(gens) == 1 and len(cls) == 1:
            counts[(gens[0], cls[0])] += 1
        else:
            unassigned += 1
    per_gen = {g: {c: counts[(g, c)] for c in classes} for g in GENIMAGE_GENERATORS}
    missing = [f"{g}/{c}" for g, by in per_gen.items() for c, n in by.items() if n == 0]
    summary = {
        "images": n_images,
        "bytes": total_bytes,
        "per_generator": per_gen,
        "unassigned_images": unassigned,
    }
    problems = []
    if missing:
        problems.append(f"no images for {', '.join(missing)}")
    if n_images != spec.params["expected_images"]:
        problems.append(f"{n_images} images, expected {spec.params['expected_images']}")
    if unassigned:
        problems.append(f"{unassigned} images not under exactly one generator and class folder")
    if problems:
        raise VerificationError("GenImage Validation: " + "; ".join(problems))
    return summary


# --------------------------------------------------------------------------------------
# GenImage SD v1.4: seeded subset from the Kaggle archive, written to Drive as tar shards
# --------------------------------------------------------------------------------------


def select_sd14_subset(members: list[Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Seeded choice of ``per_class`` training images per class, in archive order.

    A member qualifies when its path has the split folder (``train``) followed by a class
    folder (``nature`` = real, ``ai`` = generated) and an image suffix.
    """
    split, classes, per_class = params["split_dir"], params["class_dirs"], params["per_class"]
    by_class: dict[str, list[tuple[int, Any]]] = {c: [] for c in classes}
    for idx, m in enumerate(members):
        parts = PurePosixPath(m.name).parts
        if not _is_image(m.name) or split not in parts[:-1]:
            continue
        after = parts[parts.index(split) + 1 : -1]
        if after and after[0] in classes:
            by_class[after[0]].append((idx, m))
    short = {c: len(v) for c, v in by_class.items() if len(v) < per_class}
    if short:
        examples = [m.name for m in members[:10]]
        raise VerificationError(
            f"the archive has too few {split} images per class: {short} (need {per_class} "
            f"each). First members: {examples}"
        )
    rng = np.random.default_rng(params["seed"])
    chosen: list[tuple[int, Any, str]] = []
    for cls in sorted(classes):
        pool = sorted(by_class[cls], key=lambda t: t[1].name)
        pick = rng.choice(len(pool), size=per_class, replace=False)
        chosen += [(pool[i][0], pool[i][1], cls) for i in pick]
    chosen.sort(key=lambda t: t[0])  # archive order: sequential reads
    shard_n = params["shard_images"]
    return [
        {
            "shard": k // shard_n,
            "name": m.name,
            "class_dir": cls,
            "label": classes[cls],
            "bytes": m.size,
        }
        for k, (_, m, cls) in enumerate(chosen)
    ]


def _shard_name(k: int) -> str:
    return f"sd14_subset-{k:05d}.tar"


def _plan_shards(plan: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    shards: dict[int, dict[str, int]] = {}
    for row in plan:
        shards.setdefault(int(row["shard"]), {})[row["name"]] = int(row["bytes"])
    return shards


def step_kaggle_subset(ctx: Context, spec: DatasetSpec, folder: Path) -> dict[str, Any]:
    p = spec.params
    slug = spec.source.split(":", 1)[1]
    archive = ctx.paths.local_data / "downloads" / f"{spec.key}.zip"
    plan_path, manifest_path = folder / "subset_plan.csv", folder / "manifest.csv"
    plan = _read_csv(plan_path)
    shards = _plan_shards(plan) if plan else {}
    pending = [k for k, exp in shards.items() if verify_tar(folder / _shard_name(k), exp)]

    if plan is None or pending:
        if not archive.is_file():
            remaining_subset = (
                sum(sum(shards[k].values()) for k in pending)
                if plan
                else p["subset_gb_estimate"] * 1e9
            )
            ctx.drive_space(remaining_subset, "the GenImage SD v1.4 subset (estimate)")
            ctx.sources.kaggle_download(
                slug,
                archive,
                lambda rem, total: ctx.local_space(rem, "the GenImage SD v1.4 archive"),
            )
        with ArchiveReader(archive) as reader:
            members = reader.members()
            log.info("SD v1.4 archive: %d members", len(members))
            if plan is None:
                plan = select_sd14_subset(members, p)
                write_text_atomic(
                    plan_path,
                    _csv_text(
                        ["shard", "name", "class_dir", "label", "bytes"],
                        [
                            [r["shard"], r["name"], r["class_dir"], r["label"], r["bytes"]]
                            for r in plan
                        ],
                    ),
                )
                shards = _plan_shards(plan)
                pending = sorted(shards)
            sizes = {m.name: m.size for m in members}
            stale = [n for exp in shards.values() for n, b in exp.items() if sizes.get(n) != b]
            if stale:
                raise VerificationError(f"subset plan does not match the archive: {stale[:3]}")
            ctx.drive_space(
                sum(sum(shards[k].values()) + TAR_OVERHEAD * len(shards[k]) for k in pending),
                "the GenImage SD v1.4 subset",
            )
            for k in pending:
                exp = shards[k]
                need = sum(exp.values()) + TAR_OVERHEAD * len(exp)
                ctx.drive_space(need, f"shard {_shard_name(k)}")
                ctx.local_space(need, f"shard {_shard_name(k)} (Drive write cache)")
                records = write_tar_shard(folder / _shard_name(k), reader.iter_read(exp), exp)
                write_json_atomic(folder / f"{_shard_name(k)}.json", records)
                log.info("wrote %s (%d images)", _shard_name(k), len(exp))

    # every shard verified: manifest, counts, then the local archive goes
    rows, label_counts = [], Counter()
    for k, exp in sorted(shards.items()):
        problems = verify_tar(folder / _shard_name(k), exp)
        if problems:
            raise VerificationError(f"{_shard_name(k)}: {problems[:3]}")
        sidecar = read_json(folder / f"{_shard_name(k)}.json") or []
        sha = {r["name"]: r["sha256"] for r in sidecar}
        for r in (r for r in plan if int(r["shard"]) == k):
            rows.append(
                [
                    _shard_name(k),
                    r["name"],
                    r["class_dir"],
                    r["label"],
                    r["bytes"],
                    sha.get(r["name"], ""),
                ]
            )
            label_counts[int(r["label"])] += 1
    expected = {int(v): p["per_class"] for v in p["class_dirs"].values()}
    if dict(label_counts) != expected:
        raise VerificationError(f"subset counts {dict(label_counts)} != {expected}")
    write_text_atomic(
        manifest_path, _csv_text(["shard", "name", "class_dir", "label", "bytes", "sha256"], rows)
    )
    for leftover in (archive, archive.with_name(archive.name + ".part")):
        leftover.unlink(missing_ok=True)
    return {
        "images": len(rows),
        "per_label": {str(k): v for k, v in sorted(label_counts.items())},
        "shards": {_shard_name(k): len(v) for k, v in sorted(shards.items())},
        "bytes": sum(int(r[4]) for r in rows),
        "local_archive_deleted": not archive.exists(),
    }


# --------------------------------------------------------------------------------------
# DiffSeg30k: the two zips, as published
# --------------------------------------------------------------------------------------


def _zip_summary(path: Path) -> dict[str, Any]:
    with ArchiveReader(path) as reader:
        members = reader.members()
    dirs = Counter(str(PurePosixPath(m.name).parent) for m in members)
    exts = Counter(PurePosixPath(m.name).suffix.lower() or "<none>" for m in members)
    return {
        "members": len(members),
        "uncompressed_bytes": sum(m.size for m in members),
        "per_folder": dict(sorted(dirs.items())),
        "extensions": dict(exts),
    }


def step_hf_files(ctx: Context, spec: DatasetSpec, folder: Path) -> dict[str, Any]:
    repo = spec.source.split(":", 1)[1]
    files = spec.params["files"]
    info = ctx.sources.hf_file_info(repo, files)
    missing = [f for f in files if f not in info]
    if missing:
        raise VerificationError(f"{repo} has no {missing}")
    summary: dict[str, Any] = {}
    for name in files:
        size, sha = info[name]["size"], info[name]["sha256"]
        target, stamp = folder / name, folder / f"{name}.verified.json"
        known = read_json(stamp)
        ok = (
            target.is_file()
            and target.stat().st_size == size
            and isinstance(known, dict)
            and known.get("sha256") == sha
            and known.get("bytes") == size
        )
        if not ok:
            if not (target.is_file() and target.stat().st_size == size):
                ctx.drive_space(size, f"{repo} {name}")
                ctx.local_space(0, f"{repo} {name} (Drive write cache)")
                got = ctx.sources.hf_download(repo, name, folder)
                if Path(got) != target:
                    raise VerificationError(f"downloaded to {got}, expected {target}")
            actual_size = target.stat().st_size
            actual_sha = sha256_file(target)
            if actual_size != size or (sha and actual_sha != sha):
                target.unlink(missing_ok=True)
                raise VerificationError(
                    f"{name}: {actual_size} bytes / sha256 {actual_sha[:12]}..., expected "
                    f"{size} / {str(sha)[:12]}...; the file was removed, run again"
                )
            write_json_atomic(stamp, {"bytes": size, "sha256": actual_sha})
        summary[name] = {"bytes": size, "sha256": sha, **_zip_summary(target)}
    return summary


# --------------------------------------------------------------------------------------
# SID-Set: per-class caps from whole parquet shards, stored as tar files
# --------------------------------------------------------------------------------------


def _split_files(files: dict[str, int], split: str) -> list[str]:
    pat = re.compile(rf"(^|/){re.escape(split)}([-_/.]|$)")
    return sorted(f for f in files if f.endswith(".parquet") and pat.search(f))


def plan_sid_split(
    labels_per_shard: dict[str, list[int]], cap: int, labels: list[int], seed: int
) -> dict[str, Any]:
    """Seeded choice of up to ``cap`` rows per label, taking whole shards in a seeded order.

    Only the last shard used for a label is subsampled, so few shards are downloaded.
    """
    rng = np.random.default_rng(seed)
    names = sorted(labels_per_shard)
    order = [names[i] for i in rng.permutation(len(names))]
    rows: dict[str, list[int]] = {}
    counts = {}
    for lab in labels:
        need = cap
        for name in order:
            if need == 0:
                break
            idx = [i for i, v in enumerate(labels_per_shard[name]) if v == lab]
            if not idx:
                continue
            if len(idx) > need:
                idx = sorted(int(i) for i in rng.choice(idx, size=need, replace=False))
            rows.setdefault(name, []).extend(idx)
            need -= len(idx)
        counts[str(lab)] = cap - need
    return {
        "rows": {k: sorted(v) for k, v in sorted(rows.items())},
        "counts": counts,
        "cap": cap,
        "shortfall": {k: cap - v for k, v in counts.items() if v < cap},
    }


def _cell_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("bytes")
    return bytes(value) if value else None


def _sid_items(
    parquet: Path, rows: list[int], split: str, stem: str, p: dict[str, Any]
) -> tuple[list[tuple[str, bytes]], list[dict[str, Any]]]:
    """Tar members and manifest rows for the chosen rows of one local parquet shard."""
    import pyarrow.parquet as pq

    cols = [p["label_column"], p["image_column"], p["mask_column"], p["id_column"]]
    table = pq.read_table(parquet, columns=[c for c in cols if c in pq.read_schema(parquet).names])
    wanted = set(rows)
    items, meta = [], []
    data = table.to_pydict()
    for i in sorted(wanted):
        lab = int(data[p["label_column"]][i])
        name = p["labels"][lab]
        img = _cell_bytes(data[p["image_column"]][i])
        if img is None:
            raise VerificationError(f"{stem} row {i}: no image bytes")
        key = f"{stem}_{i:06d}"
        img_name = f"{split}/{name}/images/{key}{EXT.get(sniff_format(img[:16]), '.bin')}"
        items.append((img_name, img))
        mask_name = ""
        mask = _cell_bytes(data.get(p["mask_column"], [None] * (i + 1))[i])
        if mask is not None:
            mask_name = f"{split}/{name}/masks/{key}_mask{EXT.get(sniff_format(mask[:16]), '.bin')}"
            items.append((mask_name, mask))
        ids = data.get(p["id_column"])
        meta.append(
            {
                "image": img_name,
                "mask": mask_name,
                "label": lab,
                "label_name": name,
                "img_id": "" if ids is None else str(ids[i]),
                "source_file": stem,
                "row": i,
            }
        )
    return items, meta


def step_hf_parquet_subset(ctx: Context, spec: DatasetSpec, folder: Path) -> dict[str, Any]:
    p = spec.params
    repo = spec.source.split(":", 1)[1]
    files: dict[str, int] | None = None
    summary: dict[str, Any] = {}
    for s_idx, (split, cap) in enumerate(p["caps"].items()):
        plan_path = folder / f"{split}_plan.json"
        plan = read_json(plan_path)
        if plan is None:
            files = files or ctx.sources.hf_list_files(repo)
            shards = _split_files(files, split)
            if not shards:
                raise VerificationError(f"{repo}: no parquet files for split {split!r}")
            labels = {
                f: [int(v) for v in ctx.sources.hf_parquet_column(repo, f, p["label_column"])]
                for f in shards
            }
            plan = plan_sid_split(labels, cap, sorted(p["labels"]), seed=p["seed"] + s_idx)
            plan["file_bytes"] = {f: files[f] for f in plan["rows"]}
            plan["file_rows"] = {f: len(labels[f]) for f in plan["rows"]}
            plan["label_totals"] = dict(Counter(str(v) for vs in labels.values() for v in vs))
            write_json_atomic(plan_path, plan)
        meta_all: list[dict[str, Any]] = []
        for src, rows in plan["rows"].items():
            stem = PurePosixPath(src).stem
            tar_path = folder / f"sid_{split}-{stem}.tar"
            side_path = folder / f"sid_{split}-{stem}.tar.json"
            side = read_json(side_path)
            if side and tar_path.is_file() and not verify_tar(tar_path, side["sizes"]):
                meta_all += side["meta"]
                continue
            fbytes = plan["file_bytes"][src]
            ctx.local_space(fbytes, f"SID-Set {src}")
            share = len(rows) / max(1, plan["file_rows"][src])  # estimate before reading
            ctx.drive_space(fbytes * share * 1.05, f"SID-Set rows from {src}")
            tmp_dir = ctx.paths.local_data / "downloads" / "sidset"
            local = ctx.sources.hf_download(repo, src, tmp_dir)
            items, meta = _sid_items(Path(local), rows, split, stem, p)
            sizes = {n: len(b) for n, b in items}
            ctx.drive_space(sum(sizes.values()) + TAR_OVERHEAD * len(sizes), f"SID-Set {stem}")
            records = write_tar_shard(tar_path, items, sizes)
            sha = {r["name"]: r["sha256"] for r in records}
            for m in meta:
                m["sha256"] = sha[m["image"]]
                m["tar"] = tar_path.name
            write_json_atomic(side_path, {"sizes": sizes, "meta": meta})
            Path(local).unlink(missing_ok=True)
            meta_all += meta
        counts = Counter(str(m["label"]) for m in meta_all)
        if dict(counts) != {k: v for k, v in plan["counts"].items() if v}:
            raise VerificationError(f"SID-Set {split}: {dict(counts)} != plan {plan['counts']}")
        masks = Counter(str(m["label"]) for m in meta_all if m["mask"])
        if counts.get("2") and not masks.get("2"):
            raise VerificationError(
                f"SID-Set {split}: no masks for label 2 (tampered); check the label map"
            )
        header = [
            "tar",
            "image",
            "mask",
            "label",
            "label_name",
            "img_id",
            "source_file",
            "row",
            "sha256",
        ]
        write_text_atomic(
            folder / f"{split}_manifest.csv",
            _csv_text(header, [[m[h] for h in header] for m in meta_all]),
        )
        summary[split] = {
            "per_label": dict(sorted(counts.items())),
            "masks_per_label": dict(sorted(masks.items())),
            "shortfall": plan["shortfall"],
            "label_totals": plan["label_totals"],
            "source_files": len(plan["rows"]),
        }
    return summary


def _sid_listed_files(folder: Path, splits: list[str]) -> dict[str, int] | None:
    """Relative path -> size of every image and mask in the SID-Set manifests."""
    files: dict[str, int] = {}
    for split in splits:
        rows = _read_csv(folder / f"{split}_manifest.csv")
        if rows is None:
            return None
        for r in rows:
            for rel in (r["image"], r.get("mask") or ""):
                if rel:
                    files[rel] = (folder / rel).stat().st_size
    return files


def step_verify_sid_folder(ctx: Context, spec: DatasetSpec, folder: Path) -> dict[str, Any]:
    """Verify the SID-Set files already on Drive (left by the retired downloader), in place.

    Per split: the manifest lists exactly ``cap`` images per class; every listed image and
    mask exists and is not empty; no image is listed twice; each image sits in the folder of
    its class and its ``label_name`` agrees with the label map; every tampered image has a
    mask and no real or fully synthetic image has one. Nothing is downloaded or changed.
    """
    p = spec.params
    names = {int(k): v for k, v in p["labels"].items()}
    tampered = int(p.get("tampered_label", 2))
    summary: dict[str, Any] = {}
    problems: list[str] = []
    for split, cap in p["caps"].items():
        rows = _read_csv(folder / f"{split}_manifest.csv")
        if rows is None:
            problems.append(f"{split}: {split}_manifest.csv is missing")
            continue
        missing_cols = {"image", "mask", "label"} - set(rows[0] if rows else {})
        if missing_cols:
            problems.append(f"{split}: manifest has no column(s) {sorted(missing_cols)}")
            continue
        seen: set[str] = set()
        counts, masks = Counter(), Counter()
        dup, missing, empty, wrong_label, mask_on_other, tampered_no_mask = 0, [], [], [], 0, 0
        nbytes = 0
        for r in rows:
            img, lab = r["image"], int(r["label"])
            if img in seen:
                dup += 1
                continue
            seen.add(img)
            counts[lab] += 1
            name = names.get(lab)
            if name is None or r.get("label_name", name) != name or f"/{name}/" not in f"/{img}":
                wrong_label.append(img)
            for rel in (img, r.get("mask") or ""):
                if not rel:
                    continue
                path = folder / rel
                if not path.is_file():
                    missing.append(rel)
                elif path.stat().st_size == 0:
                    empty.append(rel)
                else:
                    nbytes += path.stat().st_size
            if r.get("mask"):
                masks[lab] += 1
                mask_on_other += lab != tampered
            elif lab == tampered:
                tampered_no_mask += 1
        listed = seen | {r["mask"] for r in rows if r.get("mask")}
        on_disk = {
            q.relative_to(folder).as_posix()
            for q in (folder / split).rglob("*")
            if q.is_file() and _is_image(q.name)
        }
        per_label = {str(k): counts.get(k, 0) for k in sorted(names)}
        summary[split] = {
            "per_label": per_label,
            "masks_per_label": {str(k): v for k, v in sorted(masks.items())},
            "bytes": nbytes,
            "files_on_disk_not_in_manifest": len(on_disk - listed),
        }
        wrong_counts = {k: v for k, v in per_label.items() if v != cap}
        for what, bad in (
            (f"counts per class {wrong_counts} differ from {cap}", wrong_counts),
            (f"{len(missing)} listed files missing, e.g. {missing[:3]}", missing),
            (f"{len(empty)} empty files, e.g. {empty[:3]}", empty),
            (f"{dup} images listed twice", dup),
            (
                f"{len(wrong_label)} images whose folder or label_name disagrees with the label "
                f"map {names}, e.g. {wrong_label[:3]}",
                wrong_label,
            ),
            (
                f"{mask_on_other} real or fully synthetic images have a mask (label map?)",
                mask_on_other,
            ),
            (f"{tampered_no_mask} tampered images have no mask", tampered_no_mask),
        ):
            if bad:
                problems.append(f"{split}: {what}")
    if problems:
        raise VerificationError("SID-Set: " + "; ".join(problems))
    return summary


STEPS: dict[str, Callable[[Context, DatasetSpec, Path], dict[str, Any]]] = {
    "verify_folder": step_verify_folder,
    "verify_sid_folder": step_verify_sid_folder,
    "kaggle_subset": step_kaggle_subset,
    "hf_files": step_hf_files,
    "hf_parquet_subset": step_hf_parquet_subset,
}


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def ensure_data(
    profile: str = "medium",
    datasets: list[str] | None = None,
    *,
    paths: Paths | None = None,
    sources: Sources | Callable[[], Sources] | None = None,
    reserve_gb: float = DEFAULT_RESERVE_GB,
    stage_local: bool = False,
    raise_errors: bool = True,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Make sure each dataset is on Drive and verified; optionally unpack it locally.

    A dataset whose ``_DONE`` is valid is skipped with no network call. Otherwise its step
    runs (downloads only what is missing) and ``_DONE`` is written after verification.
    Errors are collected per dataset; with ``raise_errors`` a :class:`DataStepError`
    carrying the full report is raised at the end. ``overrides`` replaces registry
    parameters per dataset (tests and dry runs use tiny subsets); the key ``method``
    selects another step (the SID-Set parquet download is kept as a tested fallback).
    """
    paths = paths or get_paths()
    if sources is None:
        factory: Callable[[], Sources] = NetworkSources
    elif callable(sources) and not hasattr(sources, "kaggle_download"):
        factory = sources  # type: ignore[assignment]
    else:
        factory = lambda: sources  # noqa: E731
    ctx = Context(paths=paths, sources_factory=factory, reserve_gb=reserve_gb)
    report: dict[str, Any] = {}
    for key in datasets or PROFILES[profile]:
        spec = REGISTRY[key]
        if overrides and key in overrides:
            ov = dict(overrides[key])
            spec = replace(spec, method=ov.pop("method", spec.method), params={**spec.params, **ov})
        folder = paths.datasets / spec.folder
        entry: dict[str, Any] = {"folder": spec.folder}
        try:
            done = read_done(folder)
            if done:
                entry.update(status="done", summary=done["summary"])
            elif spec.method == "deferred":
                entry.update(status="deferred", note=spec.role)
            else:
                log.info("%s: running %s", key, spec.method)
                summary = STEPS[spec.method](ctx, spec, folder)
                entry.update(
                    status="verified", summary=write_done(folder, spec, summary)["summary"]
                )
            if stage_local and entry["status"] in ("done", "verified"):
                entry["staged"] = stage(key, paths=paths, reserve_gb=reserve_gb)
        except Exception as exc:  # report every dataset, then raise once
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            log.error("%s: %s", key, entry["error"])
        report[key] = entry
    failed = {k: v["error"] for k, v in report.items() if v["status"] == "error"}
    if failed and raise_errors:
        raise DataStepError(f"{len(failed)} dataset(s) not ensured: {failed}", report)
    return report


def stage(
    key: str, *, paths: Paths | None = None, reserve_gb: float = DEFAULT_RESERVE_GB
) -> dict[str, Any]:
    """Unpack a verified dataset's archives from Drive to the local disk (each session)."""
    paths = paths or get_paths()
    spec = REGISTRY[key]
    folder = paths.datasets / spec.folder
    if read_done(folder) is None:
        raise VerificationError(f"{key}: no verified _DONE on Drive; run ensure_data first")
    if spec.method == "verify_folder":
        return {"staged": False, "reason": "read in place on Drive"}
    dest = paths.local_data / key
    if spec.method == "verify_sid_folder":  # loose files on Drive: copy the listed ones
        files = _sid_listed_files(folder, list(spec.params["caps"]))
        if files is None:
            raise VerificationError(f"{key}: a manifest is missing")
        res = copy_files(folder, files, dest, reserve_gb=reserve_gb)
        write_json_atomic(dest / "_STAGED.json", {**res, "dest": str(dest)})
        return {"staged": True, "skipped": res["copied"] == 0, **res, "dest": str(dest)}
    archives = sorted(folder.glob("*.zip") if spec.method == "hf_files" else folder.glob("*.tar"))
    dest = paths.local_data / key
    marker = read_json(dest / "_STAGED.json")
    names = [a.name for a in archives]
    if marker and marker.get("archives") == names:
        return {"staged": True, "skipped": True, **marker}
    files = nbytes = 0
    for arc in archives:
        res = extract_archive(arc, dest, reserve_gb=reserve_gb)
        files, nbytes = files + res["files"], nbytes + res["bytes"]
    record = {"archives": names, "files": files, "bytes": nbytes, "dest": str(dest)}
    write_json_atomic(dest / "_STAGED.json", record)
    return {"staged": True, "skipped": False, **record}
