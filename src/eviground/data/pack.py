"""Training shards: harmonized images and edit masks (Milestone 3, decision of 15 Nov 2026).

What is packed: every training and validation image of the CP1 manifest that harmonization keeps
(``h_action`` keep or compress), as its **harmonized bytes** (:func:`cp1.harmonized_bytes`: the
CP1 plan, COCO negatives prepared to their DiffSeg30k size; deterministic per uid), and for every
edited image its **edit mask** after the CP1 reading rule (an AI-generated base's value read as
0), stored as a 1-bit PNG. Real and fully generated images have no mask file (all zeros).

Layout under ``Paper 1/cache/shards/``: ``<source>/shard-00000.tar`` (~1 GB each), one index
side-car per shard, and ``index.parquet`` (one row per image: uid, labels, roles, the byte
offsets of the image and mask inside the shard, ``mask_base`` and the mask-rule version) plus
``_DONE.json``. The loader reads an image by seeking to its offset; no tar parsing at load time.

Before writing, the shard size is estimated from a seeded sample per source; packing stops with
:class:`PackSpaceError` (stop and ask Dhanush) if Drive has less free space than the estimate plus
a 10 GB margin. Packing resumes: a shard whose side-car exists and whose tar has the recorded size
is skipped. The original dataset files are never changed.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tarfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.data import cp1, crops, diffseg_mask

log = logging.getLogger(__name__)

MASK_RULE = (
    "diffseg30k: AI-generated base read as 0 (border + most frequent value), v1, 14 Nov 2026"
)
SHARD_BYTES = 1_000_000_000
MARGIN_BYTES = 10_000_000_000
INDEX_FILE = "index.parquet"
SOURCE_NAME = re.compile(r"[a-z0-9_]+")
SPLITS = ("train", "val")
INDEX_COLUMNS = (
    "uid",
    "source",
    "label",
    "generator",
    "editors",
    "split",
    "role",
    "dataset_split",
    "shard",
    "img_offset",
    "img_size",
    "img_format",
    "mask_offset",
    "mask_size",
    "width",
    "height",
    "h_action",
    "h_quality",
    "h_subsampling",
    "mask_base",
    "base_type",
    "valid_crop",
    "mask_rule",
)


class PackSpaceError(RuntimeError):
    """Not enough free space on Drive for the shards (stop and ask)."""


def _int(v: Any) -> int | None:
    return None if v in (None, "") else int(float(v))


def typed(r: dict[str, Any]) -> dict[str, Any]:
    """A manifest row (strings from the csv) with the fields packing needs typed."""
    out = dict(r)
    for k in ("width", "height", "bytes", "h_quality", "prep_w", "prep_h", "mask_base"):
        out[k] = _int(r.get(k))
    vc = r.get("valid_crop")
    out["valid_crop"] = None if vc in (None, "") else str(vc) in ("True", "true", "1")
    return out


def packable(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Training and validation rows harmonization keeps, sorted by (source, uid)."""
    keep = [
        typed(r)
        for r in rows
        if r.get("split") in SPLITS and r.get("h_action") in ("keep", "compress")
    ]
    return sorted(keep, key=lambda r: (r["source"], r["uid"]))


def target_size(r: dict[str, Any]) -> tuple[int, int]:
    if r["source"] == "coco" and r.get("prep_w"):
        return int(r["prep_w"]), int(r["prep_h"])
    return int(r["width"]), int(r["height"])


def edit_mask_png(r: dict[str, Any]) -> bytes | None:
    """1-bit PNG of the edited pixels (CP1 reading rule), for edited DiffSeg30k rows."""
    if r["label"] != "edited" or r["source"] != "diffseg30k" or not r.get("mask_path"):
        return None
    with Image.open(r["mask_path"]) as m:
        arr = np.asarray(m.convert("L"))
    edited = diffseg_mask.edit_mask(arr, r.get("mask_base")) > 0
    buf = io.BytesIO()
    Image.fromarray(edited).convert("1").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _payload(r: dict[str, Any]) -> dict[str, Any]:
    """Harmonized image bytes and edit mask of one row (worker process); ``valid_crop`` is
    filled in for an edited image whose manifest row has none."""
    img = cp1.harmonized_bytes(r)
    if img is None:
        raise ValueError(f"{r['uid']}: harmonization skips this image")
    mask = edit_mask_png(r)
    valid = r.get("valid_crop")
    if mask is not None and valid is None:
        with Image.open(io.BytesIO(mask)) as m:
            valid = crops.has_valid_window(np.asarray(m.convert("L")) > 0)
    return {"img": img, "mask": mask, "valid_crop": valid}


def image_format(data: bytes) -> str:
    if data[:2] == b"\xff\xd8":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "bin"


def _map(fn: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    if workers > 1 and len(items) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items, chunksize=8))
    return [fn(i) for i in items]


# --------------------------------------------------------------------------------------
# Size estimate and space check
# --------------------------------------------------------------------------------------


def estimate_bytes(
    rows: Sequence[dict[str, Any]], *, per_source: int = 200, seed: int = 42, workers: int = 1
) -> dict[str, Any]:
    """Estimated shard bytes per source: kept files at their size; compressed and COCO files
    from the bytes per pixel of a seeded sample of each source (masks included)."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    out: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for source, rs in sorted(by_source.items()):
        kept, made = [], []
        for r in rs:  # kept files are packed as they are; the rest is re-encoded
            (kept if r["h_action"] == "keep" and r["source"] != "coco" else made).append(r)
        est = float(sum(int(r.get("bytes") or 0) for r in kept))
        sample = (
            [made[i] for i in rng.choice(len(made), min(per_source, len(made)), replace=False)]
            if made
            else []
        )
        bpp = 0.0
        mask_mean = 0.0
        if sample:
            got = _map(_payload, sample, workers)
            pixels = sum(target_size(r)[0] * target_size(r)[1] for r in sample)
            bpp = sum(len(g["img"]) for g in got) / max(pixels, 1)
            mask_mean = sum(len(g["mask"] or b"") for g in got) / len(sample)
            est += bpp * sum(target_size(r)[0] * target_size(r)[1] for r in made)
            est += mask_mean * sum(1 for r in rs if r["label"] == "edited")
        out[source] = {
            "images": len(rs),
            "kept_files": len(kept),
            "encoded": len(made),
            "sampled": len(sample),
            "bytes_per_pixel": round(bpp, 4),
            "estimate_bytes": int(est * 1.02),  # tar headers and padding
        }
    out["total_bytes"] = sum(v["estimate_bytes"] for k, v in out.items() if isinstance(v, dict))
    return out


def check_space(
    estimate: dict[str, Any], free_bytes: int | None, margin: int = MARGIN_BYTES
) -> None:
    """Stop and ask if Drive cannot hold the shards plus the margin."""
    need = int(estimate["total_bytes"]) + margin
    if free_bytes is None:
        log.warning("Drive free space unknown: packing %.1f GB without the check", need / 1e9)
        return
    if free_bytes < need:
        raise PackSpaceError(
            f"Drive has {free_bytes / 1e9:.1f} GB free; the shards need about "
            f"{estimate['total_bytes'] / 1e9:.1f} GB + {margin / 1e9:.0f} GB margin: stop and ask"
        )


# --------------------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------------------


def inside(base: Path, rel: str) -> Path:
    """``base / rel``, refusing any name that leaves ``base`` (the shard names come from the
    manifest's source names and from the index)."""
    root = os.path.realpath(base)
    full = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, full]) != root or full == root:
        raise ValueError(f"shard path {rel!r} leaves {base}")
    return Path(full)


def plan_shards(
    rows: Sequence[dict[str, Any]], estimate: dict[str, Any], shard_bytes: int = SHARD_BYTES
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    """(source, shard number, rows): contiguous runs of a fixed count per source, sized so a
    shard holds about ``shard_bytes`` (deterministic, so a restart writes the same shards)."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    plan = []
    for source, rs in sorted(by_source.items()):
        if not SOURCE_NAME.fullmatch(source):
            raise ValueError(f"unexpected source name {source!r} in the manifest")
        est = estimate.get(source, {}).get("estimate_bytes", 0)
        per = max(1, int(len(rs) * shard_bytes / est)) if est else len(rs)
        for k in range(0, len(rs), per):
            plan.append((source, k // per, rs[k : k + per]))
    return plan


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> tuple[int, int]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0  # deterministic shards
    # offset_data is only filled in when reading; the data follows this member's header
    start = tar.offset + len(info.tobuf(tar.format, tar.encoding, tar.errors))
    tar.addfile(info, io.BytesIO(data))
    return start, info.size


def write_shard(
    rows: Sequence[dict[str, Any]], path: Path, shard_name: str, workers: int = 1
) -> list[dict[str, Any]]:
    """Write one shard atomically; returns its index entries."""
    payloads = _map(_payload, list(rows), workers)
    tmp = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for i, (r, p) in enumerate(zip(rows, payloads, strict=True)):
            fmt = image_format(p["img"])
            img_off, img_size = _add(tar, f"{i:06d}.{fmt}", p["img"])
            mask_off = mask_size = None
            if p["mask"] is not None:
                mask_off, mask_size = _add(tar, f"{i:06d}.mask.png", p["mask"])
            w, h = target_size(r)
            entries.append(
                {
                    "uid": r["uid"],
                    "source": r["source"],
                    "label": r["label"],
                    "generator": r.get("generator") or None,
                    "editors": r.get("editors") or None,
                    "split": r["split"],
                    "role": r.get("role"),
                    "dataset_split": r.get("dataset_split"),
                    "shard": shard_name,
                    "img_offset": img_off,
                    "img_size": img_size,
                    "img_format": fmt,
                    "mask_offset": mask_off,
                    "mask_size": mask_size,
                    "width": w,
                    "height": h,
                    "h_action": r["h_action"],
                    "h_quality": r.get("h_quality"),
                    "h_subsampling": r.get("h_subsampling"),
                    "mask_base": r.get("mask_base"),
                    "base_type": r.get("base_type"),
                    "valid_crop": p["valid_crop"],
                    "mask_rule": MASK_RULE if r["label"] == "edited" else None,
                }
            )
    os.replace(tmp, path)
    return entries


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".index.json")


def _shard_done(path: Path) -> list[dict[str, Any]] | None:
    side = _sidecar(path)
    if not (path.is_file() and side.is_file()):
        return None
    meta = json.loads(side.read_text())
    if meta.get("tar_bytes") != path.stat().st_size:
        return None
    return meta["entries"]


def write_index(entries: Sequence[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {c: [e.get(c) for e in entries] for c in INDEX_COLUMNS}
    tmp = path.with_name(path.name + ".part")
    pq.write_table(pa.table(cols), tmp)
    os.replace(tmp, path)


def read_index(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack(
    manifest: Path,
    out_dir: Path,
    *,
    free_bytes: int | None,
    workers: int = 1,
    shard_bytes: int = SHARD_BYTES,
    margin: int = MARGIN_BYTES,
    sample_per_source: int = 200,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Pack the CP1 manifest's training and validation images (see the module docstring)."""
    say = progress or log.info
    t0 = time.time()
    done = read_done(out_dir)
    if done is not None and done.get("manifest_sha256") == file_sha256(manifest):
        say(f"shards already packed ({done['images']} images): nothing to do")
        return done
    rows = packable(cp1._read_csv_gz(manifest))
    estimate = estimate_bytes(rows, per_source=sample_per_source, workers=workers)
    say(
        f"{len(rows)} images to pack; estimated shard size {estimate['total_bytes'] / 1e9:.1f} GB; "
        f"Drive free {'unknown' if free_bytes is None else f'{free_bytes / 1e9:.1f} GB'}"
    )
    check_space(estimate, free_bytes, margin)
    plan = plan_shards(rows, estimate, shard_bytes)
    entries: list[dict[str, Any]] = []
    written = skipped = 0
    for n, (source, k, rs) in enumerate(plan, 1):
        name = f"{source}/shard-{k:05d}.tar"
        path = inside(out_dir, name)
        done = _shard_done(path)
        if done is not None and len(done) == len(rs):
            entries += done
            skipped += 1
            continue
        got = write_shard(rs, path, name, workers)
        _sidecar(path).write_text(json.dumps({"tar_bytes": path.stat().st_size, "entries": got}))
        entries += got
        written += 1
        say(f"shard {n}/{len(plan)} {name}: {len(rs)} images, {path.stat().st_size / 1e9:.2f} GB")
    write_index(entries, out_dir / INDEX_FILE)
    total = sum(p.stat().st_size for p in out_dir.rglob("*.tar"))
    summary = {
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "mask_rule": MASK_RULE,
        "images": len(entries),
        "by_source_split_label": dict(
            sorted(Counter(f"{e['source']}/{e['split']}/{e['label']}" for e in entries).items())
        ),
        "edit_masks": sum(1 for e in entries if e["mask_size"]),
        "shards": len(plan),
        "shards_written": written,
        "shards_skipped": skipped,
        "bytes": total,
        "estimate": estimate,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "_DONE.json").write_text(json.dumps(summary, indent=1))
    return summary


def read_done(out_dir: Path) -> dict[str, Any] | None:
    """The ``_DONE.json`` summary of a finished pack, or None."""
    path = out_dir / "_DONE.json"
    if not (path.is_file() and (out_dir / INDEX_FILE).is_file()):
        return None
    return json.loads(path.read_text())


def read_image_bytes(shard_dir: Path, entry: dict[str, Any], which: str = "img") -> bytes | None:
    """The stored bytes of an index entry (``which``: img or mask), read by offset."""
    off, size = entry[f"{which}_offset"], entry[f"{which}_size"]
    if off is None or size is None:
        return None
    with open(inside(shard_dir, entry["shard"]), "rb") as fh:
        fh.seek(int(off))
        return fh.read(int(size))


def verify_sample(
    shard_dir: Path, rows: Sequence[dict[str, Any]], *, per_source: int = 20, seed: int = 42
) -> dict[str, Any]:
    """Packed bytes == a fresh harmonization (and edit mask) for a seeded sample per source."""
    index = {e["uid"]: e for e in read_index(shard_dir / INDEX_FILE)}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in packable(rows):
        by_source[r["source"]].append(r)
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for source, rs in sorted(by_source.items()):
        picks = [rs[i] for i in rng.choice(len(rs), min(per_source, len(rs)), replace=False)]
        bad = []
        for r in picks:
            e = index.get(r["uid"])
            fresh = _payload(r)
            if (
                e is None
                or read_image_bytes(shard_dir, e) != fresh["img"]
                or read_image_bytes(shard_dir, e, "mask") != fresh["mask"]
            ):
                bad.append(r["uid"])
        result[source] = {"checked": len(picks), "mismatch": len(bad), "examples": bad[:5]}
    result["all_equal"] = all(v["mismatch"] == 0 for v in result.values() if isinstance(v, dict))
    return result
