"""Read-only probe of the datasets folder on Drive (first step of checkpoint CP1).

It records what is actually present, so the manifest builders are written against the real
layout and not against assumptions:

- every folder with file counts, sizes and extensions, and Drive free space;
- per folder of images (a "group", i.e. one source and class): real file format from the
  bytes, JPEG quality histogram, chroma subsampling, image sizes, file sizes and pixel-level
  8x8 blockiness on a seeded sample;
- what is present and missing against the downloader's ``medium`` profile;
- DiffSeg30k: mask file format, mode and distinct mask values; SID-Set: label counts;
  Chameleon: image counts in ``test/0_real`` and ``test/1_fake`` (and in ``Chameleon.zip``);
  BR-Gen: archive parts;
- optionally (Colab only) the Hugging Face dataset cards of DiffSeg30k, SID-Set and BR-Gen,
  so the DiffSeg30k mask-value map can be read from the card instead of assumed.

Nothing under ``datasets/`` is written, moved, extracted or re-encoded.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from eviground.data.imageinfo import blockiness, header_info, sniff_format
from eviground.data.storage import ArchiveReader

log = logging.getLogger(__name__)

PROBE_VERSION = 1
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"})
MIN_SIDE = 256  # training crop; smaller images are skipped in training (spec section 8, M1)

DATASET_FOLDERS = (
    "01_GenImage",
    "02_DiffSeg30k",
    "03_SID-Set",
    "04_BR-Gen",
    "05_Chameleon",
    "06_MMTD-Set",
    "07_FaceForensics++_optional",
)

# GenImage generators expected in the Validation subset (Research Specification, P1).
GENIMAGE_GENERATORS: dict[str, str] = {
    "adm": r"adm",
    "biggan": r"biggan",
    "glide": r"glide",
    "midjourney": r"midjourney",
    # GenImage: imagenet_ai_0419_sdv4; Kaggle mirror: stable_diffusion_v_1_4
    "sd14": r"sdv4|sd[-_ .]?v?[-_ .]?1[-_.]?4|stable[-_ ]?diffusion[-_ ]?v?[-_ ]?1[-_.]4",
    "sd15": r"sdv5|sd[-_ .]?v?[-_ .]?1[-_.]?5|stable[-_ ]?diffusion[-_ ]?v?[-_ ]?1[-_.]5",
    "vqdm": r"vqdm",
    "wukong": r"wukong",
}

# Downloader v3, medium profile (notebooks/tools/Paper1_Dataset_Downloader_v3_medium.ipynb).
SID_CAPS = {"train": 10_000, "validation": 2_000}
SID_LABELS = {0: "real", 1: "full_synthetic", 2: "tampered"}

HF_REPOS = {
    "diffseg30k": "Chaos2629/Diffseg30k",
    "sidset": "saberzl/SID_Set",
    "brgen": "clpbc/BR-Gen",
}
EDITOR_WORDS = (
    "sd2",
    "sd 2",
    "stable diffusion 2",
    "sd3",
    "sd 3",
    "sd3.5",
    "sdxl",
    "flux",
    "glide",
    "kolors",
    "hunyuan",
    "kandinsky",
    "mask",
    "label",
    "value",
)

ARCHIVE_RE = re.compile(r"\.(zip|z\d{2}|tar|tgz|gz|7z|rar)(\.\d{3})?$|\.part\d*$|\.\d{3}$", re.I)


# --------------------------------------------------------------------------------------
# Folder scan
# --------------------------------------------------------------------------------------


def scan_tree(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Walk ``root``; return per-folder records and the image file names of each folder.

    Records hold the folder path relative to ``root``, direct file count, direct bytes,
    extension counts and subfolder names. Symlinks are not followed.
    """
    records: list[dict[str, Any]] = []
    images: dict[str, list[str]] = {}
    stack = [root]
    while stack:
        folder = stack.pop()
        rel = folder.relative_to(root).as_posix()
        n_files, n_bytes, exts, subdirs, imgs = 0, 0, Counter(), [], []
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name)
        except OSError as exc:
            records.append({"dir": rel, "error": f"{type(exc).__name__}"})
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs.append(entry.name)
                stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                n_files += 1
                try:
                    n_bytes += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    pass
                suffix = Path(entry.name).suffix.lower()
                exts[suffix or "<none>"] += 1
                if suffix in IMAGE_SUFFIXES:
                    imgs.append(entry.name)
        records.append(
            {
                "dir": rel,
                "files": n_files,
                "bytes": n_bytes,
                "ext_counts": dict(sorted(exts.items())),
                "subdirs": subdirs,
            }
        )
        if imgs:
            images[rel] = imgs
    records.sort(key=lambda r: r["dir"])
    return records, images


def _subtree_totals(records: list[dict[str, Any]], prefix: str) -> dict[str, int]:
    """Files, bytes and image files of the folder ``prefix`` and everything below it."""
    files = size = 0
    for rec in records:
        d = rec["dir"]
        if prefix in ("", ".") or d == prefix or d.startswith(prefix + "/"):
            files += rec.get("files", 0)
            size += rec.get("bytes", 0)
    return {"files": files, "bytes": size}


# --------------------------------------------------------------------------------------
# Per-group statistics (one folder of images = one source and class)
# --------------------------------------------------------------------------------------


def _summary(values: Iterable[float]) -> dict[str, float] | None:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def _open_image(source: Path | bytes) -> Image.Image:
    return Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)


def _blockiness_of(source: Path | bytes) -> float | None:
    try:
        with _open_image(source) as img:
            arr = np.asarray(img.convert("RGB"))
        return blockiness(arr)
    except Exception:
        return None


Reader = Callable[[str], "Path | bytes"]


def group_stats(
    folder: Path | None,
    names: list[str],
    *,
    sample_headers: int,
    sample_blockiness: int,
    seed: int,
    workers: int = 16,
    reader: Reader | None = None,
) -> dict[str, Any]:
    """Format, JPEG quality, subsampling, size and blockiness statistics of one group.

    A group is a folder on Drive (``folder``) or a folder inside archives (``reader``
    returns a member's bytes). Header facts use a seeded sample of up to ``sample_headers``
    files; blockiness decodes the first ``sample_blockiness`` of that sample.
    """
    read: Reader = reader or (lambda n: folder / n)  # type: ignore[operator]
    rng = np.random.default_rng(seed)
    order = sorted(names)
    idx = rng.permutation(len(order))[: min(sample_headers, len(order))]
    sample = [order[i] for i in sorted(idx)]
    n_workers = workers if reader is None else 1  # archive handles are not thread-safe
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        infos = list(pool.map(lambda n: header_info(read(n)), sample))
        blocks = list(pool.map(lambda n: _blockiness_of(read(n)), sample[:sample_blockiness]))

    ok = [i for i in infos if "error" not in i]
    errors = [f"{n}: {i['error']}" for n, i in zip(sample, infos, strict=True) if "error" in i]
    jpegs = [i for i in ok if i["format"] == "jpeg"]
    sizes = Counter(f"{i['width']}x{i['height']}" for i in ok)
    min_sides = [min(i["width"], i["height"]) for i in ok]
    return {
        "n_images": len(names),
        "example_names": order[:3],
        "ext_counts": dict(Counter(Path(n).suffix.lower() for n in names)),
        "sampled": len(sample),
        "format_counts": dict(Counter(i["format"] for i in ok)),
        "mode_counts": dict(Counter(i["mode"] for i in ok)),
        "jpeg_quality_hist": {
            str(k): v for k, v in sorted(Counter(i["jpeg_quality"] for i in jpegs).items())
        },
        "jpeg_quality_exact": sum(bool(i["jpeg_quality_exact"]) for i in jpegs),
        "subsampling_counts": dict(Counter(i["subsampling"] for i in jpegs)),
        "progressive": sum(bool(i["progressive"]) for i in jpegs),
        "has_exif": sum(i["has_exif"] for i in ok),
        "has_icc": sum(i["has_icc"] for i in ok),
        "width": _summary(i["width"] for i in ok),
        "height": _summary(i["height"] for i in ok),
        "min_side_lt_256": sum(s < MIN_SIDE for s in min_sides),
        "top_sizes": dict(sizes.most_common(10)),
        "distinct_sizes": len(sizes),
        "file_bytes": _summary(i["bytes"] for i in ok),
        "blockiness": _summary(b for b in blocks if b is not None),
        "errors": len(errors),
        "error_examples": errors[:3],
    }


# --------------------------------------------------------------------------------------
# Masks (in folders or inside archives)
# --------------------------------------------------------------------------------------


def _mask_values(source: Path | bytes) -> dict[str, Any]:
    head = source[:16] if isinstance(source, bytes) else source.read_bytes()[:16]
    with _open_image(source) as img:
        mode, size = img.mode, img.size
        arr = np.asarray(img)
        palette = img.getpalette() is not None if mode == "P" else None
    if arr.ndim == 3:  # colour mask: count colour tuples
        flat = arr.reshape(-1, arr.shape[-1])
        vals, counts = np.unique(flat, axis=0, return_counts=True)
        keys = [",".join(str(int(c)) for c in v) for v in vals]
    else:
        vals, counts = np.unique(arr, return_counts=True)
        keys = [str(int(v)) for v in vals]
    return {
        "format": sniff_format(head),
        "mode": mode,
        "size": size,
        "palette": palette,
        "values": dict(zip(keys, (int(c) for c in counts), strict=True)),
    }


def _mask_stem(name: str) -> str:
    return Path(name).stem.removesuffix("_mask")


def mask_stats(
    masks: list[str],
    images: list[str],
    *,
    read_mask: Reader,
    read_image: Reader | None,
    sample: int,
    seed: int,
) -> dict[str, Any]:
    """Mask format, mode, distinct values (sampled) and image/mask pairing (all files)."""
    stems = {_mask_stem(n): n for n in images}
    mask_stems = {_mask_stem(m) for m in masks}
    rng = np.random.default_rng(seed)
    order = sorted(masks)
    pick = [order[i] for i in sorted(rng.permutation(len(order))[: min(sample, len(order))])]
    values_pixels: Counter[str] = Counter()
    values_masks: Counter[str] = Counter()
    fmts, modes, size_match, all_zero, errors = Counter(), Counter(), Counter(), 0, []
    for name in pick:
        try:
            mv = _mask_values(read_mask(name))
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")
            continue
        fmts[mv["format"]] += 1
        modes[f"{mv['mode']}{'+palette' if mv['palette'] else ''}"] += 1
        values_pixels.update(mv["values"])
        values_masks.update(mv["values"].keys())
        all_zero += set(mv["values"]) <= {"0", "0,0,0"}
        partner = stems.get(_mask_stem(name))
        if partner and read_image is not None:
            try:
                with _open_image(read_image(partner)) as im:
                    size_match[str(im.size == mv["size"])] += 1
            except Exception:
                size_match["unreadable"] += 1
    return {
        "images": len(images),
        "masks": len(masks),
        "masks_without_image": len(mask_stems - set(stems)),
        "images_without_mask": len(set(stems) - mask_stems),
        "masks_sampled": len(pick),
        "mask_format_counts": dict(fmts),
        "mask_mode_counts": dict(modes),
        "distinct_values": sorted(values_masks, key=lambda k: [int(x) for x in k.split(",")]),
        "masks_per_value": dict(values_masks),
        "pixels_per_value": dict(values_pixels),
        "all_zero_masks": all_zero,
        "mask_matches_image_size": dict(size_match),
        "mask_errors": errors[:5],
    }


# --------------------------------------------------------------------------------------
# Archives on Drive: list members, sample images and masks without extracting
# --------------------------------------------------------------------------------------

ARCHIVE_DATASETS = ("01_GenImage", "02_DiffSeg30k", "03_SID-Set")  # BR-Gen parts: listed only


def _is_mask_dir(member_dir: str) -> bool:
    return any("mask" in part.lower() for part in member_dir.split("/"))


def probe_archives(
    root: Path, *, sample_headers: int, sample_blockiness: int, sample_masks: int, seed: int
) -> dict[str, Any]:
    """List every zip/tar in the training-data folders and sample what is inside.

    Members are grouped by (Drive folder, folder inside the archive) across all archives
    of that Drive folder, so the tar shards of one dataset form one group per class.
    """
    archives, groups, masks = [], [], []
    readers: dict[Path, ArchiveReader] = {}
    found: dict[tuple[str, str], list[tuple[Path, str]]] = {}
    try:
        for ds in ARCHIVE_DATASETS:
            for path in sorted((root / ds).rglob("*")) if (root / ds).is_dir() else []:
                if not (path.is_file() and path.suffix.lower() in (".zip", ".tar")):
                    continue
                entry: dict[str, Any] = {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                }
                try:
                    reader = readers[path] = ArchiveReader(path)
                    members = reader.members()
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    archives.append(entry)
                    continue
                per_dir = Counter(str(PurePosixPath(m.name).parent) for m in members)
                entry.update(
                    kind=reader.kind,
                    members=len(members),
                    uncompressed_bytes=sum(m.size for m in members),
                    extensions=dict(
                        Counter(Path(m.name).suffix.lower() or "<none>" for m in members)
                    ),
                    per_folder=dict(sorted(per_dir.items())[:200]),
                    folders=len(per_dir),
                )
                archives.append(entry)
                drive_dir = path.parent.relative_to(root).as_posix()
                for m in members:
                    if Path(m.name).suffix.lower() in IMAGE_SUFFIXES:
                        member_dir = str(PurePosixPath(m.name).parent)
                        found.setdefault((drive_dir, member_dir), []).append((path, m.name))

        for i, ((drive_dir, member_dir), items) in enumerate(sorted(found.items())):
            where = {n: p for p, n in items}

            def read(name: str, where: dict[str, Path] = where) -> bytes:
                return readers[where[name]].read(name)

            names = sorted(where)
            base = {
                "dataset": drive_dir.split("/", 1)[0],
                "drive_dir": drive_dir,
                "dir": f"{drive_dir}:{member_dir}",
                "archives": len({p for p, _ in items}),
            }
            if _is_mask_dir(member_dir):
                img_dir = member_dir.replace("masks", "images").replace("mask", "image")
                partner = found.get((drive_dir, img_dir), [])
                pwhere = {n: p for p, n in partner}

                def read_img(name: str, pwhere: dict[str, Path] = pwhere) -> bytes:
                    return readers[pwhere[name]].read(name)

                masks.append(
                    {
                        **base,
                        "image_dir": img_dir,
                        **mask_stats(
                            names,
                            sorted(pwhere),
                            read_mask=read,
                            read_image=read_img,
                            sample=sample_masks,
                            seed=seed + i,
                        ),
                    }
                )
            else:
                groups.append(
                    {
                        **base,
                        **group_stats(
                            None,
                            names,
                            sample_headers=sample_headers,
                            sample_blockiness=sample_blockiness,
                            seed=seed + i,
                            reader=read,
                        ),
                    }
                )
    finally:
        for reader in readers.values():
            reader.close()
    return {"archives": archives, "groups": groups, "masks": masks}


# --------------------------------------------------------------------------------------
# Dataset-specific probes
# --------------------------------------------------------------------------------------


def _read_text(path: Path, limit: int = 400) -> str | None:
    try:
        return path.read_text(errors="replace")[:limit]
    except OSError:
        return None


def _csv_rows(path: Path) -> list[dict[str, str]] | None:
    try:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None


def _verified(folder: Path) -> dict[str, Any] | None:
    """The ``_DONE`` record written by ensure_data after verification, else None."""
    try:
        done = json.loads((folder / "_DONE").read_text())
    except (OSError, ValueError):
        return None
    return done if isinstance(done, dict) and done.get("verified") is True else None


def probe_genimage(root: Path, records: list[dict[str, Any]], images: dict[str, list[str]]) -> dict:
    """Validation folder (generators found), the SD v1.4 subset tars, any legacy SD v1.4 copy."""
    gi = root / "01_GenImage"
    out: dict[str, Any] = {"exists": gi.is_dir(), "subsets": {}}
    if not gi.is_dir():
        return out
    for subset in ("Validation", "SD_v1.4"):
        d, tar = gi / subset, gi / f"{subset}.tar"
        prefix = f"01_GenImage/{subset}"
        dirs = [
            r["dir"] for r in records if r["dir"] == prefix or r["dir"].startswith(prefix + "/")
        ]
        img_dirs = {k: len(v) for k, v in images.items() if k.startswith(prefix + "/")}
        # a generator counts as found only if folders named after it actually hold images
        per_gen = {
            g: sum(n for k, n in img_dirs.items() if re.search(pat, k.lower()))
            for g, pat in GENIMAGE_GENERATORS.items()
        }
        found = {g for g, n in per_gen.items() if n > 0}
        named = {
            g
            for g, pat in GENIMAGE_GENERATORS.items()
            if any(re.search(pat, x.lower()) for x in dirs)
        }
        out["subsets"][subset] = {
            "folder": d.is_dir(),
            "tar": tar.is_file(),
            "tar_bytes": tar.stat().st_size if tar.is_file() else None,
            "done_marker": _read_text(d / "_DONE"),
            "verified": _verified(d) is not None,
            "images": sum(img_dirs.values()),
            "image_folders": img_dirs,
            "generators_found": sorted(found),
            "generators_missing": sorted(set(GENIMAGE_GENERATORS) - found),
            "generators_without_images": sorted(named - found),
            "images_per_generator": {g: n for g, n in per_gen.items() if n},
            "train_images": sum(n for k, n in img_dirs.items() if "/train" in f"/{k}/"),
        }
    sub = gi / "SD_v1.4_subset"
    done = _verified(sub)
    out["sd14_subset"] = {
        "folder": sub.is_dir(),
        "tars": sorted(p.name for p in sub.glob("*.tar")) if sub.is_dir() else [],
        "manifest": (sub / "manifest.csv").is_file(),
        "verified": done is not None,
        "per_label": (done or {}).get("summary", {}).get("per_label"),
    }
    return out


def probe_diffseg(root: Path, *, sample_masks: int, seed: int) -> dict[str, Any]:
    """The published zips (preferred) and any folders left by the retired downloader."""
    ds = root / "02_DiffSeg30k"
    out: dict[str, Any] = {"exists": ds.is_dir(), "zips": {}, "legacy_folders": {}}
    if not ds.is_dir():
        return out
    for z in sorted(ds.glob("*.zip")):
        stamp = _read_text(ds / f"{z.name}.verified.json", limit=10_000)
        out["zips"][z.name] = {"bytes": z.stat().st_size, "checksum_verified": stamp is not None}
    out["verified"] = _verified(ds) is not None
    for split_dir in sorted(p for p in ds.iterdir() if p.is_dir()):
        img_dir, mask_dir = split_dir / "images", split_dir / "masks"
        imgs = sorted(p.name for p in img_dir.iterdir()) if img_dir.is_dir() else []
        msks = sorted(p.name for p in mask_dir.iterdir()) if mask_dir.is_dir() else []
        rows = _csv_rows(ds / f"{split_dir.name}_manifest.csv")
        info = mask_stats(
            msks,
            imgs,
            read_mask=lambda n, d=mask_dir: d / n,
            read_image=lambda n, d=img_dir: d / n,
            sample=sample_masks,
            seed=seed,
        )
        info["manifest_rows"] = None if rows is None else len(rows)
        out["legacy_folders"][split_dir.name] = info
    return out


def probe_sidset(root: Path, images: dict[str, list[str]]) -> dict[str, Any]:
    """Per split and label: counts from the manifest (tar layout) or the legacy folders."""
    sd = root / "03_SID-Set"
    out: dict[str, Any] = {"exists": sd.is_dir(), "splits": {}}
    if not sd.is_dir():
        return out
    out["verified"] = _verified(sd) is not None
    out["tars"] = len(list(sd.glob("*.tar")))
    for split, cap in SID_CAPS.items():
        rows = _csv_rows(sd / f"{split}_manifest.csv")
        by_label = Counter(r.get("label") for r in rows) if rows else Counter()
        masks_by_label = Counter(r.get("label") for r in rows if r.get("mask")) if rows else {}
        folders = {}
        for lab, name in SID_LABELS.items():
            folders[name] = {
                "label": lab,
                "legacy_images": len(images.get(f"03_SID-Set/{split}/{name}/images", [])),
                "legacy_masks": len(images.get(f"03_SID-Set/{split}/{name}/masks", [])),
                "manifest_rows": by_label.get(str(lab), 0),
                "manifest_rows_with_mask": masks_by_label.get(str(lab), 0),
                "cap": cap,
            }
        other = sorted(k for k in by_label if k not in {str(x) for x in SID_LABELS})
        out["splits"][split] = {
            "manifest": rows is not None,
            "manifest_rows": None if rows is None else len(rows),
            "labels": folders,
            "unexpected_labels": other,
        }
    return out


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def probe_chameleon(root: Path) -> dict[str, Any]:
    """Image counts in 05_Chameleon/test/{0_real,1_fake} and in Chameleon.zip (not extracted)."""
    ch = root / "05_Chameleon"
    out: dict[str, Any] = {"exists": ch.is_dir(), "folders": {}, "zip": None}
    for cls in ("0_real", "1_fake"):
        d = ch / "test" / cls
        out["folders"][cls] = _count_images(d) if d.is_dir() else None
    zips = sorted({*root.glob("Chameleon*.zip"), *ch.glob("*.zip")}) if root.is_dir() else []
    if zips:
        z = zips[0]
        try:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_SUFFIXES]
            counts = {
                cls: sum(f"/{cls}/" in f"/{n}" for n in names) for cls in ("0_real", "1_fake")
            }
            out["zip"] = {
                "path": z.relative_to(root).as_posix(),
                "bytes": z.stat().st_size,
                "images": len(names),
                "per_class": counts,
            }
        except (OSError, zipfile.BadZipFile) as exc:
            out["zip"] = {"path": z.relative_to(root).as_posix(), "error": type(exc).__name__}
    if out["zip"] and "per_class" in out["zip"]:
        out["folder_matches_zip"] = all(
            out["folders"][c] == out["zip"]["per_class"][c] for c in ("0_real", "1_fake")
        )
    return out


def probe_brgen(root: Path) -> dict[str, Any]:
    """Archive parts and any extracted images in 04_BR-Gen (nothing is extracted here)."""
    br = root / "04_BR-Gen"
    out: dict[str, Any] = {"exists": br.is_dir(), "archives": [], "images": 0}
    if not br.is_dir():
        return out
    for p in sorted(br.rglob("*")):
        if p.is_file() and ARCHIVE_RE.search(p.name):
            out["archives"].append(
                {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size}
            )
    out["images"] = _count_images(br)
    return out


# --------------------------------------------------------------------------------------
# Medium-profile assessment
# --------------------------------------------------------------------------------------


def assess_medium(probe: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per expected item of the medium profile.

    Status: ``present`` (complete; for downloads also verified by ensure_data),
    ``partial``, ``missing``, ``archive`` (BR-Gen split zip, test only) or
    ``not verified`` (files there, but ensure_data has not verified them yet).
    """
    rows: list[dict[str, Any]] = []

    def add(item: str, status: str, expected: str, present: str) -> None:
        rows.append({"item": item, "status": status, "expected": expected, "present": present})

    gi = probe["genimage"]
    val = gi.get("subsets", {}).get("Validation", {})
    if not val or not (val.get("folder") or val.get("tar")):
        add("GenImage Validation", "missing", "8 generators (real + generated)", "nothing")
    else:
        n = len(val["generators_found"])
        add(
            "GenImage Validation",
            # the old downloader wrote _DONE only after a subset had been copied completely
            "present" if n == 8 and val["done_marker"] else "partial",
            "8 generators, _DONE marker",
            f"{n} found ({', '.join(val['generators_found'])}); missing: "
            f"{', '.join(val['generators_missing']) or 'none'}; {val['images']} images; "
            f"_DONE marker {'present' if val['done_marker'] else 'absent'}; "
            f"verified by ensure_data: {'yes' if val['verified'] else 'no'}"
            + (
                f"; folders without images: {', '.join(val['generators_without_images'])}"
                if val["generators_without_images"]
                else ""
            ),
        )
    sub = gi.get("sd14_subset", {})
    legacy = gi.get("subsets", {}).get("SD_v1.4", {})
    legacy_note = (
        f"; legacy SD_v1.4 folder: {legacy.get('images', 0)} images" if legacy.get("folder") else ""
    )
    if sub.get("verified"):
        add(
            "GenImage SD v1.4 subset",
            "present",
            "40,000 real + 40,000 generated (seed 42)",
            f"{len(sub['tars'])} tar files, per label {sub['per_label']}{legacy_note}",
        )
    elif sub.get("tars"):
        add(
            "GenImage SD v1.4 subset",
            "partial",
            "40,000 real + 40,000 generated (seed 42)",
            f"{len(sub['tars'])} tar files, not verified yet{legacy_note}",
        )
    else:
        add(
            "GenImage SD v1.4 subset",
            "missing",
            "40,000 real + 40,000 generated (seed 42)",
            f"no subset{legacy_note}",
        )

    dfs = probe["diffseg30k"]
    info = (probe.get("hf_cards", {}).get(HF_REPOS["diffseg30k"]) or {}).get("split_sizes") or {}
    zips = dfs.get("zips", {})
    expected = {"train.zip", "validation.zip"} | {f"{s}.zip" for s in info}
    for name in sorted(expected):
        z = zips.get(name)
        split = name.removesuffix(".zip")
        groups = [
            m
            for m in probe.get("archives", {}).get("masks", [])
            if m["drive_dir"] == "02_DiffSeg30k"
            and m["archives"] >= 1
            and m["dir"].split(":", 1)[1].split("/", 1)[0] == split
        ]
        unpaired = sum(m["masks_without_image"] + m["images_without_mask"] for m in groups)
        exp = f"{name} as published" + (
            f" ({info[split]} examples on the card)" if split in info else ""
        )
        if z is None:
            add(f"DiffSeg30k {name}", "missing", exp, "no file")
        elif not z["checksum_verified"]:
            add(f"DiffSeg30k {name}", "not verified", exp, f"{z['bytes']} bytes")
        else:
            add(
                f"DiffSeg30k {name}",
                "present" if unpaired == 0 else "partial",
                exp,
                f"{z['bytes']} bytes, checksum verified, {unpaired} unpaired image/mask names",
            )

    sid = probe["sidset"]
    if not sid.get("exists"):
        add("SID-Set", "missing", "10k/label train, 2k/label validation", "no folder")
    for split, s in sid.get("splits", {}).items():
        for name, lab in s["labels"].items():
            n, cap = lab["manifest_rows"], lab["cap"]
            mask_ok = name != "tampered" or lab["manifest_rows_with_mask"] >= cap
            if sid.get("verified") and n >= cap and mask_ok:
                status = "present"
            elif n == 0:
                status = "missing"
            else:
                status = "partial" if sid.get("verified") else "not verified"
            add(
                f"SID-Set {split}/{name}",
                status,
                f"{cap} images" + (" + masks" if name == "tampered" else ""),
                f"{n} in manifest, {lab['manifest_rows_with_mask']} with mask; legacy loose "
                f"files: {lab['legacy_images']} images, {lab['legacy_masks']} masks",
            )

    br = probe["brgen"]
    if not br.get("exists"):
        add("BR-Gen", "missing", "test only, handled before P3 (not CP1)", "no folder")
    else:
        status = "present" if br["images"] else ("archive" if br["archives"] else "missing")
        add(
            "BR-Gen",
            status,
            "test only, handled before P3 (not CP1)",
            f"{br['images']} images; {len(br['archives'])} archive files, "
            f"{sum(a['bytes'] for a in br['archives']) / 1e9:.1f} GB",
        )

    ch = probe["chameleon"]
    counts = ch.get("folders", {})
    have = all(counts.get(c) for c in ("0_real", "1_fake"))
    add(
        "Chameleon",
        "present" if have else "missing",
        "test/0_real and test/1_fake",
        f"0_real {counts.get('0_real')}, 1_fake {counts.get('1_fake')}",
    )

    mm = probe["datasets"].get("06_MMTD-Set", {})
    add(
        "MMTD-Set",
        "present" if mm.get("files") else "missing",
        "HF snapshot (JSON)",
        f"{mm.get('files', 0)} files",
    )
    return rows


# --------------------------------------------------------------------------------------
# Hugging Face dataset cards (network; Colab only)
# --------------------------------------------------------------------------------------

Opener = Callable[[urllib.request.Request, float], Any]


def _get(url: str, token_getter: Callable[[], str | None] | None, opener: Opener) -> tuple:
    """GET ``url``; on 401/403 retry once with a token. Returns (status, text or None)."""
    req = urllib.request.Request(url, headers={"User-Agent": "eviground-probe"})
    try:
        with opener(req, 30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403) or token_getter is None:
            return exc.code, None
    except (urllib.error.URLError, OSError) as exc:
        return f"error: {type(exc).__name__}", None
    token = token_getter()
    if not token:
        return "auth required, no token", None
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with opener(req, 30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, OSError) as exc:
        return f"error: {type(exc).__name__}", None


def fetch_hf_cards(
    repos: dict[str, str] | None = None,
    token_getter: Callable[[], str | None] | None = None,
    opener: Opener | None = None,
) -> dict[str, Any]:
    """README (dataset card) and split sizes of each Hugging Face dataset.

    The token, if needed, is only placed in the request header; it is never stored.
    """
    opener = opener or (lambda req, timeout: urllib.request.urlopen(req, timeout=timeout))
    out: dict[str, Any] = {}
    for repo in (repos or HF_REPOS).values():
        entry: dict[str, Any] = {}
        status, text = _get(
            f"https://huggingface.co/datasets/{repo}/raw/main/README.md", token_getter, opener
        )
        entry["readme_status"], entry["readme"] = status, text
        if text:
            entry["readme_lines_of_interest"] = [
                line.strip()
                for line in text.splitlines()
                if any(w in line.lower() for w in EDITOR_WORDS) and line.strip()
            ][:200]
        status, text = _get(
            f"https://datasets-server.huggingface.co/info?dataset={repo}", token_getter, opener
        )
        entry["info_status"] = status
        entry["split_sizes"], entry["features"] = None, None
        if text:
            try:
                info = json.loads(text).get("dataset_info", {})
                config = next(iter(info.values()), {}) if info else {}
                entry["split_sizes"] = {
                    k: v.get("num_examples") for k, v in (config.get("splits") or {}).items()
                }
                entry["features"] = config.get("features")
            except (ValueError, AttributeError):
                entry["info_status"] = f"{status} (unparsed)"
        out[repo] = entry
    return out


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def run_probe(
    datasets_root: Path,
    *,
    sample_headers: int = 300,
    sample_blockiness: int = 40,
    sample_masks: int = 300,
    seed: int = 42,
    fetch_cards: bool = False,
    token_getter: Callable[[], str | None] | None = None,
    opener: Opener | None = None,
) -> dict[str, Any]:
    """Probe ``datasets_root`` (``Paper 1/datasets``) and return a JSON-serialisable report."""
    t0 = time.time()
    root = Path(datasets_root)
    report: dict[str, Any] = {
        "probe_version": PROBE_VERSION,
        "datasets_root": str(root),
        "exists": root.is_dir(),
        "errors": [],
    }
    try:
        usage = shutil.disk_usage(root if root.is_dir() else root.parent)
        report["disk"] = {
            "total_gb": round(usage.total / 1e9, 2),
            "used_gb": round(usage.used / 1e9, 2),
            "free_gb": round(usage.free / 1e9, 2),
        }
    except OSError as exc:
        report["disk"] = {"error": type(exc).__name__}
    if not root.is_dir():
        report["elapsed_s"] = round(time.time() - t0, 1)
        return report

    log.info("scanning %s", root)
    records, images = scan_tree(root)
    report["dirs"] = records
    report["top_level"] = [
        {
            "name": p.name,
            "dir": p.is_dir(),
            **(_subtree_totals(records, p.name) if p.is_dir() else {"bytes": p.stat().st_size}),
        }
        for p in sorted(root.iterdir())
    ]
    report["datasets"] = {
        name: {
            "exists": (root / name).is_dir(),
            **_subtree_totals(records, name),
            "done_marker": _read_text(root / name / "_DONE"),
        }
        for name in DATASET_FOLDERS
    }

    log.info("sampling %d image folders", len(images))
    groups = []
    mask_folders = []
    for i, (rel, names) in enumerate(sorted(images.items())):
        base = {"dataset": rel.split("/", 1)[0], "dir": rel}
        try:
            if _is_mask_dir(rel):  # loose masks (e.g. SID-Set): values and pairing
                img_rel = rel.replace("masks", "images").replace("mask", "image")
                mask_folders.append(
                    {
                        **base,
                        "image_dir": img_rel,
                        **mask_stats(
                            names,
                            images.get(img_rel, []),
                            read_mask=lambda n, d=root / rel: d / n,
                            read_image=lambda n, d=root / img_rel: d / n,
                            sample=sample_masks,
                            seed=seed + i,
                        ),
                    }
                )
                continue
            stats = group_stats(
                root / rel,
                names,
                sample_headers=sample_headers,
                sample_blockiness=sample_blockiness,
                seed=seed + i,
            )
        except Exception as exc:  # keep going: one bad folder must not hide the rest
            report["errors"].append(f"group {rel}: {type(exc).__name__}: {exc}")
            continue
        groups.append({**base, **stats})
    report["groups"] = groups
    report["mask_folders"] = mask_folders

    if fetch_cards:
        report["hf_cards"] = fetch_hf_cards(token_getter=token_getter, opener=opener)
    for key, fn in (
        ("genimage", lambda: probe_genimage(root, records, images)),
        ("diffseg30k", lambda: probe_diffseg(root, sample_masks=sample_masks, seed=seed)),
        ("sidset", lambda: probe_sidset(root, images)),
        ("chameleon", lambda: probe_chameleon(root)),
        ("brgen", lambda: probe_brgen(root)),
        (
            "archives",
            lambda: probe_archives(
                root,
                sample_headers=sample_headers,
                sample_blockiness=sample_blockiness,
                sample_masks=sample_masks,
                seed=seed,
            ),
        ),
    ):
        try:
            report[key] = fn()
        except Exception as exc:
            report[key] = {"error": f"{type(exc).__name__}: {exc}"}
            report["errors"].append(f"{key}: {type(exc).__name__}: {exc}")
    try:
        report["medium_profile"] = assess_medium(report)
    except Exception as exc:
        report["errors"].append(f"assess_medium: {type(exc).__name__}: {exc}")
    report["elapsed_s"] = round(time.time() - t0, 1)
    return report


def format_summary(report: dict[str, Any]) -> str:
    """Short human-readable summary for the notebook output."""
    lines = [
        f"datasets root: {report['datasets_root']} (exists={report['exists']})",
        f"disk: {report.get('disk')}",
    ]
    for name, d in report.get("datasets", {}).items():
        lines.append(
            f"  {name:<30} exists={d['exists']!s:<5} files={d['files']:>8} "
            f"size={d['bytes'] / 1e9:8.2f} GB"
        )
    lines.append("medium profile:")
    for row in report.get("medium_profile", []):
        lines.append(f"  [{row['status']:>8}] {row['item']}: {row['present']}")
    lines.append("image groups (sampled): format / JPEG quality / blockiness median")
    for g in report.get("groups", []) + report.get("archives", {}).get("groups", []):
        q = g["jpeg_quality_hist"]
        top_q = max(q, key=q.get) if q else "-"
        b = g["blockiness"]["median"] if g["blockiness"] else float("nan")
        lines.append(
            f"  {g['dir']:<60} n={g['n_images']:>7} {g['format_counts']} "
            f"q_mode={top_q} block={b:.3f}"
        )
    for a in report.get("archives", {}).get("archives", []):
        lines.append(
            f"archive {a['path']}: {a.get('members', a.get('error'))} members, "
            f"{a.get('folders')} folders, {a['bytes'] / 1e9:.2f} GB"
        )
    for m in report.get("archives", {}).get("masks", []) + report.get("mask_folders", []):
        lines.append(
            f"masks {m['dir']}: {m['masks']} masks / {m['images']} images, values "
            f"{m['distinct_values']}, formats {m['mask_format_counts']}, "
            f"modes {m['mask_mode_counts']}, unpaired "
            f"{m['masks_without_image'] + m['images_without_mask']}"
        )
    if report.get("errors"):
        lines.append(f"errors: {report['errors']}")
    lines.append(f"elapsed: {report.get('elapsed_s')} s")
    return "\n".join(lines)
