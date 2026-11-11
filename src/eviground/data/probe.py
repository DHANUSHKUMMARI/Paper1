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
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.data.imageinfo import blockiness, header_info, sniff_format

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
    "sd14": r"sdv4|sd[-_ .]?v?1[-_.]?4|stable[-_ ]?diffusion[-_ ]?v?1[-_.]4",
    "sd15": r"sdv5|sd[-_ .]?v?1[-_.]?5|stable[-_ ]?diffusion[-_ ]?v?1[-_.]5",
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


def _blockiness_of(path: Path) -> float | None:
    try:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))
        return blockiness(arr)
    except Exception:
        return None


def group_stats(
    folder: Path,
    names: list[str],
    *,
    sample_headers: int,
    sample_blockiness: int,
    seed: int,
    workers: int = 16,
) -> dict[str, Any]:
    """Format, JPEG quality, subsampling, size and blockiness statistics of one folder.

    Header facts use a seeded sample of up to ``sample_headers`` files; blockiness decodes
    the first ``sample_blockiness`` of that sample.
    """
    rng = np.random.default_rng(seed)
    order = sorted(names)
    idx = rng.permutation(len(order))[: min(sample_headers, len(order))]
    sample = [order[i] for i in sorted(idx)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        infos = list(pool.map(lambda n: header_info(folder / n), sample))
        blocks = list(pool.map(lambda n: _blockiness_of(folder / n), sample[:sample_blockiness]))

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


def probe_genimage(root: Path, records: list[dict[str, Any]], images: dict[str, list[str]]) -> dict:
    """Which subsets and generators are present; image counts per folder."""
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
        found = {
            g
            for g, pat in GENIMAGE_GENERATORS.items()
            if any(re.search(pat, x.lower()) for x in dirs)
        }
        out["subsets"][subset] = {
            "folder": d.is_dir(),
            "tar": tar.is_file(),
            "tar_bytes": tar.stat().st_size if tar.is_file() else None,
            "done_marker": _read_text(d / "_DONE"),
            "images": sum(img_dirs.values()),
            "image_folders": img_dirs,
            "generators_found": sorted(found),
            "generators_missing": sorted(set(GENIMAGE_GENERATORS) - found),
            "train_images": sum(n for k, n in img_dirs.items() if "/train" in f"/{k}/"),
        }
    return out


def _mask_values(path: Path) -> dict[str, Any]:
    head = path.read_bytes()[:16]
    with Image.open(path) as img:
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


def probe_diffseg(root: Path, *, sample_masks: int, seed: int) -> dict[str, Any]:
    """Per split: images, masks, manifest rows, mask format and distinct mask values."""
    ds = root / "02_DiffSeg30k"
    out: dict[str, Any] = {"exists": ds.is_dir(), "splits": {}}
    if not ds.is_dir():
        return out
    rng = np.random.default_rng(seed)
    splits = sorted(
        {p.parent.name for p in ds.glob("*/images")}
        | {p.parent.name for p in ds.glob("*/masks")}
        | {p.name[: -len("_manifest.csv")] for p in ds.glob("*_manifest.csv")}
    )
    for split in splits:
        img_dir, mask_dir = ds / split / "images", ds / split / "masks"
        imgs = sorted(p.name for p in img_dir.iterdir()) if img_dir.is_dir() else []
        masks = sorted(p.name for p in mask_dir.iterdir()) if mask_dir.is_dir() else []
        rows = _csv_rows(ds / f"{split}_manifest.csv")
        info: dict[str, Any] = {
            "images": len(imgs),
            "masks": len(masks),
            "manifest_rows": None if rows is None else len(rows),
        }
        stems = {Path(n).stem: n for n in imgs}
        unpaired = [m for m in masks if Path(m).stem.removesuffix("_mask") not in stems]
        info["masks_without_image"] = len(unpaired)
        info["images_without_mask"] = len(
            set(stems) - {Path(m).stem.removesuffix("_mask") for m in masks}
        )
        pick = sorted(rng.permutation(len(masks))[: min(sample_masks, len(masks))])
        values_pixels: Counter[str] = Counter()
        values_masks: Counter[str] = Counter()
        fmts, modes, size_match, all_zero, errors = Counter(), Counter(), Counter(), 0, []
        for i in pick:
            name = masks[i]
            try:
                mv = _mask_values(mask_dir / name)
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}")
                continue
            fmts[mv["format"]] += 1
            modes[f"{mv['mode']}{'+palette' if mv['palette'] else ''}"] += 1
            values_pixels.update(mv["values"])
            values_masks.update(mv["values"].keys())
            all_zero += set(mv["values"]) <= {"0", "0,0,0"}
            stem = Path(name).stem.removesuffix("_mask")
            if stem in stems:
                try:
                    with Image.open(img_dir / stems[stem]) as im:
                        size_match[str(im.size == mv["size"])] += 1
                except Exception:
                    size_match["unreadable"] += 1
        info.update(
            masks_sampled=len(pick),
            mask_format_counts=dict(fmts),
            mask_mode_counts=dict(modes),
            distinct_values=sorted(values_masks, key=lambda k: [int(x) for x in k.split(",")]),
            masks_per_value=dict(values_masks),
            pixels_per_value=dict(values_pixels),
            all_zero_masks=all_zero,
            mask_matches_image_size=dict(size_match),
            mask_errors=errors[:5],
        )
        out["splits"][split] = info
    return out


def probe_sidset(root: Path, images: dict[str, list[str]]) -> dict[str, Any]:
    """Per split and label: folder image/mask counts, manifest counts, against the caps."""
    sd = root / "03_SID-Set"
    out: dict[str, Any] = {"exists": sd.is_dir(), "splits": {}}
    if not sd.is_dir():
        return out
    for split, cap in SID_CAPS.items():
        rows = _csv_rows(sd / f"{split}_manifest.csv")
        by_label = Counter(r.get("label") for r in rows) if rows else Counter()
        masks_by_label = Counter(r.get("label") for r in rows if r.get("mask")) if rows else {}
        folders = {}
        for lab, name in SID_LABELS.items():
            folders[name] = {
                "label": lab,
                "images": len(images.get(f"03_SID-Set/{split}/{name}/images", [])),
                "masks": len(images.get(f"03_SID-Set/{split}/{name}/masks", [])),
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
    """One row per expected item of the medium profile: present, partial or missing."""
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
            "present" if n == 8 else "partial",
            "8 generators",
            f"{n} found ({', '.join(val['generators_found'])}); missing: "
            f"{', '.join(val['generators_missing']) or 'none'}; {val['images']} images",
        )
    sd = gi.get("subsets", {}).get("SD_v1.4", {})
    if not sd or not (sd.get("folder") or sd.get("tar")):
        add("GenImage SD v1.4 (train)", "missing", "train set, real + generated", "nothing")
    elif sd.get("tar"):
        add("GenImage SD v1.4 (train)", "archive", "train set", f"{sd['tar_bytes']} bytes tar")
    else:
        add(
            "GenImage SD v1.4 (train)",
            "present" if sd["train_images"] else "partial",
            "train set, real + generated",
            f"{sd['train_images']} train images, {sd['images']} images in total",
        )

    dfs = probe["diffseg30k"]
    info = (probe.get("hf_cards", {}).get(HF_REPOS["diffseg30k"]) or {}).get("split_sizes") or {}
    if not dfs.get("exists"):
        add("DiffSeg30k", "missing", "about 30k images + masks (official 8:2 split)", "no folder")
    elif not dfs["splits"]:
        add("DiffSeg30k", "missing", "about 30k images + masks", "folder without splits")
    for split, s in dfs.get("splits", {}).items():
        exp = info.get(split)
        complete = (
            s["images"] > 0 and s["images"] == s["masks"] and (exp is None or s["images"] >= exp)
        )
        status = "present" if complete else ("missing" if s["images"] == 0 else "partial")
        add(
            f"DiffSeg30k {split}",
            status,
            f"{exp} pairs (dataset card)" if exp else "all pairs of the split",
            f"{s['images']} images, {s['masks']} masks, manifest rows {s['manifest_rows']}",
        )

    sid = probe["sidset"]
    if not sid.get("exists"):
        add("SID-Set", "missing", "10k/label train, 2k/label validation", "no folder")
    for split, s in sid.get("splits", {}).items():
        for name, lab in s["labels"].items():
            ok = lab["images"] >= lab["cap"] and (name != "tampered" or lab["masks"] >= lab["cap"])
            status = "present" if ok else ("missing" if lab["images"] == 0 else "partial")
            add(
                f"SID-Set {split}/{name}",
                status,
                f"{lab['cap']} images" + (" + masks" if name == "tampered" else ""),
                f"{lab['images']} images, {lab['masks']} masks",
            )

    br = probe["brgen"]
    if not br.get("exists"):
        add("BR-Gen", "missing", "HF snapshot", "no folder")
    else:
        status = "present" if br["images"] else ("archive" if br["archives"] else "missing")
        add(
            "BR-Gen",
            status,
            "images + masks (test only)",
            f"{br['images']} images; {len(br['archives'])} archive files",
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
    for i, (rel, names) in enumerate(sorted(images.items())):
        try:
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
        groups.append({"dataset": rel.split("/", 1)[0], "dir": rel, **stats})
    report["groups"] = groups

    if fetch_cards:
        report["hf_cards"] = fetch_hf_cards(token_getter=token_getter, opener=opener)
    for key, fn in (
        ("genimage", lambda: probe_genimage(root, records, images)),
        ("diffseg30k", lambda: probe_diffseg(root, sample_masks=sample_masks, seed=seed)),
        ("sidset", lambda: probe_sidset(root, images)),
        ("chameleon", lambda: probe_chameleon(root)),
        ("brgen", lambda: probe_brgen(root)),
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
    lines.append("image folders (sampled): format / JPEG quality / blockiness median")
    for g in report.get("groups", []):
        q = g["jpeg_quality_hist"]
        top_q = max(q, key=q.get) if q else "-"
        b = g["blockiness"]["median"] if g["blockiness"] else float("nan")
        lines.append(
            f"  {g['dir']:<60} n={g['n_images']:>7} {g['format_counts']} "
            f"q_mode={top_q} block={b:.3f}"
        )
    for split, s in report.get("diffseg30k", {}).get("splits", {}).items():
        lines.append(
            f"DiffSeg30k {split}: mask values {s.get('distinct_values')} "
            f"formats {s.get('mask_format_counts')} modes {s.get('mask_mode_counts')}"
        )
    if report.get("errors"):
        lines.append(f"errors: {report['errors']}")
    lines.append(f"elapsed: {report.get('elapsed_s')} s")
    return "\n".join(lines)
