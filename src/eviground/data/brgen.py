"""BR-Gen (P3, cross-dataset localization): list, subset and extract without unpacking it all.

Layout of the released archive (listed 17 Nov 2026; not the README's ``Au/Tp/Gt``)::

    BR-Gen/Forged/<Method>/<Region>/<Source>/<stem>_<region>.png   150,000 (5 x 2 x 3 x 5,000)
    BR-Gen/Mask/<Region>/<Source>/<stem>_<region>.png              30,000, shared by the methods
    BR-Gen/RealImage/<Source>/<Source>_image_list.txt              lists only; no real image

with methods BrushNet, LaMa, MAT, PowerPaint, SDXL; regions Background, Stuff; sources COCO,
ImageNet, Places. The README layout is still recognised as a fallback.

BR-Gen sits on Drive as a split zip (``04_BR-Gen/BR-Gen_split.z01`` ... plus the final ``.zip``),
too large to unpack on Colab's disk. ``7z`` reads a split zip's central directory from its last
volume, so the archive can be listed and single members extracted without unpacking the rest.

Decisions of 16 Nov 2026: the P3 subset is frozen before CP4 by its seed and the SHA-256 of its
sorted member list; BR-Gen is scored only after the CP4 fix decision, on the final configuration,
once per seed. Before that, only the loading and tiling path is checked on a few images, with no
trained model scored.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from eviground.eval.devsplit import file_list_sha256

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# BR-Gen's README: Au/ (authentic), Tp/ (tampered), Gt/ (masks named <tampered stem>_mask.png)
MASK_DIRS = {"gt", "mask", "masks", "groundtruth", "ground_truth", "label", "labels"}
IMAGE_DIRS = {"tp", "image", "images", "img", "imgs", "tampered", "fake"}
AUTHENTIC_DIRS = {"au", "authentic", "real"}
MASK_SUFFIX = re.compile(r"[_-](mask|gt|label)$", re.IGNORECASE)
SEED = 42
SALT = "brgen_subset_v1"
REPO = "clpbc/BR-Gen"  # Hugging Face dataset repository (public; 17 Nov 2026: z01-z12 + .zip)
PART_PREFIX = "BR-Gen_split."
LAST_VOLUME = PART_PREFIX + "zip"  # holds the central directory: 7z lists and extracts through it


def seven_zip() -> str | None:
    """Path of the 7z executable, if installed (``apt-get install p7zip-full``)."""
    for name in ("7z", "7za", "7zz"):
        found = shutil.which(name)
        if found:
            return found
    return None


def parse_slt(text: str) -> list[dict[str, Any]]:
    """Entries of ``7z l -slt`` output: one dict per member (``path``, ``size``, ``is_dir``)."""
    body = text.split("\n----------\n", 1)[-1]
    entries = []
    for block in re.split(r"\n\s*\n", body):
        fields = {}
        for line in block.splitlines():
            if " = " in line:
                k, v = line.split(" = ", 1)
                fields[k.strip()] = v.strip()
        if "Path" not in fields:
            continue
        is_dir = fields.get("Folder") == "+" or fields.get("Attributes", "").startswith("D")
        entries.append(
            {
                "path": fields["Path"].replace("\\", "/"),
                "size": int(fields.get("Size") or 0),
                "is_dir": is_dir,
            }
        )
    return entries


def list_archive(archive: Path, exe: str | None = None) -> list[dict[str, Any]]:
    """Members of ``archive`` (the last volume of a split zip), read with ``7z l -slt``."""
    exe = exe or seven_zip()
    if not exe:
        raise FileNotFoundError("7z not found: apt-get install -y p7zip-full")
    out = subprocess.run(
        [exe, "l", "-slt", "-ba", str(archive)], capture_output=True, text=True, check=True
    )
    return parse_slt("\n----------\n" + out.stdout)


def parse_member(path: str) -> dict[str, str] | None:
    """The released layout's fields of ``path``: ``kind`` (forged, mask or real_list),
    ``method`` (forged only), ``region``, ``source`` and ``name``; None for other paths."""
    parts = PurePosixPath(path).parts
    for i, part in enumerate(parts):
        rest = parts[i + 1 :]
        if part == "Forged" and len(rest) == 4:
            method, region, source, name = rest
            return {
                "kind": "forged",
                "method": method,
                "region": region,
                "source": source,
                "name": name,
            }
        if part == "Mask" and len(rest) == 3:
            region, source, name = rest
            return {"kind": "mask", "region": region, "source": source, "name": name}
        if part == "RealImage" and len(rest) == 2:
            return {"kind": "real_list", "source": rest[0], "name": rest[1]}
    return None


def is_image(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in IMAGE_EXT


def is_mask(path: str) -> bool:
    p = PurePosixPath(path)
    dirs = {d.lower() for d in p.parent.parts}
    return bool(dirs & MASK_DIRS) or bool(MASK_SUFFIX.search(p.stem))


def kind(path: str) -> str:
    """'mask', 'authentic', 'image' (tampered or unknown) or 'other'."""
    if not is_image(path):
        return "other"
    member = parse_member(path)
    if member:
        return {"forged": "image", "mask": "mask"}.get(member["kind"], "other")
    if is_mask(path):
        return "mask"
    if {d.lower() for d in PurePosixPath(path).parent.parts} & AUTHENTIC_DIRS:
        return "authentic"
    return "image"


def method_of(path: str) -> str:
    """The inpainting method: the folder under ``Forged/`` in the released layout, else the
    file name's prefix (README layout: ``MAT_Stuff_COCO_...`` -> ``MAT``)."""
    member = parse_member(path)
    if member and member["kind"] == "forged":
        return member["method"]
    return PurePosixPath(path).name.split("_", 1)[0]


def stratum_key(path: str, depth: int) -> str:
    """``method/region/source`` in the released layout (``depth`` is not used there); else the
    first ``depth`` folders and the file's generator prefix."""
    member = parse_member(path)
    if member and member["kind"] == "forged":
        return "/".join((member["method"], member["region"], member["source"]))
    parts = PurePosixPath(path).parts
    return "/".join([*parts[: min(depth, len(parts) - 1)], method_of(path)])


def summarize(entries: Sequence[Mapping[str, Any]], depth: int = 3, examples: int = 3) -> dict:
    """Counts per directory (up to ``depth`` levels), per extension and image / mask kind."""
    files = [e for e in entries if not e["is_dir"]]
    dirs: dict[str, Counter] = defaultdict(Counter)
    sample: dict[str, list[str]] = defaultdict(list)
    for e in files:
        parts = PurePosixPath(e["path"]).parts
        for d in range(1, min(depth, len(parts) - 1) + 1):
            key = "/".join(parts[:d])
            dirs[key]["files"] += 1
            dirs[key][PurePosixPath(e["path"]).suffix.lower() or "(none)"] += 1
        top = "/".join(parts[: min(depth, len(parts) - 1)])
        if len(sample[top]) < examples:
            sample[top].append(e["path"])
    kinds = Counter(kind(e["path"]) for e in files)
    members = [m for m in (parse_member(e["path"]) for e in files) if m]
    strata = Counter(
        "/".join((m["method"], m["region"], m["source"])) for m in members if m["kind"] == "forged"
    )
    mask_strata = Counter(
        "/".join((m["region"], m["source"])) for m in members if m["kind"] == "mask"
    )
    sources = {m["source"] for m in members if m["kind"] == "forged"}
    stems = {(m["source"], m["name"].rsplit("_", 1)[0]) for m in members if m["kind"] == "forged"}
    return {
        "layout": "released (Forged/Mask/RealImage)" if members else "other",
        "strata": dict(sorted(strata.items())),
        "mask_strata": dict(sorted(mask_strata.items())),
        "source_images": {s: sum(1 for x in stems if x[0] == s) for s in sorted(sources)},
        "real_lists": sorted(
            e["path"] for e in files if (parse_member(e["path"]) or {}).get("kind") == "real_list"
        ),
        "members": len(entries),
        "files": len(files),
        "bytes": sum(e["size"] for e in files),
        "kinds": dict(kinds),
        "extensions": dict(Counter(PurePosixPath(e["path"]).suffix.lower() for e in files)),
        "directories": {k: dict(v) for k, v in sorted(dirs.items())},
        "examples": dict(sorted(sample.items())),
    }


def _pair_key(path: str) -> str:
    """``region/source/name`` in the released layout (a forged file and its mask share it);
    else the path without its image / mask folder (``Tp`` / ``Gt``) and mask suffix."""
    member = parse_member(path)
    if member and member["kind"] in ("forged", "mask"):
        return "/".join((member["region"], member["source"], member["name"])).lower()
    p = PurePosixPath(path)
    folders = [d.lower() for d in p.parent.parts if d.lower() not in MASK_DIRS | IMAGE_DIRS]
    return "/".join([*folders, MASK_SUFFIX.sub("", p.stem).lower()])


def pair_masks(paths: Iterable[str]) -> dict[str, Any]:
    """Pairs each image with its mask: in the released layout the mask of the same region,
    source and name (one mask serves all five methods); in the README layout the mask of the same
    name stem (``x.png`` and ``x_mask.png``, or the same name under a mask folder). Returns the
    pairs and the unpaired counts; a key with more than one mask is ambiguous and left unpaired.
    """
    images, masks = {}, {}
    for p in paths:
        k = kind(p)
        if k in ("image", "mask"):
            (masks if k == "mask" else images).setdefault(_pair_key(p), []).append(p)
    pairs = {
        img: masks[k][0] for k, imgs in images.items() if len(masks.get(k, ())) == 1 for img in imgs
    }
    return {
        "pairs": pairs,
        "images_without_mask": sum(len(v) for k, v in images.items() if k not in masks),
        "masks_without_image": sum(1 for k in masks if k not in images),
        "ambiguous": sum(1 for v in masks.values() if len(v) > 1),
    }


def _rank(seed: int, salt: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{salt}|{key}".encode()).hexdigest()


def subset(
    items: Sequence[str],
    per_stratum: int | Mapping[str, int],
    stratum: Callable[[str], str],
    *,
    seed: int = SEED,
    salt: str = SALT,
) -> dict[str, Any]:
    """A seeded subset: in each stratum, the first ``per_stratum`` items ranked by a hash of
    (seed, salt, item). Frozen by the SHA-256 of the sorted chosen items."""
    groups: dict[str, list[str]] = defaultdict(list)
    for it in items:
        groups[stratum(it)].append(it)
    chosen = []
    counts = {}
    for s, its in sorted(groups.items()):
        n = per_stratum if isinstance(per_stratum, int) else per_stratum.get(s, 0)
        take = sorted(its, key=lambda x: _rank(seed, salt, x))[:n]
        chosen += take
        counts[s] = {"available": len(its), "chosen": len(take)}
    chosen.sort()
    return {
        "seed": seed,
        "salt": salt,
        "strata": counts,
        "items": chosen,
        "sha256": file_list_sha256(chosen),
    }


# --------------------------------------------------------------------------------------
# Paired subset with the training overlap left out (Dhanush, 17 Nov 2026)
# --------------------------------------------------------------------------------------

REGION_SUFFIX = re.compile(r"_(background|stuff)$", re.IGNORECASE)
# ids that file names carry: COCO image ids (12 digits), ImageNet training ids (synset_number)
ID_PATTERNS = (re.compile(r"(?<!\d)\d{12}(?!\d)"), re.compile(r"n\d{8}_\d+"))


def source_id(path: str) -> str | None:
    """The source image's id of a forged file or mask: its name without the region suffix
    (``000000232161_background.png`` -> ``000000232161``); None for other paths."""
    member = parse_member(path)
    if not member or member["kind"] not in ("forged", "mask"):
        return None
    return REGION_SUFFIX.sub("", PurePosixPath(member["name"]).stem).lower()


def training_ids(
    rows: Iterable[Mapping[str, Any]], roles: Sequence[str] = ("train_pool",)
) -> dict[str, Any]:
    """Ids of the images the models trained or were validated on (CP1 manifest rows with a role
    in ``roles``): each uid's file stem, plus every COCO or ImageNet id inside the uid.
    ``coverage`` counts, per source, the rows whose uid carries such an id: a source with none
    cannot be checked by id."""
    ids: set[str] = set()
    rows_n: Counter = Counter()
    with_id: Counter = Counter()
    for r in rows:
        if r.get("role") not in roles:
            continue
        uid = str(r["uid"])
        found = [m.lower() for pat in ID_PATTERNS for m in pat.findall(uid)]
        ids.add(PurePosixPath(uid).stem.lower())
        ids.update(found)
        rows_n[r.get("source", "")] += 1
        with_id[r.get("source", "")] += bool(found)
    coverage = {s: {"rows": n, "with_coco_or_imagenet_id": with_id[s]} for s, n in rows_n.items()}
    return {"roles": list(roles), "ids": ids, "coverage": dict(sorted(coverage.items()))}


def overlap(pairs: Mapping[str, str], ids: set[str]) -> dict[str, Any]:
    """BR-Gen source images (per source) whose id is among the training ``ids``."""
    stems: dict[str, set[str]] = defaultdict(set)
    for img in pairs:
        sid = source_id(img)
        if sid:
            stems[parse_member(img)["source"]].add(sid)
    hit = {s: sorted(v & ids) for s, v in sorted(stems.items())}
    return {
        "source_images": {s: len(v) for s, v in sorted(stems.items())},
        "in_training": {s: len(v) for s, v in hit.items()},
        "examples": {s: v[:5] for s, v in hit.items() if v},
        "ids": {i for v in hit.values() for i in v},
    }


def paired_subset(
    pairs: Mapping[str, str],
    per_group: int,
    *,
    exclude: Iterable[str] = (),
    seed: int = SEED,
    salt: str = SALT,
) -> dict[str, Any]:
    """The paired P3 subset: in each region/source group, the first ``per_group`` source images
    (ranked by a hash of seed, salt, group and id) that every method edited and that are not in
    ``exclude``; all methods' forged files of those images. Methods are thus compared on the same
    images. Frozen by the SHA-256 of the sorted forged files."""
    methods = sorted({parse_member(p)["method"] for p in pairs if source_id(p)})
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for img in pairs:
        sid = source_id(img)
        if sid:
            m = parse_member(img)
            groups[f"{m['region']}/{m['source']}"][sid].append(img)
    skip = {e.lower() for e in exclude}
    chosen: list[str] = []
    counts = {}
    for g, by_id in sorted(groups.items()):
        complete = [i for i, files in by_id.items() if len(files) == len(methods)]
        eligible = [i for i in complete if i not in skip]
        ranks = {i: _rank(seed, salt, f"{g}/{i}") for i in eligible}
        take = sorted(eligible, key=ranks.__getitem__)[:per_group]
        chosen += [f for i in take for f in by_id[i]]
        counts[g] = {
            "source_images": len(by_id),
            "not_all_methods": len(by_id) - len(complete),
            "excluded_training_overlap": len(complete) - len(eligible),
            "chosen": len(take),
            "forged_images": len(take) * len(methods),
        }
    chosen.sort()
    return {
        "design": "paired: the same source images for every method",
        "seed": seed,
        "salt": salt,
        "per_group": per_group,
        "methods": methods,
        "strata": counts,
        "items": chosen,
        "masks": {i: pairs[i] for i in chosen},
        "sha256": file_list_sha256(chosen),
    }


def sample(items: Sequence[str], n: int, *, seed: int = SEED, salt: str = SALT) -> list[str]:
    """The first ``n`` items ranked by a hash of (seed, salt, item): a seeded, order-free pick."""
    return sorted(items, key=lambda x: _rank(seed, f"{salt}/sample", x))[:n]


def extract(
    archive: Path, members: Sequence[str], out_dir: Path, exe: str | None = None
) -> list[Path]:
    """Extract only ``members`` (with their paths) into ``out_dir``."""
    exe = exe or seven_zip()
    if not exe:
        raise FileNotFoundError("7z not found: apt-get install -y p7zip-full")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listfile = out_dir / "_members.txt"
    listfile.write_text("\n".join(members) + "\n")
    subprocess.run(
        [exe, "x", "-y", f"-o{out_dir}", str(archive), f"@{listfile}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [out_dir / m for m in members]


METHODS = ("BrushNet", "LaMa", "MAT", "PowerPaint", "SDXL")
REGIONS = ("Background", "Stuff")
SOURCES = ("COCO", "ImageNet", "Places")


def released_paths(
    n_per_source: int = 2, ids: Mapping[str, Sequence[str]] | None = None
) -> list[str]:
    """Paths of a small archive in the released layout (tests and dry runs only). ``ids`` gives
    the source-image ids of a source (e.g. COCO ids, to test the training-overlap exclusion);
    other sources get ``<source>_<i>``."""
    paths = []
    for source in SOURCES:
        stems = list((ids or {}).get(source, ()))
        stems += [f"{source.lower()}_{i:06d}" for i in range(len(stems), n_per_source)]
        for stem in stems:
            for region in REGIONS:
                name = f"{stem}_{region.lower()}.png"
                paths.append(f"BR-Gen/Mask/{region}/{source}/{name}")
                paths += [f"BR-Gen/Forged/{m}/{region}/{source}/{name}" for m in METHODS]
        paths.append(f"BR-Gen/RealImage/{source}/{source}_image_list.txt")
    return paths


def fake_listing(n_per_method: int = 6, layout: str = "readme") -> str:
    """``7z l -slt`` text of a small archive (tests and dry runs only). ``layout="readme"``: as
    BR-Gen's README shows (Au/, Tp/, Gt/; masks named ``<tampered stem>_mask.png``);
    ``"released"``: as the archive really is (:func:`released_paths`, ``n_per_method`` source
    images per source)."""
    lines = ["7-Zip (z) fake", "", "----------"]

    def entry(path: str, size: int) -> None:
        lines.extend([f"Path = {path}", "Folder = -", f"Size = {size}", ""])

    if layout == "released":
        for i, path in enumerate(released_paths(n_per_method)):
            entry(path, 100 + i)
        return "\n".join(lines)
    for region in ("Stuff", "Background"):
        root = f"BR-Gen/BRGen-{region}-Val"
        for i in range(n_per_method):
            src = f"COCO_{i:011d}"
            entry(f"{root}/Au/{src}.png", 900 + i)
            for method in ("MAT", "SDXL"):
                entry(f"{root}/Tp/{method}_{region}_{src}.png", 1000 + i)
                entry(f"{root}/Gt/{method}_{region}_{src}_mask.png", 100 + i)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Completing the split archive on Drive (17 Nov 2026: Drive held z01-z08 only)
# --------------------------------------------------------------------------------------


def _published_parts(ctx: Any, repo: str, prefix: str) -> dict[str, int]:
    from eviground.data.storage import VerificationError

    expected = {
        name: size
        for name, size in ctx.sources.hf_list_files(repo).items()
        if name.startswith(prefix) and "/" not in name
    }
    if not expected:
        raise VerificationError(f"{repo} publishes no file starting with {prefix!r}")
    return expected


def _stamped(folder: Path, name: str, size: int) -> bool:
    from eviground.data.storage import read_json

    stamp = read_json(folder / f"{name}.verified.json")
    return isinstance(stamp, dict) and stamp.get("bytes") == size


def _fetch_part(ctx: Any, repo: str, folder: Path, name: str, size: int, sha: str | None) -> int:
    """Downloads one volume after the space checks, verifies and stamps it. Returns the size of
    the wrong-size local copy it replaced (0 if there was none)."""
    from eviground.data.storage import VerificationError, sha256_file, write_json_atomic

    target = folder / name
    have = target.stat().st_size if target.is_file() else 0
    ctx.drive_space(size - have, f"{repo} {name}")
    ctx.wait_for_cache(size, f"{repo} {name}")  # the volume passes Colab's local cache
    if have:  # a wrong-size copy of a published volume: replaced by the download
        target.unlink()
    got = Path(ctx.sources.hf_download(repo, name, folder))
    if got != target:
        raise VerificationError(f"{name} downloaded to {got}, expected {target}")
    actual, digest = target.stat().st_size, sha256_file(target)
    if actual != size or (sha and digest != sha):
        target.unlink(missing_ok=True)
        raise VerificationError(
            f"{name}: {actual} bytes / sha256 {digest[:12]}..., expected {size} / "
            f"{str(sha)[:12]}...; the file was removed, run the cell again"
        )
    write_json_atomic(folder / f"{name}.verified.json", {"bytes": size, "sha256": digest})
    return have


def complete_split(
    ctx: Any,
    folder: Path,
    repo: str = REPO,
    prefix: str = PART_PREFIX,
) -> dict[str, Any]:
    """Makes ``folder`` hold every volume of the split archive published in ``repo``.

    ``ctx`` is a :class:`eviground.data.download.Context` (Drive-quota check before each write,
    the wait for Colab's upload cache, lazily created sources). A volume already there with
    exactly the published size is kept and only size-checked (hashing 5 GiB volumes through
    the Drive mount is too slow); a missing or wrong-size volume is downloaded, then checked
    against the published size and SHA-256 and stamped (``<name>.verified.json``); a mismatch
    removes the file and raises. A wrong-size local copy is removed (after the space check)
    and downloaded again; it is listed under ``replaced``. Local files with the prefix that the
    repo does not publish are reported, never removed."""
    import time

    t0 = time.monotonic()
    folder = Path(folder)
    expected = _published_parts(ctx, repo, prefix)
    present = {
        n
        for n, size in expected.items()
        if (folder / n).is_file() and (folder / n).stat().st_size == size
    }
    missing = sorted(set(expected) - present)
    shas: dict[str, str | None] = {}
    if missing:  # refuse up front if all the missing volumes do not fit, before any download
        need = sum(
            expected[n] - ((folder / n).stat().st_size if (folder / n).is_file() else 0)
            for n in missing
        )
        ctx.drive_space(need, f"{repo}: {len(missing)} missing volumes")
        shas = {k: v["sha256"] for k, v in ctx.sources.hf_file_info(repo, missing).items()}
    replaced = []
    for name in missing:
        had = _fetch_part(ctx, repo, folder, name, expected[name], shas.get(name))
        if had:
            replaced.append({"name": name, "bytes": had})
    extra = sorted(
        p.name
        for p in folder.glob(f"{prefix}*")
        if p.is_file() and p.name not in expected and not p.name.endswith(".verified.json")
    )
    return {
        "repo": repo,
        "expected": dict(sorted(expected.items())),
        "kept": sorted(present),
        "size_only": sorted(n for n in present if not _stamped(folder, n, expected[n])),
        "downloaded": missing,
        "replaced": replaced,
        "extra": extra,
        "bytes_downloaded": sum(expected[n] for n in missing),
        "seconds": round(time.monotonic() - t0, 1),
    }
