"""BR-Gen (P3, cross-dataset localization): list, subset and extract without unpacking it all.

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
    if is_mask(path):
        return "mask"
    if {d.lower() for d in PurePosixPath(path).parent.parts} & AUTHENTIC_DIRS:
        return "authentic"
    return "image"


def method_of(path: str) -> str:
    """The generator prefix of a BR-Gen file name (``MAT_Stuff_COCO_...`` -> ``MAT``)."""
    return PurePosixPath(path).name.split("_", 1)[0]


def stratum_key(path: str, depth: int) -> str:
    """The first ``depth`` folders and the file's generator prefix."""
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
    return {
        "members": len(entries),
        "files": len(files),
        "bytes": sum(e["size"] for e in files),
        "kinds": dict(kinds),
        "extensions": dict(Counter(PurePosixPath(e["path"]).suffix.lower() for e in files)),
        "directories": {k: dict(v) for k, v in sorted(dirs.items())},
        "examples": dict(sorted(sample.items())),
    }


def _pair_key(path: str) -> str:
    """The path without its image / mask folder (``Tp`` / ``Gt``) and without a mask suffix."""
    p = PurePosixPath(path)
    folders = [d.lower() for d in p.parent.parts if d.lower() not in MASK_DIRS | IMAGE_DIRS]
    return "/".join([*folders, MASK_SUFFIX.sub("", p.stem).lower()])


def pair_masks(paths: Iterable[str]) -> dict[str, Any]:
    """Pairs each image with the mask of the same name stem (``x.png`` and ``x_mask.png``, or
    the same name under a mask folder); returns pairs and the unpaired counts."""
    images, masks = {}, {}
    for p in paths:
        k = kind(p)
        if k in ("image", "mask"):
            (masks if k == "mask" else images).setdefault(_pair_key(p), []).append(p)
    pairs = {i[0]: masks[k][0] for k, i in images.items() if k in masks and len(i) == 1}
    return {
        "pairs": pairs,
        "images_without_mask": sum(1 for k in images if k not in masks),
        "masks_without_image": sum(1 for k in masks if k not in images),
        "ambiguous": sum(1 for k, v in images.items() if len(v) > 1),
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


def fake_listing(n_per_method: int = 6) -> str:
    """``7z l -slt`` text of a small archive laid out as BR-Gen's README shows (Au/, Tp/, Gt/;
    masks named ``<tampered stem>_mask.png``); tests and dry runs only."""
    lines = ["7-Zip (z) fake", "", "----------"]

    def entry(path: str, size: int) -> None:
        lines.extend([f"Path = {path}", "Folder = -", f"Size = {size}", ""])

    for region in ("Stuff", "Background"):
        root = f"BR-Gen/BRGen-{region}-Val"
        for i in range(n_per_method):
            src = f"COCO_{i:011d}"
            entry(f"{root}/Au/{src}.png", 900 + i)
            for method in ("MAT", "SDXL"):
                entry(f"{root}/Tp/{method}_{region}_{src}.png", 1000 + i)
                entry(f"{root}/Gt/{method}_{region}_{src}_mask.png", 100 + i)
    return "\n".join(lines)
