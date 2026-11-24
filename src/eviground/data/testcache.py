"""One verified tar per test set on Drive (Dhanush, 24 Nov 2026).

Every GPU notebook from Phase C on stages its test images only by copying these tars and checking
them; nothing is extracted from the original archives (7z, zips) on a GPU runtime again.

- :func:`build`: the original file bytes of every row (image and, if any, mask; not re-encoded, so
  JPEG metadata and the harmonized-P1 path still work) in one tar, with a manifest: uid -> member
  names, the SHA-256 of each member and of the tar, and each row's fields (paths left out).
- :func:`publish`: copies the tar and its manifest to Drive, re-hashes the tar **from Drive** and
  decodes one seeded random member from the Drive copy; both checks go into the manifest.
- :func:`stage`: copies a tar from Drive, checks its SHA-256 and every member's, refuses any member
  the manifest does not name, and returns the rows with local paths, ready for
  :func:`eviground.eval.testsets.file_items`.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image

PATH_KEYS = ("path", "mask_path")
CHUNK = 1 << 22


class CacheError(RuntimeError):
    """A cached tar is missing, incomplete or differs from its manifest."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _stem(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()[:24]


def paths(folder: Path, name: str) -> tuple[Path, Path]:
    """The tar and manifest paths of test set ``name`` in ``folder``."""
    return Path(folder) / f"{name}.tar", Path(folder) / f"{name}_manifest.json"


def build(rows: Iterable[Mapping[str, Any]], name: str, out_dir: Path) -> dict[str, Any]:
    """Write ``<name>.tar`` and ``<name>_manifest.json`` in ``out_dir`` from ``rows`` (each with
    ``uid``, ``path`` and optionally ``mask_path``)."""
    tar_path, man_path = paths(out_dir, name)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    with tarfile.open(tar_path, "w") as tf:
        for r in rows:
            uid = r["uid"]
            if uid in entries:
                raise CacheError(f"{name}: uid {uid} appears twice")
            rec: dict[str, Any] = {"row": {k: v for k, v in r.items() if k not in PATH_KEYS}}
            for key, kind in (("path", "image"), ("mask_path", "mask")):
                src = r.get(key)
                if not src:
                    continue
                src = Path(src)
                if not src.is_file():
                    raise CacheError(f"{name}: {kind} of {uid} missing: {src}")
                member = f"{_stem(uid)}.{kind}{src.suffix.lower()}"
                tf.add(src, arcname=member, recursive=False)
                rec[kind] = member
                rec[f"{kind}_sha256"] = sha256_file(src)
            if "image" not in rec:
                raise CacheError(f"{name}: row {uid} has no image path")
            entries[uid] = rec
    manifest = {
        "name": name,
        "images": len(entries),
        "masks": sum("mask" in e for e in entries.values()),
        "tar_bytes": tar_path.stat().st_size,
        "tar_sha256": sha256_file(tar_path),
        "entries": entries,
    }
    man_path.write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def _decode_member(tar_path: Path, member: str) -> list[int]:
    with tarfile.open(tar_path) as tf:
        data = tf.extractfile(member).read()
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return [img.height, img.width]


def publish(out_dir: Path, name: str, drive_dir: Path, *, seed: int = 0) -> dict[str, Any]:
    """Copy test set ``name`` from ``out_dir`` to ``drive_dir``; re-hash the tar from Drive and
    decode one seeded random member of the Drive copy (Dhanush, 24 Nov). Either check failing
    raises :class:`CacheError` and the manifest is not written to Drive."""
    src_tar, src_man = paths(out_dir, name)
    dst_tar, dst_man = paths(drive_dir, name)
    manifest = json.loads(src_man.read_text())
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_tar, dst_tar)
    drive_sha = sha256_file(dst_tar)
    if drive_sha != manifest["tar_sha256"]:
        raise CacheError(f"{name}: the Drive copy hashes to {drive_sha[:12]}, not the tar's")
    uid = random.Random(seed).choice(sorted(manifest["entries"]))
    member = manifest["entries"][uid]["image"]
    try:
        shape = _decode_member(dst_tar, member)
    except Exception as e:  # noqa: BLE001 - any decode failure stops
        msg = f"{name}: {member} ({uid}) does not decode from the Drive copy: {e}"
        raise CacheError(msg) from e
    manifest["drive_sha256_ok"] = True
    manifest["decode_check"] = {"uid": uid, "member": member, "shape": shape}
    src_man.write_text(json.dumps(manifest, indent=1, default=str))
    dst_man.write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def is_ready(drive_dir: Path, name: str) -> bool:
    """True when the tar and a manifest that passed :func:`publish`'s checks are on Drive and
    the tar has the recorded size (the full hash is checked again by :func:`stage`)."""
    tar_path, man_path = paths(drive_dir, name)
    if not (tar_path.is_file() and man_path.is_file()):
        return False
    try:
        manifest = json.loads(man_path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(manifest.get("drive_sha256_ok")) and tar_path.stat().st_size == manifest.get(
        "tar_bytes"
    )


def stage(drive_dir: Path, name: str, local_dir: Path) -> list[dict[str, Any]]:
    """Copy and unpack test set ``name`` into ``local_dir`` with every check; returns its rows
    with ``path`` / ``mask_path`` pointing at the local files, in the manifest's order."""
    tar_src, man_src = paths(drive_dir, name)
    if not is_ready(drive_dir, name):
        raise CacheError(f"{name}: no verified tar on Drive at {tar_src}")
    manifest = json.loads(man_src.read_text())
    local_dir = Path(local_dir) / name
    local_dir.mkdir(parents=True, exist_ok=True)
    local_tar = local_dir.parent / f"{name}.tar"
    shutil.copyfile(tar_src, local_tar)
    if sha256_file(local_tar) != manifest["tar_sha256"]:
        raise CacheError(f"{name}: the copied tar differs from its manifest")
    allowed = {
        rec[k]: rec[f"{k}_sha256"]
        for rec in manifest["entries"].values()
        for k in ("image", "mask")
        if k in rec
    }
    seen = set()
    with tarfile.open(local_tar) as tf:
        for m in tf.getmembers():
            if m.name not in allowed or not m.isfile():
                raise CacheError(f"{name}: unexpected member {m.name}")
            data = tf.extractfile(m).read()
            if hashlib.sha256(data).hexdigest() != allowed[m.name]:
                raise CacheError(f"{name}: member {m.name} differs from the manifest")
            (local_dir / m.name).write_bytes(data)
            seen.add(m.name)
    local_tar.unlink()
    if missing := sorted(set(allowed) - seen):
        raise CacheError(f"{name}: {len(missing)} members missing from the tar, e.g. {missing[0]}")
    rows = []
    for uid, rec in manifest["entries"].items():
        row = dict(rec["row"], uid=uid, path=str(local_dir / rec["image"]))
        if "mask" in rec:
            row["mask_path"] = str(local_dir / rec["mask"])
        rows.append(row)
    return rows
