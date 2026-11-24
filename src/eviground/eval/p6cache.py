"""Reading the P6 image cache without the detector stack (CP6 runs in a session that never
loads the detector; :mod:`eviground.eval.p6data` imports the model code through ``tiled``).

The cache is written by :func:`eviground.eval.p6data.write_cache` (notebook 15): a tar of PNG
members and a manifest with each member's SHA-256 and the item's metadata.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ASPECT_TOLERANCE = 0.02


def aspect_mismatch(
    mask_shape: Sequence[int], image_shape: Sequence[int], tolerance: float = ASPECT_TOLERANCE
) -> bool:
    """True when the mask's width/height ratio differs from the image's by more than
    ``tolerance`` (relative)."""
    h, w = int(mask_shape[0]), int(mask_shape[1])
    H, W = int(image_shape[0]), int(image_shape[1])
    return abs(w / h - W / H) / (W / H) > tolerance


def is_aspect_mismatched(meta: Mapping[str, Any], image_shape: Sequence[int]) -> bool:
    """The flag, or (for items cached before it existed, e.g. notebook 15's P6 tar) the same test
    on the recorded original mask size."""
    if meta.get("mask_aspect_mismatch"):
        return True
    orig = meta.get("mask_resized_from")
    return bool(orig) and aspect_mismatch(orig, image_shape)


class P6CacheError(RuntimeError):
    """The P6 cache differs from its manifest."""


def read_cache(tar_path: Path, manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    """Unpack the cache to ``out_dir`` and check the tar's and every member's SHA-256 against
    the manifest; any difference stops (:class:`P6CacheError`). Returns the manifest."""
    tar_path, out_dir = Path(tar_path), Path(out_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    if hashlib.sha256(tar_path.read_bytes()).hexdigest() != manifest["tar_sha256"]:
        raise P6CacheError(f"{tar_path} differs from its manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = {
        rec[k] for rec in manifest["entries"].values() for k in ("image", "mask") if k in rec
    }
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if member.name not in allowed or not member.isfile():
                raise P6CacheError(f"unexpected member in {tar_path}: {member.name}")
            (out_dir / member.name).write_bytes(tf.extractfile(member).read())
    for uid, rec in manifest["entries"].items():
        for key in ("image", "mask"):
            if key in rec:
                data = (out_dir / rec[key]).read_bytes()
                if hashlib.sha256(data).hexdigest() != rec[f"{key}_sha256"]:
                    raise P6CacheError(f"{uid}: cached {key} differs from the manifest")
    return manifest


def cached_arrays(
    uids: Iterable[str], manifest: Mapping[str, Any], folder: Path
) -> Iterator[dict[str, Any]]:
    """``{uid, image, mask, meta}`` per uid from an unpacked cache, in the given order (the mask
    is None for real and generated images)."""
    for uid in uids:
        rec = manifest["entries"][uid]
        with Image.open(Path(folder) / rec["image"]) as im:
            image = np.asarray(im.convert("RGB"))
        mask = None
        if "mask" in rec:
            with Image.open(Path(folder) / rec["mask"]) as m:
                mask = np.asarray(m.convert("L")) > 0
        yield {"uid": uid, "image": image, "mask": mask, "meta": dict(rec["meta"])}
