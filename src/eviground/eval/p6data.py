"""P6 images for the explanation evaluation (spec section 10, P6; CP5, Dhanush 19 Nov 2026).

:func:`p6_items` yields one :class:`eviground.eval.tiled.Item` per frozen P6 uid, by its part:

- ``diffseg_heldout``: the CP1 manifest file; mask by the training rule (AI base read as 0,
  :func:`eviground.eval.testsets.file_items`);
- ``sidset_tampered``: the manifest file; mask = grey value > 127 (decided 18 Nov 2026);
- ``brgen``: the forged file extracted from the BR-Gen archive; mask = grey > 127. Each file must
  match the SHA-256 recorded by notebook 12b (:func:`check_brgen`);
- ``real``: a manifest file (GenImage, SID-Set) or, for Chameleon, the unpacked zip member;
- ``generated``: the manifest file.

Real and generated images carry no mask. An edit mask whose size differs from its image is resized
to the image by :func:`align_mask` (Dhanush's notebook 14 rule, 24 Nov 2026). A mask whose aspect
ratio differs from its image's by more than 2% stops the run, except in SID-Set: there such a mask
is resized as well and flagged ``meta['mask_aspect_mismatch']``, and the mask-based metrics (EPG,
HR-GT, CC) leave it out (:func:`mask_metric_items`; pre-registered 24 Nov 2026, before CP6). The
results with them included are reported as a sensitivity check.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eviground.eval import testsets
from eviground.eval.tiled import Item

EDITED_PARTS = ("diffseg_heldout", "sidset_tampered", "brgen")
MASK_CUT = 127
ASPECT_TOLERANCE = 0.02
# Parts whose aspect-mismatched masks are kept (flagged) instead of stopping the run: notebook 15
# found 26 of 400 SID-Set tampered masks so (24 Nov 2026), and none in the other parts.
FLAG_ASPECT_PARTS = ("sidset_tampered",)


class P6DataError(RuntimeError):
    """A P6 file is missing or differs from what notebook 12b froze."""


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def binary_mask(path: Path, cut: int = MASK_CUT) -> np.ndarray:
    """A grey-level mask file as booleans (value > ``cut``)."""
    with Image.open(path) as m:
        return np.asarray(m.convert("L")) > cut


def aspect_mismatch(
    mask_shape: Sequence[int], image_shape: Sequence[int], tolerance: float = ASPECT_TOLERANCE
) -> bool:
    """True when the mask's width/height ratio differs from the image's by more than
    ``tolerance`` (relative)."""
    h, w = int(mask_shape[0]), int(mask_shape[1])
    H, W = int(image_shape[0]), int(image_shape[1])
    return abs(w / h - W / H) / (W / H) > tolerance


def align_mask(
    mask: np.ndarray, image_shape: Sequence[int], *, tolerance: float = ASPECT_TOLERANCE
) -> tuple[np.ndarray, list[int] | None]:
    """``mask`` at the image's (H, W): resized with ``torch.nn.functional.interpolate``
    (``mode='nearest'``) and thresholded at > 0, the rule of Dhanush's notebook 14 patch. Returns
    the mask and its original [h, w] (None when the sizes already match). A mask whose aspect
    ratio differs from the image's by more than ``tolerance`` (relative) raises
    :class:`P6DataError`: that is a misaligned mask, not another resolution."""
    mask = np.asarray(mask)
    h, w = mask.shape[:2]
    H, W = int(image_shape[0]), int(image_shape[1])
    if (h, w) == (H, W):
        return mask.astype(bool), None
    if aspect_mismatch((h, w), (H, W), tolerance):
        raise P6DataError(
            f"mask {h}x{w} and image {H}x{W} differ in aspect ratio by more than "
            f"{tolerance:.0%}; stop and ask"
        )
    t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]
    out = F.interpolate(t, size=(H, W), mode="nearest")[0, 0] > 0
    return out.numpy(), [int(h), int(w)]


def _aligned(item: Item) -> Item:
    """The item with its mask (and DiffSeg editor map) at the image size; the original mask size
    is recorded in ``meta['mask_resized_from']``."""
    if item.mask is None:
        return item
    flag = item.meta.get("part") in FLAG_ASPECT_PARTS and aspect_mismatch(
        item.mask.shape, item.image.shape
    )
    mask, orig = align_mask(
        item.mask, item.image.shape[:2], tolerance=np.inf if flag else ASPECT_TOLERANCE
    )
    if orig is None:
        return item
    editors = item.editors
    if editors is not None:
        e = torch.from_numpy(np.ascontiguousarray(editors, dtype=np.float32))[None, None]
        e = F.interpolate(e, size=mask.shape, mode="nearest")[0, 0].numpy()
        editors = np.where(mask, e, 0).astype(np.asarray(item.editors).dtype)
    meta = dict(item.meta)
    meta["mask_resized_from"] = orig
    if flag:
        meta["mask_aspect_mismatch"] = True
    return Item(item.uid, item.image, mask, editors, meta)


def is_aspect_mismatched(meta: Mapping[str, Any], image_shape: Sequence[int]) -> bool:
    """The flag, or (for items cached before it existed, e.g. notebook 15's P6 tar) the same test
    on the recorded original mask size."""
    if meta.get("mask_aspect_mismatch"):
        return True
    orig = meta.get("mask_resized_from")
    return bool(orig) and aspect_mismatch(orig, image_shape)


def mask_metric_items(items: Iterable[Item], *, include_mismatched: bool = False) -> Iterator[Item]:
    """The items that enter mask-based metrics (EPG, HR-GT, CC): edited items with a mask,
    without the aspect-mismatched SID-Set masks unless ``include_mismatched`` (the sensitivity
    check). Pre-registered 24 Nov 2026."""
    for it in items:
        if it.mask is None:
            continue
        if not include_mismatched and is_aspect_mismatched(it.meta, it.image.shape[:2]):
            continue
        yield it


def resize_summary(items_meta: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per part: the number of masks resized to their image and the original sizes seen."""
    out: dict[str, Any] = {}
    for meta in items_meta:
        part = meta.get("part", "?")
        rec = out.setdefault(
            part, {"items": 0, "resized": 0, "aspect_mismatch": 0, "from_sizes": {}}
        )
        rec["items"] += 1
        rec["aspect_mismatch"] += int(bool(meta.get("mask_aspect_mismatch")))
        orig = meta.get("mask_resized_from")
        if orig:
            rec["resized"] += 1
            key = f"{orig[0]}x{orig[1]}"
            rec["from_sizes"][key] = rec["from_sizes"].get(key, 0) + 1
    return out


def check_brgen(files: Mapping[str, Mapping[str, Any]], folder: Path) -> list[str]:
    """Members of ``files`` (``P6_brgen_files['files']``) whose image or mask under ``folder`` is
    missing or has another SHA-256 than recorded."""
    bad = []
    for member, rec in files.items():
        for name, key in ((member, "sha256"), (rec["mask"], "mask_sha256")):
            path = Path(folder) / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != rec[key]:
                bad.append(name)
    return bad


def _meta(part: str, row: Mapping[str, Any] | None, **extra: Any) -> dict[str, Any]:
    meta = {"part": part, "label": "edited" if part in EDITED_PARTS else part}
    if row:
        meta.update(
            {k: row.get(k) for k in ("source", "generator", "base_type") if row.get(k) is not None}
        )
        if part in ("real", "generated"):
            meta["label"] = row.get("label") or part
    meta.update(extra)
    return meta


def _brgen_item(uid: str, brgen_dir: Path | None, brgen_masks: Mapping[str, str] | None) -> Item:
    if brgen_dir is None or brgen_masks is None:
        raise P6DataError("BR-Gen images need brgen_dir and brgen_masks")
    path, mpath = Path(brgen_dir) / uid, Path(brgen_dir) / brgen_masks[uid]
    if not path.is_file() or not mpath.is_file():
        raise P6DataError(f"BR-Gen file missing: {uid}")
    return Item(uid, _rgb(path), binary_mask(mpath), None, _meta("brgen", None, source="brgen"))


def _chameleon_item(uid: str, part: str, chameleon_dir: Path | None) -> Item:
    if part != "real" or chameleon_dir is None:
        raise P6DataError(f"{uid} ({part}) is neither in the manifest nor in Chameleon")
    path = Path(chameleon_dir) / uid
    if not path.is_file():
        raise P6DataError(f"Chameleon file missing: {uid}")
    return Item(uid, _rgb(path), None, None, _meta(part, None, source="chameleon"))


def _manifest_item(uid: str, part: str, row: Mapping[str, Any]) -> Item:
    if part == "diffseg_heldout":
        item = next(testsets.file_items([row]))
        item.meta.update(_meta(part, row))
        return item
    mask = None
    if part == "sidset_tampered":
        if not row.get("mask_path"):
            raise P6DataError(f"SID-Set tampered image without a mask: {uid}")
        mask = binary_mask(Path(row["mask_path"]))
    return Item(uid, _rgb(Path(row["path"])), mask, None, _meta(part, row))


def p6_items(
    uids: Iterable[str],
    part_of: Mapping[str, str],
    rows_by_uid: Mapping[str, Mapping[str, Any]],
    *,
    brgen_dir: Path | None = None,
    brgen_masks: Mapping[str, str] | None = None,
    chameleon_dir: Path | None = None,
) -> Iterator[Item]:
    """Whole images (with their edit masks) of the given P6 uids, in the given order."""
    for uid in uids:
        part, row = part_of[uid], rows_by_uid.get(uid)
        if part == "brgen":
            item = _brgen_item(uid, brgen_dir, brgen_masks)
        elif row is None:
            item = _chameleon_item(uid, part, chameleon_dir)
        else:
            item = _manifest_item(uid, part, row)
        yield _aligned(item)


def manifest_rows(
    p6: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The manifest rows of the P6 uids (BR-Gen and Chameleon are not in the manifest)."""
    keep = set(p6["uids"])
    return {r["uid"]: dict(r) for r in manifest if r["uid"] in keep}


def png_bytes(image: np.ndarray) -> bytes:  # used by tests and the dry run
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# One-tar cache of the decoded P6 images and masks (Dhanush, 24 Nov 2026): later notebooks copy
# the tar instead of re-extracting BR-Gen with 7z.
# --------------------------------------------------------------------------------------


def _name(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()[:24]


def write_cache(items: Iterable[Item], tar_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Every item's decoded image (PNG, lossless) and mask (PNG 0/255) in one tar, with a
    manifest: uid -> member names, SHA-256 of each member, and the item's metadata."""
    tar_path, manifest_path = Path(tar_path), Path(manifest_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    with tarfile.open(tar_path, "w") as tf:
        for it in items:
            rec: dict[str, Any] = {"meta": it.meta}
            files = {"image": png_bytes(it.image)}
            if it.mask is not None:
                files["mask"] = png_bytes(np.asarray(it.mask, dtype=np.uint8) * 255)
            for key, data in files.items():
                name = f"{_name(it.uid)}.{key}.png"
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
                rec[key] = name
                rec[f"{key}_sha256"] = hashlib.sha256(data).hexdigest()
            entries[it.uid] = rec
    manifest = {
        "images": len(entries),
        "tar_sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def read_cache(tar_path: Path, manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    """Unpack the cache to ``out_dir`` and check the tar's and every member's SHA-256 against
    the manifest; any difference stops (:class:`P6DataError`). Returns the manifest."""
    tar_path, out_dir = Path(tar_path), Path(out_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    if hashlib.sha256(tar_path.read_bytes()).hexdigest() != manifest["tar_sha256"]:
        raise P6DataError(f"{tar_path} differs from its manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = {
        rec[k] for rec in manifest["entries"].values() for k in ("image", "mask") if k in rec
    }
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if member.name not in allowed or not member.isfile():
                raise P6DataError(f"unexpected member in {tar_path}: {member.name}")
            (out_dir / member.name).write_bytes(tf.extractfile(member).read())
    for uid, rec in manifest["entries"].items():
        for key in ("image", "mask"):
            if key in rec:
                data = (out_dir / rec[key]).read_bytes()
                if hashlib.sha256(data).hexdigest() != rec[f"{key}_sha256"]:
                    raise P6DataError(f"{uid}: cached {key} differs from the manifest")
    return manifest


def cached_items(uids: Iterable[str], manifest: Mapping[str, Any], folder: Path) -> Iterator[Item]:
    """Items from an unpacked cache, in the given order."""
    for uid in uids:
        rec = manifest["entries"][uid]
        mask = binary_mask(Path(folder) / rec["mask"], 0) if "mask" in rec else None
        yield Item(uid, _rgb(Path(folder) / rec["image"]), mask, None, dict(rec["meta"]))
