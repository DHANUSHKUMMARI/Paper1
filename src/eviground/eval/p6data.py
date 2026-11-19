"""P6 images for the explanation evaluation (spec section 10, P6; CP5, Dhanush 19 Nov 2026).

:func:`p6_items` yields one :class:`eviground.eval.tiled.Item` per frozen P6 uid, by its part:

- ``diffseg_heldout``: the CP1 manifest file; mask by the training rule (AI base read as 0,
  :func:`eviground.eval.testsets.file_items`);
- ``sidset_tampered``: the manifest file; mask = grey value > 127 (decided 18 Nov 2026);
- ``brgen``: the forged file extracted from the BR-Gen archive; mask = grey > 127. Each file must
  match the SHA-256 recorded by notebook 12b (:func:`check_brgen`);
- ``real``: a manifest file (GenImage, SID-Set) or, for Chameleon, the unpacked zip member;
- ``generated``: the manifest file.

Real and generated images carry no mask.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.eval import testsets
from eviground.eval.tiled import Item

EDITED_PARTS = ("diffseg_heldout", "sidset_tampered", "brgen")
MASK_CUT = 127


class P6DataError(RuntimeError):
    """A P6 file is missing or differs from what notebook 12b froze."""


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def binary_mask(path: Path, cut: int = MASK_CUT) -> np.ndarray:
    """A grey-level mask file as booleans (value > ``cut``)."""
    with Image.open(path) as m:
        return np.asarray(m.convert("L")) > cut


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
            yield _brgen_item(uid, brgen_dir, brgen_masks)
        elif row is None:
            yield _chameleon_item(uid, part, chameleon_dir)
        else:
            yield _manifest_item(uid, part, row)


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
