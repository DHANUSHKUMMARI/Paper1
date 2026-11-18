"""Chameleon (P4 hard real-world test; Yan et al., ICLR 2025), from the authors' archive.

Dhanush, 18 Nov 2026: the authors' full ``Chameleon.zip`` (2.68 GB) is a Drive shortcut in
``datasets/05_Chameleon``. On a CPU runtime the zip is checked (size, CRC of every member, about
26,000 images), unpacked in full to the local disk (never to Drive), and every image is opened.
The P4 subset is drawn from the unpacked list and frozen before any scoring (``eval.subsets``).

Class labels come from the path: a component ``0_real`` / ``real`` is real, ``1_fake`` / ``fake``
is AI-generated (the layout of the AIDE release). A member with neither label is reported, not
guessed.
"""

from __future__ import annotations

import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL = {"0_real", "real"}
FAKE = {"1_fake", "fake"}
ZIP_NAME = "Chameleon.zip"


class ChameleonError(RuntimeError):
    """The archive is missing, unreadable or not what it should be: stop and ask Dhanush."""


def label_of(name: str) -> str | None:
    """'real', 'generated' or None from a member path."""
    parts = {p.lower() for p in PurePosixPath(name).parts[:-1]}
    if parts & REAL and not parts & FAKE:
        return "real"
    if parts & FAKE and not parts & REAL:
        return "generated"
    return None


def is_image(name: str) -> bool:
    p = PurePosixPath(name)
    return (
        p.suffix.lower() in IMAGE_EXT and not p.name.startswith(".") and "__MACOSX" not in p.parts
    )


def inspect_zip(path: Path, *, expect_images: int = 26000, tolerance: float = 0.10) -> dict:
    """Size, CRC check of every member (``testzip``), image count per class. Raises
    :class:`ChameleonError` on a missing or corrupt archive, or an image count more than
    ``tolerance`` away from ``expect_images``."""
    path = Path(path)
    if not path.is_file():
        raise ChameleonError(f"{path} is missing (is the Drive shortcut in 05_Chameleon?)")
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            names = [i.filename for i in zf.infolist() if not i.is_dir()]
    except zipfile.BadZipFile as e:
        raise ChameleonError(f"{path.name} is not a readable zip: {e}") from e
    if bad is not None:
        raise ChameleonError(f"{path.name}: CRC error in member {bad}")
    images = [n for n in names if is_image(n)]
    labels = Counter(label_of(n) or "unlabelled" for n in images)
    report = {
        "zip": path.name,
        "bytes": path.stat().st_size,
        "members": len(names),
        "images": len(images),
        "per_class": dict(sorted(labels.items())),
        "other_members": len(names) - len(images),
    }
    lo, hi = expect_images * (1 - tolerance), expect_images * (1 + tolerance)
    if not lo <= len(images) <= hi:
        raise ChameleonError(f"{len(images)} images, expected about {expect_images}: {report}")
    return report


def unpack(path: Path, dest: Path) -> list[str]:
    """Extract the whole archive under ``dest`` (local disk); returns the image member names."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)
        return sorted(i.filename for i in zf.infolist() if not i.is_dir() and is_image(i.filename))


def unreadable(root: Path, names: Iterable[str]) -> list[str]:
    """Members that PIL cannot open and decode (``verify`` + ``load``)."""
    from PIL import Image

    bad = []
    for n in names:
        try:
            with Image.open(Path(root) / n) as im:
                im.verify()
            with Image.open(Path(root) / n) as im:
                im.load()
        except Exception:  # noqa: BLE001 - any decode failure counts
            bad.append(n)
    return bad


def items(names: Iterable[str]) -> list[dict[str, Any]]:
    """One row per labelled image: ``uid`` (the member path) and ``label``."""
    return [{"uid": n, "label": lab} for n in names if (lab := label_of(n))]
