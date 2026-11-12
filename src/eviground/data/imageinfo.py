"""File-level image facts used by the data probe and the CP1 checks.

Everything here reads files; nothing writes or re-encodes them.

- :func:`sniff_format` identifies the real format from the first bytes (not the extension).
- :func:`jpeg_quality` estimates the IJG quality factor from the quantization tables.
- :func:`chroma_subsampling` reports 4:4:4, 4:2:2 or 4:2:0.
- :func:`header_info` collects these plus size and metadata flags without decoding pixels.
- :func:`blockiness` measures pixel-level 8x8 block artefacts (compression history).
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, JpegImagePlugin

_MAGIC: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "png"),
    (b"\xff\xd8\xff", 0, "jpeg"),
    (b"GIF8", 0, "gif"),
    (b"BM", 0, "bmp"),
    (b"II*\x00", 0, "tiff"),
    (b"MM\x00*", 0, "tiff"),
)

LOSSLESS_FORMATS = frozenset({"png", "bmp", "tiff", "gif"})
_SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def sniff_format(head: bytes) -> str:
    """Format from the leading bytes: png, jpeg, webp, gif, bmp, tiff or unknown."""
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    for magic, offset, name in _MAGIC:
        if head[offset : offset + len(magic)] == magic:
            return name
    return "unknown"


@lru_cache(maxsize=1)
def _reference_tables() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Luma and chroma quantization tables of libjpeg at qualities 1-100.

    Built by encoding a tiny image with Pillow at each quality, so the tables are in
    exactly the order and scaling Pillow reports when it reads a file.
    """
    img = Image.new("RGB", (8, 8), (128, 128, 128))
    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for q in range(1, 101):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        qt = Image.open(buf).quantization
        tables[q] = (np.asarray(qt[0], dtype=np.int64), np.asarray(qt[1], dtype=np.int64))
    return tables


def jpeg_quality(img: Image.Image) -> tuple[int, bool] | None:
    """Estimate the IJG quality factor of an opened JPEG.

    Returns ``(quality, exact)``: ``exact`` is True when the tables are the standard libjpeg
    tables for that quality (files written by libjpeg, Pillow, OpenCV); otherwise ``quality``
    is the closest standard quality (cameras and editors often use their own tables).
    Returns None for non-JPEG images or files without quantization tables.
    """
    qt = getattr(img, "quantization", None)
    if not qt or 0 not in qt:
        return None
    luma = np.asarray(qt[0], dtype=np.int64)
    chroma = np.asarray(qt[1], dtype=np.int64) if 1 in qt else None
    if luma.size != 64:
        return None
    best_q, best_d = 0, None
    for q, (ref_l, ref_c) in _reference_tables().items():
        d = int(np.abs(luma - ref_l).sum())
        if chroma is not None and chroma.size == 64:
            d += int(np.abs(chroma - ref_c).sum())
        if best_d is None or d < best_d or (d == best_d and q > best_q):
            best_q, best_d = q, d
    return best_q, best_d == 0


def chroma_subsampling(img: Image.Image) -> str | None:
    """'4:4:4', '4:2:2', '4:2:0' or 'other' for a JPEG; None for other formats or greyscale."""
    if img.format != "JPEG" or img.mode not in {"RGB", "YCbCr", "CMYK"}:
        return None
    return _SUBSAMPLING.get(JpegImagePlugin.get_sampling(img), "other")


def header_info(source: Path | str | bytes) -> dict[str, Any]:
    """Format, size and compression facts of one image file, without decoding the pixels.

    ``source`` is a path or the file's bytes. For a path only the header is read (Pillow
    opens lazily), which matters on Google Drive. Unreadable files give ``{"error": ...}``.
    """
    try:
        if isinstance(source, bytes):
            n_bytes, head, handle = len(source), source[:16], io.BytesIO(source)
        else:
            path = Path(source)
            n_bytes = path.stat().st_size
            with open(path, "rb") as fh:
                head = fh.read(16)
            handle = path
    except OSError as exc:
        return {"error": f"read: {type(exc).__name__}"}
    info: dict[str, Any] = {"bytes": n_bytes, "format": sniff_format(head)}
    try:
        with Image.open(handle) as img:
            info.update(width=img.width, height=img.height, mode=img.mode)
            info["has_exif"] = bool(img.info.get("exif"))
            info["has_icc"] = bool(img.info.get("icc_profile"))
            q = jpeg_quality(img) if img.format == "JPEG" else None
            info["jpeg_quality"] = q[0] if q else None
            info["jpeg_quality_exact"] = q[1] if q else None
            info["subsampling"] = chroma_subsampling(img)
            info["progressive"] = bool(img.info.get("progressive")) if q else None
    except Exception as exc:  # corrupt or unsupported file: report, never crash a scan
        info["error"] = f"decode: {type(exc).__name__}"
    return info


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma as float64 from an HxW or HxWxC uint8 array."""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _axis_ratios(gray: np.ndarray, axis: int, block: int) -> np.ndarray:
    """Mean |difference| across each of the ``block`` grid phases, divided by the others.

    Entry ``p`` compares differences between pixel ``j`` and ``j+1`` where ``(j+1) % block
    == p`` (a block boundary for a grid starting at phase ``p``) with all other positions.
    """
    diffs = np.abs(np.diff(gray, axis=axis))
    profile = diffs.mean(axis=1 - axis)  # one mean per column (axis=1) or row (axis=0)
    phase = (np.arange(profile.size) + 1) % block
    ratios = np.ones(block)
    for p in range(block):
        on, off = profile[phase == p], profile[phase != p]
        if on.size and off.size and off.mean() > 0:
            ratios[p] = on.mean() / off.mean()
    return ratios


def blockiness(img: np.ndarray, block: int = 8, phase: tuple[int, int] | None = (0, 0)) -> float:
    """Pixel-level 8x8 blockiness: boundary gradient over within-block gradient.

    About 1.0 for an image without block artefacts; above 1 when JPEG block edges are
    visible. ``phase=(row, col)`` is where the block grid starts ((0, 0) for an uncropped
    image); ``phase=None`` takes the strongest phase on each axis, for crops whose grid
    offset is unknown.
    """
    gray = to_gray(img)
    if min(gray.shape[:2]) < 2 * block:
        raise ValueError(f"image too small for blockiness: {gray.shape[:2]}")
    rows = _axis_ratios(gray, axis=0, block=block)
    cols = _axis_ratios(gray, axis=1, block=block)
    if phase is None:
        return float((rows.max() + cols.max()) / 2)
    return float((rows[phase[0] % block] + cols[phase[1] % block]) / 2)
