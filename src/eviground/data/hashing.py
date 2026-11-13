"""Image hashes for the CP1 leakage checks.

- ``sha256`` of the file bytes and of the decoded pixels (exact duplicates: the CP1 criterion
  is 0 shared hashes across train / validation / test);
- a 64-bit DCT perceptual hash (near-duplicates, compared by Hamming distance).

Decision of 12 Nov 2026: exact duplicates across splits must be 0; within-split duplicates are
reported only; near-duplicates of a test image in train are counted, and more than 1% of a test
set means stop and ask.
"""

from __future__ import annotations

import hashlib
import io
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

PHASH_SIZE = 32  # the image is reduced to 32x32 grey before the DCT
PHASH_LOW = 8  # the 8x8 lowest frequencies give 64 bits
NEAR_DUP_THRESHOLDS = (0, 2, 4, 6, 8, 10)
NEAR_DUP_PRIMARY = 6  # Hamming distance used for the >1% rule and the COCO negative filter


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_rgb_array(img: Image.Image) -> np.ndarray:
    """Decoded pixels as HxWx3 uint8 (greyscale, palette, CMYK and alpha images converted)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def pixel_sha256(rgb: np.ndarray) -> str:
    """Hash of the decoded pixels and their shape (same picture saved twice -> same hash)."""
    h = hashlib.sha256(f"{rgb.shape}".encode())
    h.update(np.ascontiguousarray(rgb).tobytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def _dct_matrix(n: int = PHASH_SIZE) -> np.ndarray:
    k = np.arange(n)
    mat = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    mat[0] *= 1 / np.sqrt(2)
    return mat * np.sqrt(2 / n)


def phash(img: Image.Image) -> int:
    """64-bit perceptual hash: grey 32x32, 2-D DCT, 8x8 low frequencies above their median."""
    grey = img.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(grey, dtype=np.float64)
    d = _dct_matrix()
    low = (d @ arr @ d.T)[:PHASH_LOW, :PHASH_LOW].ravel()
    bits = low > np.median(low)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def image_hashes(data: bytes) -> dict[str, Any]:
    """All hashes of one image file: file sha256, pixel sha256, pHash, width, height."""
    out: dict[str, Any] = {"sha256": file_sha256(data)}
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            rgb = to_rgb_array(img)
            out.update(
                pixel_sha256=pixel_sha256(rgb),
                phash=phash(Image.fromarray(rgb)),
                width=rgb.shape[1],
                height=rgb.shape[0],
            )
    except Exception as exc:  # report, never crash a scan
        out["error"] = f"decode: {type(exc).__name__}"
    return out


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits of each uint64."""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.uint8)
    return _POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1, dtype=np.uint8)


def as_uint64(hashes: Any) -> np.ndarray:
    """pHash values (Python ints, possibly stored as signed int64) as a uint64 array."""
    return np.array([int(h) & 0xFFFFFFFFFFFFFFFF for h in hashes], dtype=np.uint64)


def min_hamming(
    queries: np.ndarray, reference: np.ndarray, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """For each query hash, the smallest Hamming distance to any reference hash and its index.

    Brute force in chunks (memory ``chunk x len(reference)`` uint64). Empty reference gives
    distance 65 and index -1.
    """
    q = np.asarray(queries, dtype=np.uint64)
    r = np.asarray(reference, dtype=np.uint64)
    dist = np.full(q.size, 65, dtype=np.int16)
    idx = np.full(q.size, -1, dtype=np.int64)
    if r.size == 0 or q.size == 0:
        return dist, idx
    for start in range(0, q.size, chunk):
        block = popcount64(q[start : start + chunk, None] ^ r[None, :])
        arg = block.argmin(axis=1)
        dist[start : start + chunk] = block[np.arange(block.shape[0]), arg]
        idx[start : start + chunk] = arg
    return dist, idx


def near_dup_counts(dist: np.ndarray, thresholds: tuple[int, ...] = NEAR_DUP_THRESHOLDS) -> dict:
    """How many queries have a reference within each Hamming threshold."""
    return {f"<={t}": int((dist <= t).sum()) for t in thresholds}
