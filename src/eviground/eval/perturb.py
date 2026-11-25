"""P5 robustness perturbations, applied at run time (spec section 10, P5: JPEG 95 to 50, resize
0.5-1.5x, blur sigma 1-3, noise; half-size P5 frozen 24 Nov 2026).

Each perturbation maps an (H, W, 3) uint8 image, and for resizing also its (H, W) bool mask, to a
new image (and mask). Everything is deterministic: the JPEG encoder, PIL's resampling and blur,
and the noise draw seeded per (uid, perturbation, level) by :func:`seed_for`, so a rerun or a
resumed session sees the same pixels. The levels are parameters; ``LEVELS`` holds the
pre-registered ones (spec card, set before Phase C).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

# Pre-registered levels (proposed 25 Nov 2026; the spec card decides): three per perturbation, the
# spec's end points included, plus the clean image (level None).
LEVELS: dict[str, tuple[float, ...]] = {
    "jpeg": (95, 75, 50),  # quality, 4:2:0 subsampling
    "resize": (0.5, 0.75, 1.5),  # scale factor; bicubic for images, nearest for masks
    "blur": (1.0, 2.0, 3.0),  # Gaussian sigma in pixels
    "noise": (5.0, 10.0, 20.0),  # additive Gaussian sigma on the 0-255 scale
}
KINDS = tuple(LEVELS)


def seed_for(uid: str, kind: str, level: float, base: int = 42) -> int:
    digest = hashlib.sha256(f"{base}:{uid}:{kind}:{level}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def jpeg(image: np.ndarray, quality: float) -> np.ndarray:
    """Re-encode once as JPEG at ``quality`` with 4:2:0 chroma subsampling, then decode."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        buf, "JPEG", quality=int(quality), subsampling=2
    )
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def resize(
    image: np.ndarray, scale: float, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Scale both sides by ``scale`` (rounded, at least 1 pixel): bicubic for the image, nearest
    for the mask, so the mask stays aligned with the image."""
    h, w = image.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    out = np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(size, Image.BICUBIC))
    if mask is None:
        return out, None
    m = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(size, Image.NEAREST)
    return out, np.asarray(m) > 127


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with standard deviation ``sigma`` pixels (PIL's radius is the sigma)."""
    im = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=float(sigma))))


def noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Additive Gaussian noise (``sigma`` on the 0-255 scale), rounded and clipped."""
    rng = np.random.default_rng(seed)
    x = np.asarray(image, dtype=np.float32) + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def apply(
    image: np.ndarray,
    kind: str | None,
    level: float | None,
    *,
    uid: str = "",
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """One perturbation; ``kind=None`` returns the clean image. Only resizing changes the mask."""
    if kind is None:
        return image, mask
    if kind == "jpeg":
        return jpeg(image, level), mask
    if kind == "resize":
        return resize(image, level, mask)
    if kind == "blur":
        return blur(image, level), mask
    if kind == "noise":
        return noise(image, level, seed_for(uid, kind, level)), mask
    raise ValueError(f"unknown perturbation {kind!r}")


def conditions(levels: Mapping[str, Sequence[float]] = LEVELS) -> list[tuple[str | None, Any]]:
    """The clean condition first, then every (kind, level) in a fixed order."""
    return [(None, None)] + [(k, lv) for k in levels for lv in levels[k]]


def condition_name(kind: str | None, level: Any) -> str:
    return "clean" if kind is None else f"{kind}_{level:g}"


def iter_conditions(
    levels: Mapping[str, Sequence[float]] = LEVELS,
) -> Iterator[tuple[str, str | None, Any]]:
    for kind, level in conditions(levels):
        yield condition_name(kind, level), kind, level
