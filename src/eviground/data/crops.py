"""Training crops (Research Specification section 8, M1).

A random 256x256 crop at native resolution: never padded or resized. Images with a side under
256 are skipped and counted. For edited images the crop must contain both edited and unedited
pixels: edited share between 1% and 90%, up to 10 tries.

Two rules for edited images:

- :func:`edited_box` (CP1 runs 1-4): a uniform random window; kept unchanged so the recorded CP1
  runs stay reproducible.
- :func:`edited_crop` (training loader, decision of 14 Nov 2026): a random edited pixel, then a
  window drawn uniformly among all in-image windows that contain it. The edited pixel lands
  anywhere in the crop, never preferentially at its centre (test tiles have no centre prior).
"""

from __future__ import annotations

import numpy as np

CROP = 256
EDITED_SHARE = (0.01, 0.90)
TRIES = 10


def random_box(width: int, height: int, rng: np.random.Generator, size: int = CROP):
    """(left, top, right, bottom) of a uniform random crop, or None if the image is too small."""
    if width < size or height < size:
        return None
    left = int(rng.integers(0, width - size + 1))
    top = int(rng.integers(0, height - size + 1))
    return left, top, left + size, top + size


def _integral(edited: np.ndarray) -> np.ndarray:
    """Integral image with a zero first row and column: the sum of any box in O(1)."""
    return np.pad(edited.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _box_sum(integral: np.ndarray, left: int, top: int, size: int) -> int:
    right, bottom = left + size, top + size
    inner = integral[bottom, right] - integral[top, right]
    return int(inner - integral[bottom, left] + integral[top, left])


def edited_box(
    mask: np.ndarray,
    rng: np.random.Generator,
    size: int = CROP,
    share: tuple[float, float] = EDITED_SHARE,
    tries: int = TRIES,
) -> tuple[tuple[int, int, int, int] | None, float | None, int]:
    """A crop whose edited share (``mask > 0``) lies in ``share`` (CP1 rule: uniform window).

    Returns ``(box, edited_share, tries_used)``; ``box`` is None when no try succeeded (the
    caller draws another image and counts it) or the image is too small.
    """
    edited = np.asarray(mask) > 0
    h, w = edited.shape[:2]
    if w < size or h < size:
        return None, None, 0
    integral = _integral(edited)
    lo, hi = share
    last = None
    for k in range(1, tries + 1):
        left, top, right, bottom = random_box(w, h, rng, size)  # type: ignore[misc]
        s = (
            integral[bottom, right]
            - integral[top, right]
            - integral[bottom, left]
            + integral[top, left]
        ) / (size * size)
        last = float(s)
        if lo <= s <= hi:
            return (left, top, right, bottom), last, k
    return None, last, tries


def _accept(n_edited: int, size: int, share: tuple[float, float], min_pixels: int | None) -> bool:
    """Edited share within ``share``; with ``min_pixels`` the lower bound is a pixel count."""
    s = n_edited / (size * size)
    lo, hi = share
    if s > hi:
        return False
    if min_pixels is not None:
        return n_edited >= min_pixels
    return s >= lo


def edited_crop(
    mask: np.ndarray,
    rng: np.random.Generator,
    size: int = CROP,
    share: tuple[float, float] = EDITED_SHARE,
    tries: int = TRIES,
    min_pixels: int | None = None,
) -> tuple[tuple[int, int, int, int] | None, float | None, int]:
    """Training crop for an edited image (decision of 14 Nov 2026).

    Each try picks a uniform random edited pixel, then a window uniformly among all in-image
    ``size`` x ``size`` windows that contain it, so the pixel's position inside the crop is
    uniform, not centred. The window is kept if its edited share lies in ``share`` (or, with
    ``min_pixels``, if it holds at least that many edited pixels and at most ``share[1]``).
    Returns ``(box, edited_share, tries_used)``; ``box`` is None after ``tries`` failures, for an
    image smaller than ``size`` or for an empty mask (the loader then replaces the image).
    """
    edited = np.asarray(mask) > 0
    h, w = edited.shape[:2]
    if w < size or h < size:
        return None, None, 0
    flat = np.flatnonzero(edited)
    if flat.size == 0:
        return None, 0.0, 0
    last = None
    for k in range(1, tries + 1):
        y, x = divmod(int(flat[int(rng.integers(0, flat.size))]), w)
        left = int(rng.integers(max(0, x - size + 1), min(x, w - size) + 1))
        top = int(rng.integers(max(0, y - size + 1), min(y, h - size) + 1))
        # one window: summing it is cheaper than an integral image of the whole mask
        n = int(np.count_nonzero(edited[top : top + size, left : left + size]))
        last = n / (size * size)
        if _accept(n, size, share, min_pixels):
            return (left, top, left + size, top + size), last, k
    return None, last, tries


def window_counts(mask: np.ndarray, size: int = CROP) -> np.ndarray:
    """Edited-pixel count of every in-image ``size`` window (rows: top, columns: left)."""
    edited = np.asarray(mask) > 0
    h, w = edited.shape[:2]
    if w < size or h < size:
        return np.zeros((0, 0), dtype=np.int64)
    ii = _integral(edited)
    return ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]


def has_valid_window(
    mask: np.ndarray,
    size: int = CROP,
    share: tuple[float, float] = EDITED_SHARE,
    min_pixels: int | None = None,
) -> bool:
    """Whether any in-image window passes the rule of :func:`edited_crop` (exhaustive)."""
    counts = window_counts(mask, size)
    if counts.size == 0:
        return False
    hi = share[1] * size * size
    lo = min_pixels if min_pixels is not None else share[0] * size * size
    return bool(np.any((counts >= lo) & (counts <= hi)))
