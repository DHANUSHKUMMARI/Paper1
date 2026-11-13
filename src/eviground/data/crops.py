"""Training crops (Research Specification section 8, M1).

A random 256x256 crop at native resolution: never padded or resized. Images with a side under
256 are skipped and counted. For edited images the crop must contain both edited and unedited
pixels: edited share between 1% and 90%, up to 10 tries.
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


def edited_box(
    mask: np.ndarray,
    rng: np.random.Generator,
    size: int = CROP,
    share: tuple[float, float] = EDITED_SHARE,
    tries: int = TRIES,
) -> tuple[tuple[int, int, int, int] | None, float | None, int]:
    """A crop whose edited share (``mask > 0``) lies in ``share``.

    Returns ``(box, edited_share, tries_used)``; ``box`` is None when no try succeeded (the
    caller draws another image and counts it) or the image is too small.
    """
    edited = np.asarray(mask) > 0
    h, w = edited.shape[:2]
    if w < size or h < size:
        return None, None, 0
    # integral image: share of any box in O(1)
    integral = np.pad(edited.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
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
