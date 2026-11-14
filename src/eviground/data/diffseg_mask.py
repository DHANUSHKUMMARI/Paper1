"""How a DiffSeg30k mask is read (decision of 14 Nov 2026, after notebook 03).

The release labels every pixel of an AI-generated base image with the value of the model that
generated it; edits on top carry their editor's value. A real (COCO) base keeps 0 as background.
Notebook 03 measured this: 15,687 of 34,999 masks have no 0 (12,050 at 1024 x 1024).

Reading rule (Dhanush, 14 Nov 2026):

- a mask with any 0 pixel: real base; the mask is used as it is (``base_type`` "real");
- a mask with no 0 pixel (AI base): the base is the value covering most of the image border, and
  it must also be the value with the most pixels ("agree"); it is read as 0. If the two differ the
  image leaves training ("disagree", counted). If no value is left once the base is removed, the
  image has no edit to learn ("no_edit") and is dropped, never relabelled;
- an empty mask (all 0): an unedited real photo ("unedited"), a real negative.

:func:`edit_mask` gives the mask target and what the crop rule uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def border_counts(mask: np.ndarray) -> np.ndarray:
    """Pixel count of every value 0-255 on the one-pixel image border (corners once)."""
    m = np.asarray(mask)
    if m.shape[0] < 2 or m.shape[1] < 2:
        return np.bincount(m.ravel(), minlength=256)
    border = np.concatenate([m[0, :], m[-1, :], m[1:-1, 0], m[1:-1, -1]])
    return np.bincount(border, minlength=256)


def read_base(mask: np.ndarray) -> dict[str, Any]:
    """Base type, base value and edit values of one mask (see the module docstring)."""
    m = np.asarray(mask)
    counts = np.bincount(m.ravel(), minlength=256)
    values = [int(v) for v in np.flatnonzero(counts) if v != 0]
    if not values:
        return _out("real", "unedited", None, [], None, None)
    if counts[0] > 0:
        return _out("real", "real", None, values, None, None)
    dominant = int(np.argmax(counts))
    border = int(np.argmax(border_counts(m)))
    if dominant != border:
        return _out("ai", "disagree", None, values, dominant, border)
    edits = [v for v in values if v != dominant]
    return _out("ai", "agree" if edits else "no_edit", dominant, edits, dominant, border)


def _out(
    base_type: str,
    reading: str,
    base: int | None,
    edits: list[int],
    dominant: int | None,
    border: int | None,
) -> dict[str, Any]:
    return {
        "base_type": base_type,
        "base_reading": reading,
        "mask_base": base,
        "edit_values": ",".join(str(v) for v in sorted(edits)),
        "dominant_value": dominant,
        "border_value": border,
    }


def edit_mask(mask: np.ndarray, base: int | None) -> np.ndarray:
    """The mask with an AI base's value read as 0 (unchanged for a real base)."""
    m = np.asarray(mask)
    if base is None:
        return m
    out = m.copy()
    out[m == base] = 0
    return out
