"""Test-time tiles (Research Specification section 8, M1, test-time rule; Implementation Brief D0).

256 x 256 tiles, stride 256, the last tile aligned to the border, so every pixel is covered (full
coverage, decision of 16 Nov 2026: a 1024 x 1024 image takes 16 tiles); an image with a side
under 256 is one tile at its own size. Never padded or resized. Where the border-aligned last tile
overlaps the one before, the stitched map is the mean of the overlapping tile probabilities.

``max_per_axis`` keeps the earlier rule (at most 3 positions per axis, 9 tiles) available for
comparison; it no longer covers sides longer than 768 pixels, and :func:`stitch` then fills an
uncovered pixel with the value of the nearest covered pixel along each axis.
"""

from __future__ import annotations

import math

import numpy as np
import torch

TILE = 256
MAX_PER_AXIS = None  # full coverage (16 Nov 2026); 3 was the earlier rule
TILE_BATCH = 32  # tiles per forward pass at test time (fixed-size batches)


def tile_positions(
    length: int, tile: int = TILE, max_per_axis: int | None = MAX_PER_AXIS
) -> list[int]:
    """Start positions of the tiles along one axis: stride ``tile``, the last one aligned to the
    border. With ``max_per_axis``, a longer side gets that many evenly spaced positions instead."""
    if length <= tile:
        return [0]
    n = math.ceil(length / tile)
    if max_per_axis is None or n <= max_per_axis:
        return [min(i * tile, length - tile) for i in range(n)]
    last = length - tile
    return [round(i * last / (max_per_axis - 1)) for i in range(max_per_axis)]


def tile_boxes(
    height: int, width: int, tile: int = TILE, max_per_axis: int | None = MAX_PER_AXIS
) -> list[tuple[int, int, int, int]]:
    """(left, top, right, bottom) of every tile; a side under ``tile`` is taken whole."""
    th, tw = min(tile, height), min(tile, width)
    return [
        (x, y, x + tw, y + th)
        for y in tile_positions(height, tile, max_per_axis)
        for x in tile_positions(width, tile, max_per_axis)
    ]


def coverage(length: int, tile: int = TILE, max_per_axis: int | None = MAX_PER_AXIS) -> float:
    covered = np.zeros(length, dtype=bool)
    for p in tile_positions(length, tile, max_per_axis):
        covered[p : p + min(tile, length)] = True
    return float(covered.mean())


def tile_count(height: int, width: int, tile: int = TILE) -> int:
    """Number of tiles of an image under the full-coverage rule."""
    return len(tile_positions(height, tile)) * len(tile_positions(width, tile))


def _nearest_index(covered: np.ndarray) -> np.ndarray:
    """For every position, the nearest covered position (itself if covered)."""
    idx = np.flatnonzero(covered)
    pos = np.arange(covered.size)
    j = np.clip(np.searchsorted(idx, pos), 1, max(1, idx.size - 1))
    left, right = idx[np.maximum(j - 1, 0)], idx[np.minimum(j, idx.size - 1)]
    return np.where(np.abs(pos - left) <= np.abs(right - pos), left, right)


def stitch(
    tiles: torch.Tensor, boxes: list[tuple[int, int, int, int]], height: int, width: int
) -> torch.Tensor:
    """Mean of overlapping tile maps (each (h, w)) into a (height, width) map; uncovered pixels
    take the nearest covered value."""
    total = torch.zeros(height, width, dtype=tiles.dtype, device=tiles.device)
    count = torch.zeros(height, width, dtype=tiles.dtype, device=tiles.device)
    for t, (left, top, right, bottom) in zip(tiles, boxes, strict=True):
        total[top:bottom, left:right] += t
        count[top:bottom, left:right] += 1
    covered = (count > 0).cpu().numpy()
    out = total / count.clamp(min=1)
    if covered.all():
        return out
    rows = _nearest_index(covered.any(axis=1))
    cols = _nearest_index(covered.any(axis=0))
    return out[torch.as_tensor(rows, device=out.device)][
        :, torch.as_tensor(cols, device=out.device)
    ]
