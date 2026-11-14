"""Test-time tiles (Research Specification section 8, M1, test-time rule; Implementation Brief D0).

256 x 256 tiles, stride 256, the last tile aligned to the border, at most 3 positions per axis
(at most 9 tiles on a regular grid); an image with a side under 256 is one tile at its own size.
Never padded or resized.

For a side longer than 3 x 256 = 768 pixels, 3 evenly spaced positions cannot cover the whole
axis; :func:`tile_positions` then reports the covered share, and :func:`stitch` fills an
uncovered pixel with the value of the nearest covered pixel along each axis. This is provisional
(open question for Dhanush, 15 Nov 2026): it matters for 1024 x 1024 test images in Milestone 5.
"""

from __future__ import annotations

import math

import numpy as np
import torch

TILE = 256
MAX_PER_AXIS = 3


def tile_positions(length: int, tile: int = TILE, max_per_axis: int = MAX_PER_AXIS) -> list[int]:
    """Start positions of the tiles along one axis."""
    if length <= tile:
        return [0]
    n = math.ceil(length / tile)
    if n <= max_per_axis:
        return [min(i * tile, length - tile) for i in range(n)]
    last = length - tile
    return [round(i * last / (max_per_axis - 1)) for i in range(max_per_axis)]


def tile_boxes(height: int, width: int, tile: int = TILE) -> list[tuple[int, int, int, int]]:
    """(left, top, right, bottom) of every tile; a side under ``tile`` is taken whole."""
    th, tw = min(tile, height), min(tile, width)
    return [
        (x, y, x + tw, y + th)
        for y in tile_positions(height, tile)
        for x in tile_positions(width, tile)
    ]


def coverage(length: int, tile: int = TILE) -> float:
    covered = np.zeros(length, dtype=bool)
    for p in tile_positions(length, tile):
        covered[p : p + min(tile, length)] = True
    return float(covered.mean())


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
