"""Splits (Research Specification section 9) and the DiffSeg30k editor hold-out (section 10).

- Validation: a fixed, stratified 10% held out from each training source (split seed 42).
- DiffSeg30k mask values (dataset card, read 13 Nov 2026): 0 background, 1 SD 2 inpainting,
  2 Kolors, 3 SD 3.5 medium, 4 FLUX, 5 SDXL inpainting 0.1, 6 Glide, 7 HunyuanDiT,
  8 Kandinsky 2.2 inpaint. Glide (6) and Kolors (2) are held out at image level: training
  images are those whose every edit comes from the 6 training editors; test images (P2) are
  validation images with at least one Glide or Kolors pixel.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

SPLIT_SEED = 42
VAL_FRACTION = 0.10

DIFFSEG_EDITORS = {
    1: "stabilityai/stable-diffusion-2-inpainting",
    2: "kolors",
    3: "stabilityai/stable-diffusion-3.5-medium",
    4: "flux",
    5: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    6: "glide",
    7: "Tencent-Hunyuan/HunyuanDiT-Diffusers",
    8: "kandinsky-community/kandinsky-2-2-decoder-inpaint",
}
HOLDOUT_VALUES = frozenset({2, 6})  # Kolors, Glide
TRAIN_EDITOR_VALUES = frozenset(DIFFSEG_EDITORS) - HOLDOUT_VALUES


def editors_text(values: Iterable[int]) -> str:
    """Sorted non-zero mask values as text, e.g. '1,5'."""
    return ",".join(str(v) for v in sorted({int(v) for v in values} - {0}))


def parse_editors(text: str | float | None) -> set[int]:
    if text is None or (isinstance(text, float) and np.isnan(text)) or text == "":
        return set()
    return {int(v) for v in str(text).split(",")}


def diffseg_role(dataset_split: str, editors: set[int]) -> str:
    """Where a DiffSeg30k image goes.

    train split: 'train_pool' (only training editors), 'excluded_holdout' (any Glide / Kolors
    pixel), 'excluded_no_edit' (empty mask), 'excluded_unknown_value' (value outside 1-8).
    validation split: 'test_P2' (any Glide / Kolors pixel), else 'unused_validation'.
    """
    if editors - set(DIFFSEG_EDITORS):
        return "excluded_unknown_value"
    held = bool(editors & HOLDOUT_VALUES)
    if dataset_split == "validation":
        return "test_P2" if held else "unused_validation"
    if not editors:
        return "excluded_no_edit"
    return "excluded_holdout" if held else "train_pool"


def stratified_val(
    uids: Sequence[str], strata: Sequence[str], frac: float = VAL_FRACTION, seed: int = SPLIT_SEED
) -> list[str]:
    """'train' or 'val' per row: round(frac * n) of each stratum go to validation.

    Deterministic: rows are ordered by uid within each stratum before the seeded permutation,
    so the result does not depend on the order of the input.
    """
    out = ["train"] * len(uids)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(str(s), []).append(i)
    for key in sorted(groups):
        idx = sorted(groups[key], key=lambda i: uids[i])
        n_val = int(round(frac * len(idx)))
        for j in rng.permutation(len(idx))[:n_val]:
            out[idx[j]] = "val"
    return out
