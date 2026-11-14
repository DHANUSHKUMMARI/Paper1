"""Splits (Research Specification section 9) and the DiffSeg30k editor hold-out (section 10).

- Validation: a fixed, stratified 10% held out from each training source (split seed 42).
- DiffSeg30k mask values (dataset card, read 13 Nov 2026): 0 background, 1 SD 2 inpainting,
  2 Kolors, 3 SD 3.5 medium, 4 FLUX, 5 SDXL inpainting 0.1, 6 Glide, 7 HunyuanDiT,
  8 Kandinsky 2.2 inpaint. An AI-generated base carries its generator's value on every pixel
  (notebook 03); :mod:`eviground.data.diffseg_mask` reads it as 0. Glide (6) and Kolors (2) are
  held out at image level, strictly: training images have no Glide or Kolors pixel at all (base
  or edit); test images (P2) are validation images where Glide or Kolors made an edit.
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


def diffseg_role(
    dataset_split: str,
    values: set[int],
    edit_values: set[int] | None = None,
    reading: str | None = None,
) -> str:
    """Where a DiffSeg30k image goes (strict hold-out, decision of 14 Nov 2026).

    ``values``: every non-zero mask value, the base of an AI-generated image included;
    ``edit_values``: the values left once that base is read as 0 (default: ``values``);
    ``reading``: :func:`eviground.data.diffseg_mask.read_base`'s ``base_reading``.

    - any Glide (6) or Kolors (2) pixel, base or edit: train split -> 'excluded_holdout';
      validation split -> 'test_P2' if Glide or Kolors made an edit, else
      'excluded_holdout_base_only';
    - train split otherwise: an empty mask -> 'real_pool' (an unedited real photo, a real
      negative), a base that border and pixel count disagree on -> 'excluded_base_unclear', an AI
      base with no edit left -> 'excluded_no_edit', else 'train_pool';
    - validation split otherwise: 'unused_validation';
    - a value outside 1-8: 'excluded_unknown_value'.
    """
    edits = set(values) if edit_values is None else set(edit_values)
    if set(values) - set(DIFFSEG_EDITORS):
        return "excluded_unknown_value"
    if set(values) & HOLDOUT_VALUES:
        if dataset_split != "validation":
            return "excluded_holdout"
        return "test_P2" if edits & HOLDOUT_VALUES else "excluded_holdout_base_only"
    if dataset_split == "validation":
        return "unused_validation"
    if not values:
        return "real_pool"
    if reading == "disagree":
        return "excluded_base_unclear"
    if reading == "no_edit" or not edits:
        return "excluded_no_edit"
    return "train_pool"


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
