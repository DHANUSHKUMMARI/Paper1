"""Training draws for one epoch with crop replacement (decision of 14 Nov 2026).

The sampler (:mod:`eviground.data.sampler`) fixes how many crops each (class, source) pool gives.
An edited image whose crop fails the edited-share rule after its tries is replaced by another
image from the same pool, so the 1 : 1 : 1 class balance and the source shares hold exactly.
Failures are counted per image and replacements per pool, for the training log. Images that can
never give a valid crop are found once beforehand (:func:`screen_never_valid`) and left out.

Torch-free: the Milestone 3 data loader turns the plan into tensors.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from eviground.data import crops, sampler

Box = tuple[int, int, int, int]
CropFn = Callable[[int, np.random.Generator], Box | None]


@dataclass
class EpochPlan:
    """Row index and crop box of every draw, in training order, plus the failure log."""

    items: list[tuple[int, Box]] = field(default_factory=list)
    failures: Counter[str] = field(default_factory=Counter)  # uid -> failed draws
    replacements: Counter[str] = field(default_factory=Counter)  # "class/source" -> count

    def per_pool(self, sources: Sequence[str], labels: Sequence[str]) -> dict[str, int]:
        c: Counter[str] = Counter(f"{labels[i]}/{sources[i]}" for i, _ in self.items)
        return dict(sorted(c.items()))

    def log(self) -> dict[str, object]:
        return {
            "draws": len(self.items),
            "replaced": sum(self.replacements.values()),
            "replacements_per_pool": dict(sorted(self.replacements.items())),
            "images_with_failures": len(self.failures),
            "most_failed": self.failures.most_common(10),
        }


def epoch_plan(
    sources: Sequence[str],
    labels: Sequence[str],
    uids: Sequence[str],
    n_per_class: int,
    seed: int,
    crop_fn: CropFn,
    *,
    profile: str = "unified",
    max_replacements: int | None = None,
) -> EpochPlan:
    """Draw ``n_per_class`` crops per class; a failed crop is replaced from the same pool.

    ``crop_fn(row, rng)`` returns the crop box or None. Deterministic for a given ``seed``.
    Raises if more than ``max_replacements`` replacements are needed (default: one per draw),
    which means the pools hold too many images without a valid crop.
    """
    src, lab = np.asarray(sources), np.asarray(labels)
    picks = sampler.draw(sources, labels, n_per_class, seed, profile)
    pools = {
        (label, source): np.flatnonzero((src == source) & (lab == label))
        for (label, source), _n in sampler.quotas(n_per_class, profile).items()
    }
    limit = len(picks) if max_replacements is None else max_replacements
    replace_rng = np.random.default_rng([seed, 1])
    plan = EpochPlan()
    for k, first in enumerate(picks):
        i, attempt = int(first), 0
        box = crop_fn(i, np.random.default_rng([seed, 2, k, attempt]))
        while box is None:
            key = (str(lab[i]), str(src[i]))
            plan.failures[uids[i]] += 1
            plan.replacements[f"{key[0]}/{key[1]}"] += 1
            if sum(plan.replacements.values()) > limit:
                raise RuntimeError(
                    f"more than {limit} crop replacements: too many images without a valid "
                    f"crop (last pool {key[0]}/{key[1]})"
                )
            i = int(replace_rng.choice(pools[key]))
            attempt += 1
            box = crop_fn(i, np.random.default_rng([seed, 2, k, attempt]))
        plan.items.append((i, box))
    return plan


def screen_never_valid(
    uids: Iterable[str],
    load_mask: Callable[[str], np.ndarray],
    size: int = crops.CROP,
    share: tuple[float, float] = crops.EDITED_SHARE,
    min_pixels: int | None = None,
) -> list[str]:
    """Edited images with no window that passes the crop rule (checked over every window)."""
    return sorted(
        u for u in uids if not crops.has_valid_window(load_mask(u), size, share, min_pixels)
    )
