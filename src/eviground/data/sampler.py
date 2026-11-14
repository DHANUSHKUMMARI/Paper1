"""Class and source balance of the training sampler (Research Specification section 9, rev 49).

Unified model: 1 : 1 : 1 over real / fully generated / edited. Spec rev 49: SID-Set supplies
one third of every class. Decisions of 14 Nov 2026 after the first CP1 run: SID-Set tampered
images are left out of training (blockiness fallback) and DiffSeg30k's unedited images are real
negatives, so real = GenImage 1/3, SID-Set 1/3, COCO 1/6, DiffSeg30k 1/6; generated = GenImage
2/3, SID-Set 1/3; edited = DiffSeg30k only.
Detection-protocol model: GenImage SD v1.4 only, real : generated = 1 : 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CLASSES = ("real", "generated", "edited")

SHARES: dict[str, dict[str, dict[str, float]]] = {
    "unified": {
        # decisions of 14 Nov 2026 after CP1: SID-Set tampered images left out of training
        # (blockiness fallback), DiffSeg30k's unedited images added as real negatives
        "real": {"genimage": 1 / 3, "sidset": 1 / 3, "coco": 1 / 6, "diffseg30k": 1 / 6},
        "generated": {"genimage": 2 / 3, "sidset": 1 / 3},
        "edited": {"diffseg30k": 1.0},
    },
    "detection": {
        "real": {"genimage": 1.0},
        "generated": {"genimage": 1.0},
    },
}


def quotas(n_per_class: int, profile: str = "unified") -> dict[tuple[str, str], int]:
    """Number of draws per (class, source); each class sums exactly to ``n_per_class``."""
    out: dict[tuple[str, str], int] = {}
    for label, shares in SHARES[profile].items():
        sources = sorted(shares)
        raw = {s: n_per_class * shares[s] for s in sources}
        base = {s: int(np.floor(v)) for s, v in raw.items()}
        rest = n_per_class - sum(base.values())
        for s in sorted(sources, key=lambda s: (-(raw[s] - base[s]), s))[:rest]:
            base[s] += 1
        out.update({(label, s): base[s] for s in sources})
    return out


def draw(
    sources: Sequence[str],
    labels: Sequence[str],
    n_per_class: int,
    seed: int,
    profile: str = "unified",
) -> np.ndarray:
    """Row indices drawn with replacement, uniformly within each (class, source) pool.

    ``sources`` / ``labels`` describe the eligible rows. Raises if a pool the profile needs is
    empty.
    """
    src, lab = np.asarray(sources), np.asarray(labels)
    rng = np.random.default_rng(seed)
    picks = []
    for (label, source), n in sorted(quotas(n_per_class, profile).items()):
        pool = np.flatnonzero((src == source) & (lab == label))
        if pool.size == 0:
            raise ValueError(f"sampler: no eligible {label} images from {source}")
        picks.append(rng.choice(pool, size=n, replace=True))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out
