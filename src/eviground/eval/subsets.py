"""Frozen evaluation subsets (Dhanush, 18 Nov 2026: compute is the constraint).

- P1: per GenImage generator, 1,250 real + 1,250 generated images drawn from the part of the
  validation folders outside the CP4 development hold-out (seed 42, salt ``p1_test_v1``).
- P4: 2,500 real + 2,500 AI-generated Chameleon images (seed 42, salt ``chameleon_v1``); if a
  class has fewer, the 5,000 are split in proportion to the class sizes.

Inside each stratum, items are ranked by SHA-256 of (seed, salt, uid) and the first ``n`` taken.
Each subset is frozen by the SHA-256 of its sorted uid list (``devsplit.file_list_sha256``) and
saved before any model is scored on it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from eviground.eval.devsplit import _rank, file_list_sha256

SEED = 42
P1_SALT = "p1_test_v1"
P4_SALT = "chameleon_v1"
P1_PER_LABEL = 1250
P4_TOTAL = 5000


def take(
    rows: Iterable[Mapping[str, Any]], n: int, seed: int, salt: str
) -> list[Mapping[str, Any]]:
    """The first ``n`` rows by the seeded rank (all of them if fewer)."""
    return sorted(rows, key=lambda r: (_rank(seed, salt, r["uid"]), r["uid"]))[:n]


def _record(name: str, chosen: Sequence[Mapping[str, Any]], seed: int, salt: str, **extra) -> dict:
    uids = sorted(r["uid"] for r in chosen)
    return {
        "name": name,
        "seed": seed,
        "salt": salt,
        "images": len(uids),
        "uids": uids,
        "sha256": file_list_sha256(uids),
        **extra,
    }


def p1_test_subset(
    manifest_rows: Iterable[Mapping[str, Any]],
    dev_uids: Iterable[str],
    *,
    per_label: int = P1_PER_LABEL,
    seed: int = SEED,
    salt: str = P1_SALT,
) -> dict[str, Any]:
    """P1 rows outside the dev split, ``per_label`` real and generated per generator."""
    dev = set(dev_uids)
    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in manifest_rows:
        if r.get("role") == "test_P1" and r.get("split") == "test" and r["uid"] not in dev:
            strata[(r["generator"], r["label"])].append(r)
    short = {f"{g}/{lab}": len(rs) for (g, lab), rs in strata.items() if len(rs) < per_label}
    if short:
        raise ValueError(f"strata with fewer than {per_label} images: {short}")
    chosen = [r for key in sorted(strata) for r in take(strata[key], per_label, seed, salt)]
    counts = {f"{g}/{lab}": per_label for g, lab in sorted(strata)}
    available = {f"{g}/{lab}": len(rs) for (g, lab), rs in sorted(strata.items())}
    return _record("P1", chosen, seed, salt, per_stratum=counts, available=available)


def balanced_counts(available: Mapping[str, int], total: int) -> dict[str, int]:
    """Half each when both classes allow it, else proportional to the class sizes."""
    half = total // 2
    if all(v >= half for v in available.values()) and len(available) == 2:
        return {k: half for k in available}
    n = sum(available.values())
    if n <= total:
        return dict(available)
    out = {k: int(total * v / n) for k, v in available.items()}
    rest = total - sum(out.values())
    for k in sorted(available, key=lambda k: -(total * available[k] / n - out[k]))[:rest]:
        out[k] += 1
    return out


def p4_subset(
    items: Iterable[Mapping[str, Any]],
    *,
    total: int = P4_TOTAL,
    seed: int = SEED,
    salt: str = P4_SALT,
) -> dict[str, Any]:
    """The P4 Chameleon subset from labelled items (``uid``, ``label``)."""
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in items:
        by_label[r["label"]].append(r)
    available = {k: len(v) for k, v in sorted(by_label.items())}
    counts = balanced_counts(available, total)
    chosen = [r for k in sorted(counts) for r in take(by_label[k], counts[k], seed, salt)]
    rule = "balanced" if len(set(counts.values())) == 1 else "stratified (a class is short)"
    return _record("P4", chosen, seed, salt, per_class=counts, available=available, rule=rule)


def summary(subset: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in subset.items() if k != "uids"}


def save(subsets: Mapping[str, Mapping[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(subsets), indent=1))
    return path


def label_counts(subset: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    keep = set(subset["uids"])
    return dict(sorted(Counter(r["label"] for r in rows if r["uid"] in keep).items()))
