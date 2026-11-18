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
LABELS = ("generated", "real")
P1_GENERATORS = 8


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
    generators: int | None = None,
) -> dict[str, Any]:
    """P1 rows outside the dev split, ``per_label`` real and generated per generator.

    Every generator in the P1 rows must keep both labels after the dev split is removed (a
    missing stratum is an error, not a smaller subset); with ``generators``, their number is
    checked too (8 for GenImage)."""
    dev = set(dev_uids)
    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    gens: set[str] = set()
    for r in manifest_rows:
        if r.get("role") == "test_P1" and r.get("split") == "test":
            gens.add(r["generator"])
            if r["uid"] not in dev:
                strata.setdefault((r["generator"], r["label"]), []).append(r)
    missing = [f"{g}/{lab}" for g in sorted(gens) for lab in LABELS if (g, lab) not in strata]
    if missing or (generators is not None and len(gens) != generators):
        raise ValueError(f"P1 strata missing: {missing}; generators found: {sorted(gens)}")
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
        return dict.fromkeys(available, half)
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


def intact(subset: Mapping[str, Any]) -> bool:
    """True when a stored subset's hash and size match its own uid list."""
    uids = subset.get("uids") or []
    return subset.get("sha256") == file_list_sha256(uids) and subset.get("images") == len(uids)


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


# --------------------------------------------------------------------------------------
# P6: explanation quality (spec section 10, P6 row; Dhanush, 18 Nov 2026)
# --------------------------------------------------------------------------------------

P6_SALT = "p6_v1"
P6_PARTS = {  # the spec's mix: 1,500 images
    "diffseg_heldout": 400,
    "sidset_tampered": 400,
    "brgen": 400,
    "real": 150,
    "generated": 150,
}
P6_EDITED = ("diffseg_heldout", "sidset_tampered", "brgen")
P6_REAL_SOURCES = ("genimage", "diffseg_unedited", "chameleon")
P6_MLLM = 500
P6_DELETION = 300
NEAR_DUP = 6  # pHash Hamming distance (the CP1 leakage rule)


class TrainingIndex:
    """Exact hashes and pHashes of the training pool, for the leakage check of extra images."""

    def __init__(self, train_rows: Iterable[Mapping[str, Any]]) -> None:
        from eviground.data import hashing

        rows = list(train_rows)
        self.exact = {r[k] for r in rows for k in ("sha256", "pixel_sha256") if r.get(k)}
        self.phash = hashing.as_uint64(r["phash"] for r in rows if r.get("phash") not in (None, ""))

    def clean(self, row: Mapping[str, Any]) -> bool:
        """No exact match and no pHash within :data:`NEAR_DUP` of a training image."""
        from eviground.data import hashing

        if any(row.get(k) in self.exact for k in ("sha256", "pixel_sha256") if row.get(k)):
            return False
        if row.get("phash") in (None, "") or self.phash.size == 0:
            return True
        dist, _ = hashing.min_hamming(hashing.as_uint64([row["phash"]]), self.phash)
        return int(dist[0]) > NEAR_DUP


def take_clean(
    rows: Iterable[Mapping[str, Any]], n: int, seed: int, salt: str, index: TrainingIndex
) -> tuple[list[Mapping[str, Any]], int]:
    """The first ``n`` rows by the seeded rank that pass the training check, and how many were
    skipped on the way."""
    chosen, skipped = [], 0
    for r in sorted(rows, key=lambda r: (_rank(seed, salt, r["uid"]), r["uid"])):
        if len(chosen) == n:
            break
        if index.clean(r):
            chosen.append(r)
        else:
            skipped += 1
    return chosen, skipped


def chameleon_candidates(
    rows: Iterable[Mapping[str, Any]], k: int, seed: int = SEED, salt: str = P6_SALT
) -> list[Mapping[str, Any]]:
    """The first ``k`` Chameleon real rows in the order :func:`p6_subset` reads them, so only
    these need hashing (the pick is the same while at least 50 of them pass the training check)."""
    return take(rows, k, seed, f"{salt}/real/ch")


def p6_pools(
    manifest_rows: Iterable[Mapping[str, Any]],
    dev_uids: Iterable[str],
    p1_uids: Iterable[str],
) -> dict[str, list[Mapping[str, Any]]]:
    """The manifest pools of P6 (BR-Gen and Chameleon come from outside the manifest)."""
    dev, p1 = set(dev_uids), set(p1_uids)
    pools: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in manifest_rows:
        src, role, label, uid = r.get("source"), r.get("role"), r.get("label"), r["uid"]
        if role == "test_P2" and uid not in dev:
            pools["diffseg_heldout"].append(r)
        elif src == "sidset" and role == "test_P3b":
            pools["sidset_tampered" if label == "edited" else f"sidset_{label}"].append(r)
        elif role == "test_P1" and uid not in dev:
            if label == "generated":
                pools["generated"].append(r)
            elif uid not in p1:
                pools["genimage_real"].append(r)
        elif src == "diffseg30k" and role == "unused_validation" and not r.get("editors"):
            pools["diffseg_unedited"].append(r)
    return dict(pools)


def even_over(rows: Sequence[Mapping[str, Any]], key: str, n: int, seed: int, salt: str) -> list:
    """``n`` rows spread as evenly as possible over the values of ``key`` (sorted; the first
    values take the remainder)."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    names = sorted(groups)
    base, extra = divmod(n, len(names))
    out = []
    for i, g in enumerate(names):
        out += take(groups[g], base + (1 if i < extra else 0), seed, f"{salt}/{g}")
    return out


def p6_subset(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    brgen_items: Sequence[str],
    chameleon_real: Sequence[Mapping[str, Any]],
    index: TrainingIndex,
    *,
    seed: int = SEED,
    salt: str = P6_SALT,
) -> dict[str, Any]:
    """The P6 set (``P6_PARTS``) and its two nested subsets.

    ``brgen_items``: the frozen BR-Gen subset's forged members (SDXL is dropped here);
    ``chameleon_real``: Chameleon real rows outside P4, with their hashes. The 150 real images
    are about 50 per source; DiffSeg30k unedited validation images are topped up from SID-Set
    validation real when short. Every real image passes the training check."""
    from eviground.data import brgen

    parts: dict[str, list[Mapping[str, Any]]] = {}
    parts["diffseg_heldout"] = take(
        pools.get("diffseg_heldout", []), P6_PARTS["diffseg_heldout"], seed, f"{salt}/p2"
    )
    parts["sidset_tampered"] = take(
        pools.get("sidset_tampered", []), P6_PARTS["sidset_tampered"], seed, f"{salt}/sid"
    )
    br = [{"uid": i} for i in brgen_items if brgen.parse_member(i).get("method") != "SDXL"]
    parts["brgen"] = take(br, P6_PARTS["brgen"], seed, f"{salt}/brgen")
    parts["generated"] = even_over(
        pools.get("generated", []), "generator", P6_PARTS["generated"], seed, f"{salt}/gen"
    )
    per = P6_PARTS["real"] // len(P6_REAL_SOURCES)
    real, skipped, top_up = {}, {}, 0
    real["genimage"], skipped["genimage"] = take_clean(
        pools.get("genimage_real", []), per, seed, f"{salt}/real/gi", index
    )
    real["diffseg_unedited"], skipped["diffseg_unedited"] = take_clean(
        pools.get("diffseg_unedited", []), per, seed, f"{salt}/real/ds", index
    )
    if len(real["diffseg_unedited"]) < per:
        extra, skipped["sidset_real"] = take_clean(
            pools.get("sidset_real", []),
            per - len(real["diffseg_unedited"]),
            seed,
            f"{salt}/real/sid",
            index,
        )
        top_up = len(extra)
        real["diffseg_unedited"] = [*real["diffseg_unedited"], *extra]
    real["chameleon"], skipped["chameleon"] = take_clean(
        chameleon_real, per, seed, f"{salt}/real/ch", index
    )
    parts["real"] = [r for k in P6_REAL_SOURCES for r in real[k]]
    counts = {k: len(v) for k, v in parts.items()}
    short = {k: (counts[k], n) for k, n in P6_PARTS.items() if counts[k] != n}
    if short:
        raise ValueError(f"P6 parts short (got, wanted): {short}")
    part_of = {r["uid"]: k for k, v in parts.items() for r in v}
    all_rows = [r for v in parts.values() for r in v]
    p6 = _record(
        "P6",
        all_rows,
        seed,
        salt,
        per_part=counts,
        real_sources={k: len(v) for k, v in real.items()},
        sidset_top_up=top_up,
        skipped_near_training=skipped,
        part_of=part_of,
    )
    mllm_counts = balanced_counts(counts, P6_MLLM)
    mllm = [
        r for k in sorted(mllm_counts) for r in take(parts[k], mllm_counts[k], seed, f"{salt}/mllm")
    ]
    del_counts = balanced_counts({k: counts[k] for k in P6_EDITED}, P6_DELETION)
    deletion = [
        r for k in sorted(del_counts) for r in take(parts[k], del_counts[k], seed, f"{salt}/del")
    ]
    return {
        "P6": p6,
        "P6_mllm": _record("P6_mllm", mllm, seed, f"{salt}/mllm", per_part=mllm_counts),
        "P6_deletion": _record("P6_deletion", deletion, seed, f"{salt}/del", per_part=del_counts),
    }
