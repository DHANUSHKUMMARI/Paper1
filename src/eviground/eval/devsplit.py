"""CP4 development hold-out (decisions of 15 and 16 Nov 2026).

The CP4 gate and the decision on its one allowed fix are judged only on a frozen development
split, disjoint from the images the paper reports:

- P1: a seeded 20% of each GenImage validation folder (8 generators x real / generated);
- P2: a seeded 20% of the held-out-editor images, stratified by base type (real / AI).

Images with identical bytes or pixels (a GenImage real image can appear in several generator
folders) form one group and always land on the same side. Inside each stratum the groups are
ranked by a hash of (seed, salt, group key) and taken until the stratum's 20% is reached; a group
taken for one stratum brings its members in other strata with it.

The split is frozen by its seed and salt and by the SHA-256 of its sorted uid list; the same
manifest always gives the same split, and :func:`verify` recomputes it.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SEED = 42
SALT = "cp4_dev_v1"
FRACTION = 0.20
ROLES = ("test_P1", "test_P2")


def stratum(r: Mapping[str, Any]) -> str | None:
    """``P1/<generator>/<label>`` or ``P2/<base_type>``; None for rows outside P1 and P2."""
    if r.get("split") != "test":
        return None
    if r.get("role") == "test_P1":
        return f"P1/{r['generator']}/{r['label']}"
    if r.get("role") == "test_P2":
        return f"P2/{r.get('base_type') or 'unknown'}"
    return None


def file_list_sha256(uids: Iterable[str]) -> str:
    """SHA-256 of the sorted uids, one per line (UTF-8, trailing newline)."""
    text = "".join(f"{u}\n" for u in sorted(uids))
    return hashlib.sha256(text.encode()).hexdigest()


def _rank(seed: int, salt: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{salt}|{key}".encode()).hexdigest()


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {u: u for u in items}

    def find(self, u: str) -> str:
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # the group key is its smallest uid


def _groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """uid -> group key: union of rows sharing ``sha256`` or ``pixel_sha256``; the key is the
    group's smallest uid."""
    uf = _UnionFind(r["uid"] for r in rows)
    for col in ("sha256", "pixel_sha256"):
        first: dict[str, str] = {}
        for r in rows:
            h = r.get(col)
            if h:
                uf.union(first.setdefault(h, r["uid"]), r["uid"])
    return {u: uf.find(u) for u in uf.parent}


def build_dev_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = SEED,
    salt: str = SALT,
    fraction: float = FRACTION,
) -> dict[str, Any]:
    """The development split of P1 and P2 from CP1 manifest rows (see the module docstring)."""
    eligible = [r for r in rows if stratum(r)]
    group = _groups(eligible)
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    members: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in eligible:
        by_stratum[stratum(r)].append(r)
        members[group[r["uid"]]].append(r)
    dev_groups: set[str] = set()
    for s in sorted(by_stratum):
        rs = by_stratum[s]
        target = int(fraction * len(rs) + 0.5)  # half rounds up
        have = sum(1 for r in rs if group[r["uid"]] in dev_groups)
        for r in sorted(rs, key=lambda r: (_rank(seed, salt, group[r["uid"]]), r["uid"])):
            if have >= target:
                break
            g = group[r["uid"]]
            if g in dev_groups:
                continue
            dev_groups.add(g)
            have += sum(1 for m in members[g] if stratum(m) == s)
    dev = sorted(r["uid"] for r in eligible if group[r["uid"]] in dev_groups)
    rest = sorted(r["uid"] for r in eligible if group[r["uid"]] not in dev_groups)
    devset = set(dev)
    crossing = sum(1 for ms in members.values() if len({stratum(m) for m in ms}) > 1)
    return {
        "seed": seed,
        "salt": salt,
        "fraction": fraction,
        "strata": {
            s: {"images": len(rs), "dev": sum(1 for r in rs if r["uid"] in devset)}
            for s, rs in sorted(by_stratum.items())
        },
        "totals": {
            p: {
                "images": sum(len(rs) for s, rs in by_stratum.items() if s.startswith(p + "/")),
                "dev": sum(
                    1 for r in eligible if r["uid"] in devset and stratum(r).startswith(p + "/")
                ),
            }
            for p in ("P1", "P2")
        },
        "duplicate_groups": sum(1 for ms in members.values() if len(ms) > 1),
        "groups_across_strata": crossing,
        "dev_uids": dev,
        "dev_sha256": file_list_sha256(dev),
        "reported_uids_count": len(rest),
        "reported_sha256": file_list_sha256(rest),
    }


def summary(split: Mapping[str, Any]) -> dict[str, Any]:
    """The split without its uid list (for records and the spec card)."""
    return {k: v for k, v in split.items() if k != "dev_uids"}


def save(split: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(split), indent=1))
    return path


def load(path: Path) -> dict[str, Any]:
    split = json.loads(Path(path).read_text())
    if file_list_sha256(split["dev_uids"]) != split["dev_sha256"]:
        raise ValueError(f"{path}: the uid list does not match its recorded SHA-256")
    return split


def verify(split: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebuild the split from ``rows`` with the recorded seed, salt and fraction and compare."""
    again = build_dev_split(
        rows, seed=split["seed"], salt=split["salt"], fraction=split["fraction"]
    )
    same = again["dev_sha256"] == split["dev_sha256"]
    return {"same": same, "recorded": split["dev_sha256"], "rebuilt": again["dev_sha256"]}
