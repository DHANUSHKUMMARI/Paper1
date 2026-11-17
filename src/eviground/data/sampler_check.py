"""The CP1 shortcut checks on the training sampler's own draws (CP4 fix, 17 Nov 2026).

The CP4 fix relabels edit-free context crops of DiffSeg30k AI-base images as ``generated``, so
some DiffSeg30k files move between classes. Before any GPU time, Dhanush asked for the CP1
shortcut checks again, on the sampler as the trainer will draw it:

- the CP1 metadata classifier (a lookup table on exact file facts: format, JPEG quality and the
  crop's width and height, as CP1 defined it), 5-fold balanced accuracy, real vs not real
  (criterion <= 0.60) and over the three classes; the stored image's own size is reported
  beside it as a diagnostic, as CP1 reported the original size (not a criterion);
- the quality-75 share per class and per class and source (:func:`cp1.quality_75_share`);
- the source mix per class.

Each is computed for the CP3 sampler (no context crops) and the fixed one on the same epochs, so
the difference shows what the fix changes. :func:`compare` flags a material change; Dhanush
decides. Everything reads the shard index and edit masks only (CPU, no model).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from eviground.data import cp1, sampler, shortcut
from eviground.data.datasets import CONTEXT_GENERATED, ForensicDataset

# a change beyond these is "material": stop and ask (Dhanush, 17 Nov 2026)
MATERIAL = {"balanced_accuracy": 0.02, "q75_share": 0.03, "source_share": 0.03}
FEATURES = ("format", "jpeg_quality", "crop_w", "crop_h")  # the CP1 criterion features


def plan_records(ds: ForensicDataset, epochs: Iterable[int]) -> list[dict[str, Any]]:
    """One record per training draw of ``epochs``: the label after the context relabel, the
    source and the stored file's facts (from the shard index)."""
    out = []
    for ep in epochs:
        ds.set_epoch(ep)
        for i, spec in ds.items:
            r = ds.rows[i]
            generated = len(spec) > 5 and spec[5] == CONTEXT_GENERATED
            fmt = r.get("img_format")
            left, top, right, bottom = spec[:4]
            out.append(
                {
                    "epoch": ep,
                    "crop_w": right - left,
                    "crop_h": bottom - top,
                    "label": "generated" if generated else r["label"],
                    "source": r["source"],
                    "context": spec[5] if len(spec) > 5 else None,
                    "h_format": "jpeg" if fmt == "jpg" else fmt,
                    "h_quality": r.get("h_quality"),
                    "h_subsampling": r.get("h_subsampling"),
                    "width": r.get("width"),
                    "height": r.get("height"),
                }
            )
    return out


def source_mix(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Per class, the share of draws from each source."""
    out: dict[str, dict[str, float]] = {}
    for c in sampler.CLASSES:
        srcs = Counter(r["source"] for r in records if r["label"] == c)
        n = sum(srcs.values())
        out[c] = {s: round(k / n, 4) for s, k in sorted(srcs.items())} if n else {}
    return out


def check(records: Sequence[Mapping[str, Any]], seed: int = 42) -> dict[str, Any]:
    """The CP1 shortcut checks on ``records`` (from :func:`plan_records`)."""
    feats = {
        "format": [r["h_format"] for r in records],
        "jpeg_quality": [r["h_quality"] for r in records],
        "crop_w": [r["crop_w"] for r in records],
        "crop_h": [r["crop_h"] for r in records],
    }
    stored_size = {"width": [r["width"] for r in records], "height": [r["height"] for r in records]}
    y3 = [r["label"] for r in records]
    yb = ["real" if y == "real" else "not_real" for y in y3]
    binary = shortcut.feature_report(feats, yb, seed=seed)
    return {
        "draws": len(records),
        "class_counts": dict(sorted(Counter(y3).items())),
        "context": dict(sorted(Counter(r["context"] for r in records if r["context"]).items())),
        "metadata_real_vs_not_real": binary,
        "metadata_three_class": shortcut.feature_report(feats, y3, seed=seed),
        "criterion_score": binary["balanced_accuracy"],
        "criterion_pass": binary["balanced_accuracy"] <= shortcut.CRITERION,
        "diagnostic_stored_size_real_vs_not_real": shortcut.feature_report(
            stored_size, yb, seed=seed
        )["balanced_accuracy"],
        "quality_75": cp1.quality_75_share(list(records)),
        "source_mix": source_mix(records),
        # the same without the edit-free context crops: the fix moves DiffSeg30k into the
        # generated class by design; materiality is judged on everything else
        "source_mix_without_context_generated": source_mix(
            [r for r in records if r["context"] != CONTEXT_GENERATED]
        ),
    }


def _number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _diffs(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    """Absolute differences of the numbers in two nested mappings (dotted keys); an entry present
    on one side only counts as 0 on the other (a source new to a class is a change)."""
    out = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, Mapping) or isinstance(y, Mapping):
            sub = _diffs(x if isinstance(x, Mapping) else {}, y if isinstance(y, Mapping) else {})
            out.update({f"{k}.{kk}": v for kk, v in sub.items()})
        elif _number(x) or _number(y):
            out[k] = round(abs(float(y if _number(y) else 0) - float(x if _number(x) else 0)), 4)
    return out


def compare(cp3: Mapping[str, Any], fix: Mapping[str, Any]) -> dict[str, Any]:
    """What the fix changes, and whether any change is material (stop and ask)."""
    acc = {
        "real_vs_not_real": round(fix["criterion_score"] - cp3["criterion_score"], 4),
        "three_class": round(
            fix["metadata_three_class"]["balanced_accuracy"]
            - cp3["metadata_three_class"]["balanced_accuracy"],
            4,
        ),
    }
    q75 = _q75_changes(cp3["quality_75"], fix["quality_75"])
    key = "source_mix_without_context_generated"
    mix = _diffs(cp3[key], fix[key])
    return {
        "thresholds": MATERIAL,
        "balanced_accuracy_change": acc,
        "quality_75_change": q75,
        "source_share_change_designed": _diffs(cp3["source_mix"], fix["source_mix"]),
        "source_share_change": mix,
        "material": _material(fix, acc, q75, mix),
    }


def _q75_changes(cp3: Mapping[str, Any], fix: Mapping[str, Any]) -> dict[str, float]:
    """Quality-75 share changes per class and per class and source. A class/source pool that
    exists on one side only (DiffSeg30k in the generated class with the fix) has no share to
    compare; it shows in the source mix instead."""
    out = {}
    for level in ("per_class", "per_class_source"):
        a, b = cp3.get(level) or {}, fix.get(level) or {}
        shared = {k: (a[k], b[k]) for k in sorted(set(a) & set(b))}
        for key, (x, y) in shared.items():
            for field in ("q75_any", "q75_420"):
                if x.get(field) is not None and y.get(field) is not None:
                    out[f"{level}.{key}.{field}"] = round(abs(y[field] - x[field]), 4)
    return out


def _material(
    fix: Mapping[str, Any],
    acc: Mapping[str, float],
    q75: Mapping[str, float],
    mix: Mapping[str, float],
) -> list[str]:
    """The changes beyond :data:`MATERIAL` (and a criterion above 0.60)."""
    out = []
    if not fix["criterion_pass"]:
        out.append(f"real vs not real {fix['criterion_score']} > {shortcut.CRITERION}")
    lim = MATERIAL
    out += [
        f"balanced accuracy {k} moved {v:+}"
        for k, v in acc.items()
        if abs(v) > lim["balanced_accuracy"]
    ]
    out += [f"quality-75 {k} moved {v}" for k, v in q75.items() if v > lim["q75_share"]]
    out += [f"source share {k} moved {v}" for k, v in mix.items() if v > lim["source_share"]]
    return out
