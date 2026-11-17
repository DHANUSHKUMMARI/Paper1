"""Whole images for CP4 evaluation (decisions of 15-16 Nov 2026).

- :func:`shard_items`: validation images from the packed shards, whole (as harmonized for
  training), with their 1-bit edit masks: the localization diagnosis and the false "edited" rate
  on in-distribution real images. No test image is in the shards.
- :func:`file_items`: P1 and P2 images from their original files (CP1 manifest rows), raw or
  harmonized (P1: the GenImage training compression rule applied to the test files, no resizing,
  nothing skipped); P2 with its edit mask (AI base read as 0, as in training) and per-pixel
  editor values (held-out-editor pixels).
- :func:`stage_files`: copies the files of the chosen rows to the local disk with parallel reads
  (each read through Colab's Drive mount costs about a quarter of a second).
"""

from __future__ import annotations

import io
import shutil
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.data import diffseg_mask, harmonize
from eviground.data.datasets import ShardReader
from eviground.eval.tiled import Item

META_KEYS = ("source", "label", "generator", "editors", "base_type", "role", "split")


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


def _meta(r: Mapping[str, Any]) -> dict[str, Any]:
    meta = {k: r.get(k) for k in META_KEYS}
    eds = r.get("editors")
    meta["editors"] = [e for e in str(eds).split(",") if e] if eds not in (None, "") else []
    return meta


def shard_items(shard_dir: Path, rows: Sequence[Mapping[str, Any]] | None = None) -> Iterator[Item]:
    """Whole images of ``rows`` (default: every validation row) from the shards."""
    reader = ShardReader(shard_dir)
    rows = [r for r in reader.rows if r["split"] == "val"] if rows is None else rows
    for r in rows:
        mask = reader.mask(r) if r["label"] == "edited" else None
        yield Item(r["uid"], np.asarray(reader.image(r)), mask, None, _meta(r))


def p1_harmonized_bytes(
    r: Mapping[str, Any], data: bytes, dist: harmonize.QualityDistribution
) -> bytes:
    """The GenImage training rule on a P1 test file: generated PNGs compressed with the real
    images' quality distribution (seeded by uid); real JPEGs and anything else untouched."""
    from eviground.data.imageinfo import header_info

    info = header_info(data)
    plan = harmonize.plan_image(
        "genimage",  # the training rule of GenImage, applied to its validation (P1) files
        r["label"],
        info,
        r["uid"],
        dist,
        min_side=None,
    )
    if plan["action"] != "compress":
        return data
    out = harmonize.apply_plan(data, plan)
    return data if out is None else out


def file_items(
    rows: Sequence[Mapping[str, Any]],
    *,
    version: str = "raw",
    dist: harmonize.QualityDistribution | None = None,
) -> Iterator[Item]:
    """P1 / P2 images from their files. ``version``: 'raw' or 'harmonized' (P1)."""
    for r in rows:
        data = Path(r["path"]).read_bytes()
        if version == "harmonized" and r.get("role") == "test_P1":
            if dist is None:
                raise ValueError("the harmonized version needs the GenImage quality distribution")
            data = p1_harmonized_bytes(r, data, dist)
        img = _decode(data)
        mask = editors = None
        if r.get("mask_path"):
            with Image.open(r["mask_path"]) as m:
                raw = np.asarray(m.convert("L"))
            base = r.get("mask_base")
            base = None if base in (None, "") else int(float(base))
            mask = diffseg_mask.edit_mask(raw, base) > 0
            editors = np.where(mask, raw, 0)
        meta = _meta(r)
        meta["version"] = version
        yield Item(r["uid"], img, mask, editors, meta)


def stage_files(
    rows: Sequence[Mapping[str, Any]], dest: Path, workers: int = 16
) -> list[dict[str, Any]]:
    """Copies each row's file (and mask) under ``dest`` keyed by uid; returns the rows with the
    local paths. A file already there with the same size is not copied again."""
    dest = Path(dest)

    def one(r: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(r)
        for key in ("path", "mask_path"):
            src = r.get(key)
            if not src:
                continue
            target = dest / (r["uid"] + (".mask" if key == "mask_path" else ""))
            target = target.with_suffix(target.suffix + Path(src).suffix)
            if not (target.is_file() and target.stat().st_size == Path(src).stat().st_size):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
            out[key] = str(target)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, rows))


def dev_rows(
    manifest_rows: Sequence[Mapping[str, Any]], dev_uids: Sequence[str], role: str
) -> list[dict[str, Any]]:
    """The manifest rows of ``role`` ('test_P1' or 'test_P2') in the development split."""
    keep = set(dev_uids)
    return [dict(r) for r in manifest_rows if r.get("role") == role and r["uid"] in keep]
