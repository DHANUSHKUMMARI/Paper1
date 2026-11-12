"""Offline stand-ins for Kaggle and Hugging Face, serving small synthetic archives.

Used by the tests and the notebook dry runs, so the download logic (space checks, resume,
verification, ``_DONE``) runs in CI without network access. The archive layouts imitate the
real sources only as far as the download code depends on them; the real layouts are
recorded by the data probe.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.data.sources import SpaceCheck

GENIMAGE_SLUG = "vtphatt2/GenImage-stable-diffusion-v1-4"
DIFFSEG_REPO = "Chaos2629/Diffseg30k"
SID_REPO = "saberzl/SID_Set"
SD14_PREFIX = "stable_diffusion_v_1_4/imagenet_ai_0419_sdv4"


def _png(rng: np.random.Generator, size: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _jpeg(rng: np.random.Generator, size: int = 32, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _mask(values: tuple[int, ...], size: int = 32) -> bytes:
    arr = np.zeros((size, size), dtype=np.uint8)
    for k, v in enumerate(values):
        arr[2 + 8 * k : 8 + 8 * k, 4:20] = v
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def make_genimage_sd14_zip(
    path: Path, train_per_class: int, val_per_class: int = 2, seed: int = 0
) -> Path:
    """A zip shaped like the SD v1.4 Kaggle archive: <prefix>/{train,val}/{ai,nature}/..."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for split, n in (("train", train_per_class), ("val", val_per_class)):
            for i in range(n):
                zf.writestr(f"{SD14_PREFIX}/{split}/ai/{i:06d}_sdv4_00000.png", _png(rng))
                zf.writestr(
                    f"{SD14_PREFIX}/{split}/nature/ILSVRC2012_{split}_{i:08d}.JPEG", _jpeg(rng)
                )
        zf.writestr(f"{SD14_PREFIX}/README.txt", b"not an image")
    return path


def make_diffseg_zip(path: Path, split: str, n: int, seed: int = 0) -> Path:
    """A zip with <split>/images/*.png and <split>/masks/*.png (values 0-8)."""
    rng = np.random.default_rng(seed)
    values = {"train": (1, 3, 5), "validation": (2, 6, 8)}.get(split, (1,))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            zf.writestr(f"{split}/images/{split}_{i:06d}.png", _png(rng))
            zf.writestr(f"{split}/masks/{split}_{i:06d}.png", _mask(values if i else ()))
    return path


def make_sid_parquet(
    folder: Path,
    split: str,
    shards: int,
    per_label_per_shard: int,
    seed: int = 0,
    grouped: bool = True,
) -> list[Path]:
    """Parquet shards with columns image{bytes,path}, mask{bytes,path}|null, label, img_id.

    ``grouped=True`` puts one label per shard (as in many HF exports), which the subset
    selection must handle.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(shards):
        labels = (
            [s % 3] * (3 * per_label_per_shard)
            if grouped
            else [lab for lab in (0, 1, 2) for _ in range(per_label_per_shard)]
        )
        rows = {"image": [], "mask": [], "label": [], "img_id": []}
        for i, lab in enumerate(labels):
            iid = f"{split}_{s:03d}_{i:04d}"
            data = _jpeg(rng) if lab == 0 else _png(rng)
            rows["image"].append({"bytes": data, "path": f"{iid}.img"})
            rows["mask"].append(
                {"bytes": _mask((255,)), "path": f"{iid}_m.png"} if lab == 2 else None
            )
            rows["label"].append(lab)
            rows["img_id"].append(iid)
        path = folder / f"data/{split}-{s:05d}-of-{shards:05d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(rows), path, row_group_size=4)
        paths.append(path)
    return paths


class FakeSources:
    """Serves local files as if they were Kaggle datasets and Hugging Face repositories.

    ``fail_after`` makes the first transfer of a file stop after that many bytes (an
    interrupted session); ``calls`` records every method call.
    """

    def __init__(
        self,
        kaggle: dict[str, Path] | None = None,
        hf: dict[str, dict[str, Path]] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.kaggle = kaggle or {}
        self.hf = hf or {}
        self.fail_after = fail_after
        self._failed: set[str] = set()
        self.calls: list[str] = []

    def _copy(self, key: str, src: Path, part: Path, start: int) -> None:
        with open(src, "rb") as fin, open(part, "ab" if start else "wb") as fout:
            fin.seek(start)
            if self.fail_after is not None and key not in self._failed:
                self._failed.add(key)
                fout.write(fin.read(self.fail_after))
                raise ConnectionError(f"simulated interruption of {key}")
            shutil.copyfileobj(fin, fout)

    def kaggle_download(self, slug: str, dest: Path, before_write: SpaceCheck) -> int:
        self.calls.append(f"kaggle_download {slug}")
        src = self.kaggle[slug]
        part = Path(dest).with_name(Path(dest).name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        have = part.stat().st_size if part.exists() else 0
        total = src.stat().st_size
        before_write(total - have, total)
        self._copy(slug, src, part, have)
        os.replace(part, dest)
        return total

    def hf_file_info(self, repo: str, filenames: list[str]) -> dict[str, dict[str, Any]]:
        self.calls.append(f"hf_file_info {repo}")
        out = {}
        for name in filenames:
            src = self.hf[repo][name]
            out[name] = {
                "size": src.stat().st_size,
                "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            }
        return out

    def hf_list_files(self, repo: str) -> dict[str, int]:
        self.calls.append(f"hf_list_files {repo}")
        return {name: p.stat().st_size for name, p in sorted(self.hf[repo].items())}

    def hf_download(self, repo: str, filename: str, dest_dir: Path) -> Path:
        self.calls.append(f"hf_download {repo} {filename}")
        src = self.hf[repo][filename]
        dest = Path(dest_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".incomplete")
        have = part.stat().st_size if part.exists() else 0
        self._copy(f"{repo}/{filename}", src, part, have)
        os.replace(part, dest)
        return dest

    def hf_parquet_column(self, repo: str, filename: str, column: str) -> list[Any]:
        import pyarrow.parquet as pq

        self.calls.append(f"hf_parquet_column {repo} {filename}")
        return pq.read_table(self.hf[repo][filename], columns=[column]).column(column).to_pylist()


def make_fake_world(
    remote: Path,
    *,
    sd14_per_class: int = 12,
    diffseg: int = 6,
    sid_shards: int = 6,
    sid_per_shard: int = 4,
    fail_after: int | None = None,
) -> FakeSources:
    """Build all fake remote datasets under ``remote`` and return sources serving them."""
    remote = Path(remote)
    kaggle = {GENIMAGE_SLUG: make_genimage_sd14_zip(remote / "kaggle/sd14.zip", sd14_per_class)}
    diff = {
        "train.zip": make_diffseg_zip(remote / "diffseg/train.zip", "train", diffseg),
        "validation.zip": make_diffseg_zip(
            remote / "diffseg/validation.zip", "validation", max(2, diffseg // 2), seed=1
        ),
        "README.md": remote / "diffseg/README.md",
    }
    diff["README.md"].write_text("# DiffSeg30k (fake)\n")
    sid_root = remote / "sid"
    sid = {}
    for split, seed in (("train", 0), ("validation", 1)):
        for p in make_sid_parquet(sid_root, split, sid_shards, sid_per_shard, seed):
            sid[p.relative_to(sid_root).as_posix()] = p
    return FakeSources(kaggle=kaggle, hf={DIFFSEG_REPO: diff, SID_REPO: sid}, fail_after=fail_after)


GENIMAGE_GENERATOR_DIRS = (
    "imagenet_ai_0508_adm",
    "imagenet_ai_0419_biggan",
    "imagenet_glide",
    "imagenet_midjourney",
    "imagenet_ai_0419_sdv4",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_wukong",
)


def make_genimage_validation(folder: Path, per_class: int = 2, seed: int = 0) -> int:
    """A complete fake GenImage Validation folder (8 generators); returns the image count."""
    rng = np.random.default_rng(seed)
    for gen in GENIMAGE_GENERATOR_DIRS:
        for i in range(per_class):
            ai = folder / gen / "val" / "ai" / f"{i:03d}_{gen}.PNG"
            nature = folder / gen / "val" / "nature" / f"ILSVRC2012_val_{i:08d}.JPEG"
            for path, data in ((ai, _png(rng)), (nature, _jpeg(rng))):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    (folder / "_DONE").write_text("vtphatt2/GenImage-Validation")  # the old downloader's marker
    return len(GENIMAGE_GENERATOR_DIRS) * 2 * per_class


def dry_run_overrides(val_images: int) -> dict[str, dict[str, Any]]:
    """Registry parameters scaled to the fake world (tests and notebook dry runs)."""
    return {
        "genimage_val": {"expected_images": val_images},
        "genimage_sd14": {"per_class": 5, "shard_images": 4, "subset_gb_estimate": 0.001},
        "sidset": {"caps": {"train": 6, "validation": 3}},
    }


SID_LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}


def make_sid_legacy(folder: Path, caps: dict[str, int], seed: int = 0) -> Path:
    """SID-Set as the retired downloader left it: loose files plus ``<split>_manifest.csv``.

    ``<split>/<label_name>/images/<id>.<ext>``, tampered masks in ``.../masks/<id>_mask.png``;
    manifest columns image, mask, label, label_name (paths relative to ``folder``).
    """
    import csv

    rng = np.random.default_rng(seed)
    folder = Path(folder)
    for split, cap in caps.items():
        rows = []
        for lab, name in SID_LABEL_NAMES.items():
            for i in range(cap):
                iid = f"{split}_{name}_{i:05d}"
                ext, data = (".jpg", _jpeg(rng)) if lab == 0 else (".png", _png(rng))
                img = f"{split}/{name}/images/{iid}{ext}"
                (folder / img).parent.mkdir(parents=True, exist_ok=True)
                (folder / img).write_bytes(data)
                mask = ""
                if lab == 2:
                    mask = f"{split}/{name}/masks/{iid}_mask.png"
                    (folder / mask).parent.mkdir(parents=True, exist_ok=True)
                    (folder / mask).write_bytes(_mask((255,)))
                rows.append([img, mask, lab, name])
        with open(folder / f"{split}_manifest.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "mask", "label", "label_name"])
            writer.writerows(rows)
    return folder
