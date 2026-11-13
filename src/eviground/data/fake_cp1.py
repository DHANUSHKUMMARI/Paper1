"""A small fake ``Paper 1/datasets`` folder with the layouts the data probe found on Drive.

Used by the CP1 tests and the dry run of ``notebooks/01_cp1_data_integrity.ipynb``. What is
copied from the real data (probe of 13 Nov 2026):

- GenImage SD v1.4 subset: tar files ``sd14_subset-0000k.tar`` with members
  ``GenImage/stable_diffusion_v_1_4/train/{ai,nature}/<name>``; ``manifest.csv``; real images
  JPEG (mostly quality 96, 4:4:4; some other qualities, 4:2:0, greyscale, a few under 256 px),
  generated images PNG of one size.
- GenImage Validation: loose files ``GenImage/<Generator>/val/{ai,nature}/`` (8 generators).
- DiffSeg30k: ``train.zip`` / ``validation.zip``, each one folder (``train_copy/``,
  ``validation_copy/``) mixing ``<id>.image.png`` (RGB) and ``<id>.mask.png`` (L, values 0-8,
  several editors per image), plus one ``.jsonl``.
- SID-Set: loose ``<split>/<class>/images/`` and ``tampered/masks/``, ``<split>_manifest.csv``;
  real JPEG at many qualities, synthetic PNG, tampered JPEG quality 75 4:2:0 (some PNG);
  train masks lossy JPEG RGB, validation masks PNG L with values near 0 and 255.
- COCO: ``val2017.zip`` (``val2017/<12 digits>.jpg``) and ``annotations_trainval2017.zip``
  (``annotations/instances_val2017.json``), served by :class:`FakeSources` as URLs.

Images are smooth random textures (each one different), so perceptual hashes behave as on
photographs.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eviground.data.fake_sources import FakeSources
from eviground.data.harmonize import SUBSAMPLING_CODE

GENERATORS = (
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
)
COCO_CATEGORIES = {
    1: "person",
    3: "car",
    62: "chair",
    63: "couch",
    67: "dining table",
    73: "laptop",
}
COCO_URLS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/"
    "annotations_trainval2017.zip",
}


def texture(seed: int, width: int, height: int, mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(seed)
    # upscale factor 7.3: not a multiple of 8, so the texture has no 8x8 grid (blockiness ~1.0)
    small = rng.integers(
        0, 256, (max(2, int(height / 7.3)), max(2, int(width / 7.3)), 3), dtype=np.uint8
    )
    img = Image.fromarray(small).resize((width, height), Image.Resampling.BICUBIC)
    return img.convert(mode)


def jpeg(img: Image.Image, quality: int, subsampling: str | None = "4:4:4") -> bytes:
    buf = io.BytesIO()
    kw: dict[str, Any] = {"quality": quality}
    if img.mode == "RGB" and subsampling:
        kw["subsampling"] = SUBSAMPLING_CODE[subsampling]
    img.save(buf, "JPEG", **kw)
    return buf.getvalue()


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def _mask(width: int, height: int, editors: list[int], seed: int, mode: str = "L") -> Image.Image:
    """Rectangles with the editor values; together they cover roughly 15-40% of the image."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((height, width), np.uint8)
    for v in editors:
        w, h = (
            int(rng.integers(width // 4, width // 2)),
            int(rng.integers(height // 4, height // 2)),
        )
        x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
        arr[y : y + h, x : x + w] = v
    return Image.fromarray(arr).convert(mode)


class _Seeds:
    def __init__(self, start: int) -> None:
        self.n = start

    def __call__(self) -> int:
        self.n += 1
        return self.n


def _write_done(folder: Path, key: str, method: str, summary: dict[str, Any]) -> None:
    from dataclasses import replace

    from eviground.data.download import write_done
    from eviground.data.registry import REGISTRY

    write_done(folder, replace(REGISTRY[key], method=method), summary)


def make_sd14_subset(folder: Path, per_class: int, seed: _Seeds, shard: int = 6) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    items = []
    qualities = [(96, "4:4:4")] * 6 + [(75, "4:2:0"), (80, "4:2:0"), (90, "4:4:4")]
    for i in range(per_class):
        s = seed()
        if i == 1:  # greyscale ImageNet image
            data = jpeg(texture(s, 300, 280, "L"), 96)
        elif i == 2:  # too small for a 256 crop
            data = jpeg(texture(s, 200, 150), 96)
        else:
            q, ss = qualities[i % len(qualities)]
            data = jpeg(texture(s, 300 + 8 * (i % 3), 270 + 4 * (i % 4)), q, ss)
        items.append(("nature", f"n{1440764 + i:08d}_{i}.JPEG", data))
        items.append(("ai", f"{i:03d}_sdv4_{i:05d}.png", png(texture(seed(), 320, 320))))
    for k in range(0, len(items), shard):
        name = f"sd14_subset-{k // shard:05d}.tar"
        with tarfile.open(folder / name, "w") as tf:
            for class_dir, fname, data in items[k : k + shard]:
                info = tarfile.TarInfo(f"GenImage/stable_diffusion_v_1_4/train/{class_dir}/{fname}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
                rows.append(
                    [k // shard, fname, class_dir, 0 if class_dir == "nature" else 1, len(data), ""]
                )
    with open(folder / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shard", "name", "class_dir", "label", "bytes", "sha256"])
        w.writerows(rows)
    _write_done(folder, "genimage_sd14", "kaggle_subset", {"images": len(rows)})


def make_validation(folder: Path, per_class: int, seed: _Seeds) -> None:
    for gen in GENERATORS:
        for class_dir in ("ai", "nature"):
            d = folder / "GenImage" / gen / "val" / class_dir
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                if class_dir == "nature":
                    (d / f"ILSVRC2012_val_{i:08d}.JPEG").write_bytes(
                        jpeg(texture(seed(), 300, 270), 96)
                    )
                else:
                    (d / f"{i}_{gen.lower()}_{i}.png").write_bytes(png(texture(seed(), 256, 256)))
    from eviground.data.download import _count_validation
    from eviground.data.registry import REGISTRY

    summary = _count_validation(folder, REGISTRY["genimage_val"].params["class_dirs"])
    _write_done(folder, "genimage_val", "verify_folder", summary)


# DiffSeg30k editor sets per image: training editors, hold-out (2 Kolors, 6 Glide), mixed
DIFFSEG_TRAIN_SETS = [[1], [3, 5], [4], [7, 8], [5], [1, 4, 8], [2], [6, 3], []]
DIFFSEG_VAL_SETS = [[2], [6], [6, 1], [3], [2, 7, 5], [8]]


def make_diffseg(folder: Path, n_train: int, n_val: int, seed: _Seeds) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for split, n, sets in (
        ("train", n_train, DIFFSEG_TRAIN_SETS),
        ("validation", n_val, DIFFSEG_VAL_SETS),
    ):
        path = folder / f"{split}.zip"
        meta = []
        with zipfile.ZipFile(path, "w") as zf:
            for i in range(n):
                s = seed()
                ident = f"{s * 2654435761 % 2**32:08x}"
                w, h = [(320, 288), (288, 288), (256, 256), (240, 300)][i % 4]
                editors = sets[i % len(sets)]
                zf.writestr(f"{split}_copy/{ident}.image.png", png(texture(s, w, h)))
                zf.writestr(f"{split}_copy/{ident}.mask.png", png(_mask(w, h, editors, s)))
                meta.append({"id": ident, "edits": editors})
            zf.writestr(
                f"{split}_copy/metadata.jsonl", "\n".join(json.dumps(m) for m in meta) + "\n"
            )
        size = path.stat().st_size
        from eviground.data.storage import sha256_file

        (folder / f"{split}.zip.verified.json").write_text(
            json.dumps({"bytes": size, "sha256": sha256_file(path)})
        )
    _write_done(folder, "diffseg30k", "hf_files", {"files": ["train.zip", "validation.zip"]})


SID_CLASSES = {0: "real", 1: "full_synthetic", 2: "tampered"}


def make_sidset(folder: Path, caps: dict[str, int], seed: _Seeds) -> None:
    real_q = [(99, "4:2:2"), (90, "4:2:0"), (97, "4:4:4"), (75, "4:2:0"), (70, "4:2:0")]
    for split, cap in caps.items():
        rows = []
        for lab, name in SID_CLASSES.items():
            for i in range(cap):
                s = seed()
                iid = f"{split}_{name}_{i:05d}"
                if lab == 0:
                    q, ss = real_q[i % len(real_q)]
                    data, ext = jpeg(texture(s, 320, 288), q, ss), ".jpg"
                elif lab == 1:
                    data, ext = png(texture(s, 320, 320)), ".png"
                elif i % 5 == 4:
                    data, ext = png(texture(s, 320, 320)), ".png"
                else:
                    data, ext = jpeg(texture(s, 320, 320), 75, "4:2:0"), ".jpg"
                img = f"{split}/{name}/images/{iid}{ext}"
                (folder / img).parent.mkdir(parents=True, exist_ok=True)
                (folder / img).write_bytes(data)
                mask = ""
                if lab == 2:
                    m = _mask(320, 320, [255], s)
                    if split == "train":  # lossy RGB JPEG masks
                        mask, mdata = (
                            f"{split}/{name}/masks/{iid}_mask.jpg",
                            jpeg(m.convert("RGB"), 75),
                        )
                    else:
                        mask, mdata = f"{split}/{name}/masks/{iid}_mask.png", png(m)
                    (folder / mask).parent.mkdir(parents=True, exist_ok=True)
                    (folder / mask).write_bytes(mdata)
                rows.append([img, mask, lab, name])
        with open(folder / f"{split}_manifest.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "mask", "label", "label_name"])
            w.writerows(rows)
    _write_done(folder, "sidset", "verify_sid_folder", {"caps": caps})


def make_coco_zips(remote: Path, n_images: int, seed: _Seeds) -> dict[str, Path]:
    """val2017.zip and annotations zip; every image has one or two annotated objects."""
    remote.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    images, anns = [], []
    with zipfile.ZipFile(remote / "val2017.zip", "w") as zf:
        for i in range(n_images):
            iid = 139 + 37 * i
            w, h = [(640, 480), (480, 640), (500, 375), (640, 427)][i % 4]
            zf.writestr(f"val2017/{iid:012d}.jpg", jpeg(texture(seed(), w, h), 90, "4:2:0"))
            images.append({"id": iid, "file_name": f"{iid:012d}.jpg", "width": w, "height": h})
            cats = list(COCO_CATEGORIES)
            for k in range(1 + i % 2):
                cat = cats[int(rng.integers(0, len(cats)))]
                share = float(rng.choice([0.01, 0.1, 0.3]))
                anns.append(
                    {
                        "id": len(anns) + 1,
                        "image_id": iid,
                        "category_id": cat,
                        "area": share * w * h,
                        "iscrowd": int(k == 1 and i % 7 == 0),
                        "bbox": [0, 0, 10, 10],
                    }
                )
    inst = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": k, "name": v} for k, v in COCO_CATEGORIES.items()],
    }
    with zipfile.ZipFile(remote / "annotations_trainval2017.zip", "w") as zf:
        zf.writestr("annotations/instances_val2017.json", json.dumps(inst))
        zf.writestr("annotations/captions_val2017.json", "{}")
    return {name: remote / name for name in COCO_URLS}


def make_cp1_world(
    datasets: Path,
    remote: Path,
    *,
    sd14_per_class: int = 12,
    val_per_class: int = 2,
    diffseg: tuple[int, int] = (18, 12),
    sid_caps: dict[str, int] | None = None,
    coco_images: int = 40,
    seed: int = 1000,
) -> FakeSources:
    """Write the fake Drive datasets (with ``_DONE`` markers, as on the real Drive) and return
    the fake sources that serve the COCO zips."""
    s = _Seeds(seed)
    datasets = Path(datasets)
    make_sd14_subset(datasets / "01_GenImage/SD_v1.4_subset", sd14_per_class, s)
    make_validation(datasets / "01_GenImage/Validation", val_per_class, s)
    make_diffseg(datasets / "02_DiffSeg30k", *diffseg, s)
    make_sidset(datasets / "03_SID-Set", sid_caps or {"train": 10, "validation": 5}, s)
    zips = make_coco_zips(Path(remote) / "coco", coco_images, s)
    return FakeSources(urls={COCO_URLS[k]: v for k, v in zips.items()})


def cp1_overrides(coco_images: int, claims_n: int = 6) -> dict[str, dict[str, Any]]:
    """Registry parameters for the fake world (tiny COCO)."""
    from eviground.data.registry import REGISTRY

    files = json.loads(json.dumps(REGISTRY["coco_val2017"].params["files"]))
    files["val2017.zip"]["members"]["count"] = coco_images
    for f in files.values():
        f["gb_estimate"] = 0.001
    claims = {**REGISTRY["coco_val2017"].params["claims_edit"], "n": claims_n}
    return {"coco_val2017": {"files": files, "claims_edit": claims, "negatives_size": 288}}
